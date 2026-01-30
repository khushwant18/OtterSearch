"""
Flask API routes for OtterSearch web server
"""
import threading
from pathlib import Path
from flask import Flask, request, jsonify, send_file
import fitz
from .config import config
from .indexer import Indexer, indexing_status
from .searcher import HybridSearcher
from .storage import MetadataStore
from .ml_models import ModelManager
import subprocess
import platform

def create_app(ui_html: str) -> Flask:
    """Create and configure Flask application"""
    app = Flask(__name__)

    @app.route('/preview/<path:filepath>')
    def serve_preview(filepath):
        try:
            if not filepath.startswith('/'):
                filepath = '/' + filepath

            file_path = Path(filepath)
            if not file_path.exists():
                return "File not found", 404
            
            if file_path.suffix.lower() == '.pdf':
                # Generate preview of first page
                doc = fitz.open(file_path)
                page = doc[0]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for quality
                img_data = pix.tobytes("png")
                doc.close()
                
                from io import BytesIO
                return send_file(BytesIO(img_data), mimetype='image/png')
            else:
                # For images, serve directly
                return send_file(file_path)
        except Exception as e:
            return str(e), 500

    @app.route('/open/<path:filepath>')
    def open_file(filepath):
        """Open file in default application"""
        try:
            if not filepath.startswith('/'):
                filepath = '/' + filepath

            file_path = Path(filepath)
            if not file_path.exists():
                return jsonify({"error": "File not found"}), 404
            
            system = platform.system()
            if system == "Darwin":  # macOS
                subprocess.run(["open", str(file_path)])
            elif system == "Windows":
                subprocess.run(["start", str(file_path)], shell=True)
            else:  # Linux
                subprocess.run(["xdg-open", str(file_path)])
            
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/reveal/<path:filepath>')
    def reveal_in_finder(filepath):
        """Reveal file in Finder/Explorer"""
        try:
            if not filepath.startswith('/'):
                filepath = '/' + filepath
                
            file_path = Path(filepath)
            if not file_path.exists():
                return jsonify({"error": "File not found"}), 404
            
            
            system = platform.system()
            if system == "Darwin":  # macOS
                subprocess.run(["open", "-R", str(file_path)])
            elif system == "Windows":
                subprocess.run(["explorer", "/select,", str(file_path)])
            else:  # Linux
                subprocess.run(["xdg-open", str(file_path.parent)])
            
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/')
    def index():
        return ui_html

    def run_indexing(path, recursive, update_mode=False):
        try:
            indexing_status["running"] = True
            indexing_status["progress"] = "Starting..."
            indexing_status["error"] = None
            indexing_status["total"] = 0
            indexing_status["processed"] = 0

            model_manager = ModelManager()
            model_manager.unload_slm()
            
            indexer = Indexer()
            if update_mode:
                indexing_status["progress"] = "Finding new files..."
            else:
                indexing_status["progress"] = "Indexing files..."

            indexer.index_directory(path, recursive=recursive, update_mode=update_mode)
            
            with MetadataStore() as store:
                stats = store.get_stats()
                indexing_status["count"] = stats['total']
            
            indexing_status["progress"] = "Completed"
            indexing_status["running"] = False
        except Exception as e:
            indexing_status["error"] = str(e)
            indexing_status["running"] = False

    @app.route('/index', methods=['POST'])
    def index_endpoint():
        
        if indexing_status["running"]:
            return jsonify({"error": "Indexing already in progress"}), 400
        
        try:
            data = request.json
            path = Path(data.get('path', '').replace('~', str(Path.home())))
            recursive = data.get('recursive', True)
            
            if not path.exists():
                return jsonify({"error": "Path does not exist"}), 400
            
            # Start background thread
            thread = threading.Thread(target=run_indexing, args=(path, recursive))
            thread.daemon = True
            thread.start()
            
            return jsonify({"success": True, "message": "Indexing started"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/index/status')
    def index_status():
        return jsonify(indexing_status)

    @app.route('/index/all', methods=['POST'])
    def index_all_endpoint():
        """Index Documents, Desktop, and Downloads"""
        
        if indexing_status["running"]:
            return jsonify({"error": "Indexing already in progress"}), 400
        
        try:
            paths = [
                Path.home() / "Documents",
                Path.home() / "Desktop",
                Path.home() / "Downloads"
            ]
            
            # Start background thread for all paths - full index
            thread = threading.Thread(target=index_multiple_paths, args=(paths,))
            thread.daemon = True
            thread.start()
            
            return jsonify({"success": True, "message": "Indexing all directories started"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def index_multiple_paths(paths):
        """Index multiple directories sequentially (full index)"""
        for path in paths:
            if path.exists():
                indexing_status["progress"] = f"Indexing {path.name}..."
                run_indexing(path, recursive=True, update_mode=False)

    @app.route('/index/update', methods=['POST'])
    def update_index_endpoint():
        """Update index with only new files in Documents, Desktop, and Downloads"""
        
        if indexing_status["running"]:
            return jsonify({"error": "Indexing already in progress"}), 400
        
        try:
            paths = [
                Path.home() / "Documents",
                Path.home() / "Desktop",
                Path.home() / "Downloads"
            ]
            
            # Start background thread for all paths - update mode (new files only)
            thread = threading.Thread(target=index_multiple_paths_update, args=(paths,))
            thread.daemon = True
            thread.start()
            
            return jsonify({"success": True, "message": "Update started for all directories"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def index_multiple_paths_update(paths):
        """Update index for multiple directories sequentially (new files only)"""
        for path in paths:
            if path.exists():
                indexing_status["progress"] = f"Updating {path.name}..."
                run_indexing(path, recursive=True, update_mode=True)

    @app.route('/search')
    def search_endpoint():
        query = request.args.get('q', '')
        if not query:
            return jsonify({"results": []})
        
        try:
            searcher = HybridSearcher()
            results = searcher.search(query)
            
            return jsonify({
                "results": [r.to_dict() for r in results]
            })
        except Exception as e:
            print(f"Search error: {e}")
            return jsonify({"error": str(e)}), 500
        
    @app.route('/stats')
    def stats_endpoint():
        """Get current index statistics"""
        try:
            with MetadataStore() as store:
                stats = store.get_stats()
            return jsonify(stats)
        except Exception as e:
            return jsonify({"error": str(e), "total": 0, "by_type": {}}), 500

    @app.route('/file/<path:filepath>')
    def serve_file(filepath):
        try:
            file_path = Path(filepath)
            if file_path.exists() and file_path.is_file():
                return send_file(file_path)
            return "File not found", 404
        except Exception as e:
            return str(e), 500

    return app
