"""
Web server launcher for OtterSearch
"""
from pathlib import Path
from .api import create_app
from .config import config


def run_server():
    """Launch web server - simple, lightweight"""
    ui_html_path = Path(__file__).parent.parent / "ui.html"
    if not ui_html_path.exists():
        print(f"Error: ui.html not found at {ui_html_path}")
        return
    
    UI_HTML = ui_html_path.read_text()
    app = create_app(UI_HTML)
    
    print(f"🦦 OtterSearch running at http://{config.host}:{config.port}")
    print("Open this URL in your browser to search images & PDFs")
    print("Press Ctrl+C to stop\n")
    app.run(host=config.host, port=config.port, debug=False)


if __name__ == "__main__":
    run_server()
