"""
OtterSearch - Web UI launcher
"""
import os
from pathlib import Path
from ottersearch.api import create_app
from ottersearch.config import config

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

UI_HTML = (Path(__file__).parent / "ui.html").read_text()

app = create_app(UI_HTML)

if __name__ == "__main__":
    print(f"🦦 OtterSearch running at http://{config.host}:{config.port}")
    print("Open this URL in your browser to search")
    app.run(host=config.host, port=config.port, debug=False)
