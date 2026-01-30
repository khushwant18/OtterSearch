"""
Document extraction utilities for OtterSearch
"""
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
from .config import config


def extract_pdf(path: Path, max_pages: int = None) -> tuple[str, int]:
    """Extract text from PDF, return (text, page_count)"""
    max_pages = max_pages or config.pdf_max_pages
    
    try:
        with fitz.open(path) as doc:
            pages = min(len(doc), max_pages)
            text_parts = []
            
            for page_num in range(pages):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    text_parts.append(text)
            
            return "\n\n".join(text_parts), len(doc)
    except Exception as e:
        raise RuntimeError(f"Failed to open file '{path}'.") from e


def extract_image(path: Path) -> Path:
    """Load and return PIL Image"""
    try:
        return path
        # img = Image.open(path)
        # if img.mode == 'P':  # Palette mode with transparency
        #     img = img.convert('RGBA')
        # return img.convert("RGB")
    except Exception as e:
        raise RuntimeError(f"cannot identify image file '{path}'") from e
