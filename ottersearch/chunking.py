"""
Text chunking utilities for OtterSearch
"""
from .config import config


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
    """Split text into overlapping chunks by words"""
    chunk_size = chunk_size or config.chunk_size
    overlap = overlap or config.chunk_overlap
    
    words = text.split()
    words_per_chunk = int(chunk_size * 0.75)
    overlap_words = int(overlap * 0.75)
    
    if len(words) <= words_per_chunk:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + words_per_chunk
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start = end - overlap_words
        
        if len(words) - start < overlap_words:
            break
    
    return chunks
