"""
OtterSearch - Image & PDF semantic search with AI
"""

__version__ = "1.0.0"
__author__ = "OtterSearch Contributors"

from .config import Config, config
from .models import Document, SearchResult
from .indexer import Indexer
from .searcher import HybridSearcher

__all__ = [
    "Config",
    "config",
    "Document",
    "SearchResult",
    "Indexer",
    "HybridSearcher",
]
