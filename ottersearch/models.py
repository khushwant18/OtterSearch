"""
Data models for OtterSearch application
"""
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Literal


@dataclass
class Document:
    path: Path
    content: str | Path
    doc_type: Literal["pdf", "image"]
    modified_at: datetime
    size_bytes: int
    page_count: int | None = None
    chunk_index: int = 0
    total_chunks: int = 1
    
    @property
    def hash(self) -> str:
        content = f"{self.path}:{self.size_bytes}:{self.modified_at.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    @property
    def id(self) -> str:
        base_id = self.hash[:16]
        if self.total_chunks > 1:
            return f"{base_id}_c{self.chunk_index}"
        return base_id
    
    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "doc_type": self.doc_type,
            "size_bytes": self.size_bytes,
            "modified_at": self.modified_at.isoformat(),
            "page_count": self.page_count,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "content": str(self.content) if isinstance(self.content, str) else None
        }
    
    @classmethod
    def from_path(cls, path: Path, content: str | object, doc_type: Literal["pdf", "image"]) -> "Document":
        stat = path.stat()
        return cls(
            path=path,
            content=content,
            doc_type=doc_type,
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            size_bytes=stat.st_size,
        )


@dataclass
class SearchResult:
    document: Document
    score: float
    source: str
    snippet: str | None = None
    
    def to_dict(self) -> dict:
        return {
            "document": self.document.to_dict(),
            "score": self.score,
            "source": self.source,
            "snippet": self.snippet
        }
