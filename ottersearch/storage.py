"""
Storage backends for OtterSearch (Vector Store and Metadata Store)
"""
import sqlite3
import json
from pathlib import Path
from datetime import datetime
import hnswlib
import numpy as np
from .config import config
from .models import Document


class VectorStore:
    def __init__(self, index_path: Path = None, store_type: str = "combined", dim: int = None):
        self.store_type = store_type
        self.dim = dim or config.text_vector_dim 
        if index_path is None:
            index_path = config.data_dir / f"vectors_{store_type}.hnsw"
        self.index_path = index_path
        self.index: hnswlib.Index | None = None
        self.doc_ids: list[str] = []
        
    def save(self):
        if self.index is not None:
            self.index.save_index(str(self.index_path))
            # Save with store_type in filename
            doc_ids_file = self.index_path.parent / f"doc_ids_{self.store_type}.json"
            doc_ids_file.write_text(json.dumps(self.doc_ids))

    def initialize(self, dimension: int = None):
        dimension = dimension or self.dim
      
        self.index = hnswlib.Index(space="cosine", dim=dimension)
        
        if self.index_path.exists():
            self.index.load_index(str(self.index_path))
            self.index.set_ef(config.hnsw_ef_search)
            
            # Load with store_type in filename
            doc_ids_path = self.index_path.parent / f"doc_ids_{self.store_type}.json"
            if doc_ids_path.exists():
                self.doc_ids = json.loads(doc_ids_path.read_text())
        else:
            self.index.init_index(
                max_elements=1_000_000,
                ef_construction=config.hnsw_ef_construction,
                M=config.hnsw_m,
            )
    
    def add_vectors(self, doc_ids: list[str], vectors: np.ndarray):
        if self.index is None:
            raise RuntimeError("Index not initialized")
        
        current_count = self.index.get_current_count()
        indices = np.arange(current_count, current_count + len(doc_ids))
        
        self.index.add_items(vectors, indices)
        self.doc_ids.extend(doc_ids)
        
    def search(self, query_vector: np.ndarray, k: int = None) -> list[tuple[str, float]]:
        k = k or config.top_k
        
        if self.index is None or len(self.doc_ids) == 0:
            return []
        
        labels, distances = self.index.knn_query(query_vector, k=min(k, len(self.doc_ids)))
        
        results = []
        for label, dist in zip(labels[0], distances[0]):
            if label < len(self.doc_ids):
                results.append((self.doc_ids[label], float(1 - dist)))
        
        return results
        


class MetadataStore:
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or config.data_dir / "index.db"
        self.conn: sqlite3.Connection | None = None
        
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, *args):
        self.close()
        
    def connect(self):
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
    
    def close(self):
        if self.conn:
            self.conn.close()
    
    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                hash TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                modified_at TEXT NOT NULL,
                page_count INTEGER,
                chunk_index INTEGER DEFAULT 0,
                total_chunks INTEGER DEFAULT 1,
                content TEXT
            );
            
            CREATE INDEX IF NOT EXISTS idx_hash ON documents(hash);
            CREATE INDEX IF NOT EXISTS idx_path ON documents(path);
        """)
        self.conn.commit()
    
    def upsert_document(self, doc: Document):
        content_str = str(doc.content) if isinstance(doc.content, str) else ""
        
        self.conn.execute("""
            INSERT OR REPLACE INTO documents 
            (id, path, hash, doc_type, size_bytes, modified_at, 
            page_count, chunk_index, total_chunks, content)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            doc.id, str(doc.path), doc.hash, doc.doc_type,
            doc.size_bytes, doc.modified_at.isoformat(),
            doc.page_count, doc.chunk_index, doc.total_chunks, content_str
        ))
        
    
    def get_document(self, doc_id: str) -> Document | None:
        cursor = self.conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        row = cursor.fetchone()
        if not row:
            return None
        
        return Document(
            path=Path(row["path"]),
            content=row["content"],
            doc_type=row["doc_type"],
            size_bytes=row["size_bytes"],
            modified_at=datetime.fromisoformat(row["modified_at"]),
            page_count=row["page_count"],
            chunk_index=row["chunk_index"],
            total_chunks=row["total_chunks"],
        )
    
    def document_exists(self, path: Path) -> bool:
        cursor = self.conn.execute("SELECT 1 FROM documents WHERE path = ?", (str(path),))
        return cursor.fetchone() is not None
    
    def get_stats(self) -> dict:
        cursor = self.conn.execute("SELECT COUNT(*) as count FROM documents")
        total = cursor.fetchone()["count"]
        
        cursor = self.conn.execute(
            "SELECT doc_type, COUNT(*) as count FROM documents GROUP BY doc_type"
        )
        by_type = {row["doc_type"]: row["count"] for row in cursor.fetchall()}
        
        return {"total": total, "by_type": by_type}
