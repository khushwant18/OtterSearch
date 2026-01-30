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
    """HNSW-based vector store for semantic search.
    
    Manages vector embeddings and provides fast approximate nearest neighbor search
    using HNSW (Hierarchical Navigable Small World) algorithm.
    
    Attributes:
        store_type: Type identifier (e.g., 'pdf', 'image', 'combined')
        dim: Embedding dimension
        index_path: Path to HNSW index file on disk
        index: HNSW index instance (None until initialized)
        doc_ids: List mapping HNSW internal indices to document IDs
    """
    def __init__(self, index_path: Path = None, store_type: str = "combined", dim: int = None):
        self.store_type = store_type
        self.dim = dim or config.text_vector_dim 
        if index_path is None:
            index_path = config.data_dir / f"vectors_{store_type}.hnsw"
        self.index_path = index_path
        self.index: hnswlib.Index | None = None
        self.doc_ids: list[str] = []
        
    def save(self):
        """Save HNSW index and document ID mappings to disk.
        
        Persists both the vector index and the doc_ids list to separate files.
        Index saved to {store_type}.hnsw, doc IDs to doc_ids_{store_type}.json.
        
        Error Handling:
            - Silently skips if index is not initialized (None)
            - File write errors propagate to caller
            - Overwrites existing files without backup
        """
        if self.index is not None:
            self.index.save_index(str(self.index_path))
            doc_ids_file = self.index_path.parent / f"doc_ids_{self.store_type}.json"
            doc_ids_file.write_text(json.dumps(self.doc_ids))

    def initialize(self, dimension: int = None):
        """Initialize or load HNSW index from disk.
        
        Creates new index if file doesn't exist, loads existing index otherwise.
        
        Args:
            dimension: Embedding dimension (uses self.dim if not provided)
            
        Error Handling:
            - Creates new empty index if file doesn't exist
            - Loads existing index and doc_ids if files exist
            - Missing doc_ids file results in empty list
            - Index loading errors propagate to caller
            - Dimension mismatch with existing index causes errors
        """
        dimension = dimension or self.dim
      
        self.index = hnswlib.Index(space="cosine", dim=dimension)
        
        if self.index_path.exists():
            self.index.load_index(str(self.index_path))
            self.index.set_ef(config.hnsw_ef_search)
            
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
        """Add new vectors to the HNSW index.
        
        Args:
            doc_ids: List of document IDs corresponding to vectors
            vectors: numpy array of shape (N, dim) containing embeddings
            
        Raises:
            RuntimeError: If index is not initialized
            
        Error Handling:
            - Raises RuntimeError if called before initialize()
            - Vectors automatically assigned sequential internal indices
            - HNSW capacity errors propagate to caller
            - Vector dimension mismatches cause errors
        """
        if self.index is None:
            raise RuntimeError("Index not initialized")
        
        current_count = self.index.get_current_count()
        indices = np.arange(current_count, current_count + len(doc_ids))
        
        self.index.add_items(vectors, indices)
        self.doc_ids.extend(doc_ids)
        
    def search(self, query_vector: np.ndarray, k: int = None) -> list[tuple[str, float]]:
        """Search for nearest neighbors to query vector.
        
        Args:
            query_vector: numpy array of shape (1, dim) containing query embedding
            k: Number of results to return (uses config.top_k if not provided)
            
        Returns:
            List of (doc_id, similarity_score) tuples, sorted by similarity descending.
            Similarity scores range from 0 (dissimilar) to 1 (identical).
            
        Error Handling:
            - Returns empty list if index is None or doc_ids is empty
            - Automatically limits k to available document count
            - Invalid vector dimensions cause errors
            - Filters out invalid label indices (safety check)
        """
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
    """SQLite-based metadata storage for document information.
    
    Stores document metadata including paths, hashes, types, sizes, and content.
    Implements context manager protocol for automatic connection management.
    
    Attributes:
        db_path: Path to SQLite database file
        conn: SQLite connection instance (None until connected)
    
    Usage:
        with MetadataStore() as store:
            store.upsert_document(doc)
    """
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or config.data_dir / "index.db"
        self.conn: sqlite3.Connection | None = None
        
    def __enter__(self):
        """Enter context manager - opens database connection."""
        self.connect()
        return self
    
    def __exit__(self, *args):
        """Exit context manager - closes database connection."""
        self.close()
        
    def connect(self):
        """Open SQLite connection and initialize schema.
        
        Error Handling:
            - Creates database file if it doesn't exist
            - Schema initialization is idempotent (IF NOT EXISTS)
            - Connection errors propagate to caller
        """
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
    
    def close(self):
        """Close SQLite connection if open.
        
        Safe to call multiple times.
        """
        if self.conn:
            self.conn.close()
    
    def _init_schema(self):
        """Create database tables and indices if they don't exist.
        
        Creates:
            - documents table with all metadata fields
            - Index on hash for duplicate detection
            - Index on path for existence checks
            
        Idempotent - safe to call multiple times.
        """
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
        """Insert or update document metadata in database.
        
        Uses INSERT OR REPLACE to handle both new documents and updates.
        
        Args:
            doc: Document instance to store
            
        Error Handling:
            - Replaces existing document with same ID
            - Path content converted to string if not already
            - Database constraint violations propagate to caller
            - No automatic commit - caller must commit transaction
        """
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
        """Retrieve document metadata by ID.
        
        Args:
            doc_id: Document ID to look up
            
        Returns:
            Document instance if found, None otherwise
            
        Error Handling:
            - Returns None for non-existent doc_id
            - Database errors propagate to caller
            - Assumes all fields are present in row
        """
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
        """Check if document with given path exists in database.
        
        Args:
            path: Path to check
            
        Returns:
            True if document exists, False otherwise
            
        Error Handling:
            - Database errors propagate to caller
            - Uses path string comparison (exact match required)
        """
        cursor = self.conn.execute("SELECT 1 FROM documents WHERE path = ?", (str(path),))
        return cursor.fetchone() is not None
    
    def get_stats(self) -> dict:
        """Get index statistics including total count and breakdown by type.
        
        Returns:
            Dictionary with:
                - 'total': Total number of documents
                - 'by_type': Dict mapping doc_type to count
                
        Error Handling:
            - Database errors propagate to caller
            - Returns 0 counts if database is empty
        """
        cursor = self.conn.execute("SELECT COUNT(*) as count FROM documents")
        total = cursor.fetchone()["count"]
        
        cursor = self.conn.execute(
            "SELECT doc_type, COUNT(*) as count FROM documents GROUP BY doc_type"
        )
        by_type = {row["doc_type"]: row["count"] for row in cursor.fetchall()}
        
        return {"total": total, "by_type": by_type}
