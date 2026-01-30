"""
Configuration module for OtterSearch application
"""
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Config:
    data_dir: Path = field(default_factory=lambda: Path.home() / ".ottersearch")
    watch_dirs: list[Path] = field(default_factory=lambda: [Path.home() / "Documents"])
    
    # Models
    bge_model: str = "BAAI/bge-base-en-v1.5"
    text_vector_dim: int = 384  # MiniLM
    image_vector_dim: int = 512  # CLIP
    slm_model: str = "LiquidAI/LFM2-350M"
    
    # PDF extraction
    pdf_max_pages: int = 2
    chunk_size: int = 128
    chunk_overlap: int = 25
    
    # Indexing
    batch_size: int = 32
    
    # Vector store (HNSW)
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 100
    
    # Search
    top_k: int = 20
    num_query_variations: int = 4
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "models").mkdir(exist_ok=True)
        (self.data_dir / "cache").mkdir(exist_ok=True)


config = Config()
