"""
Unit tests for ottersearch.config module
"""
import pytest
from pathlib import Path
from ottersearch.config import Config


class TestConfig:
    """Test configuration management"""
    
    def test_config_defaults(self):
        """Test default configuration values"""
        config = Config()
        
        # Check default paths
        assert config.data_dir == Path.home() / ".ottersearch"
        assert config.watch_dirs == [Path.home() / "Documents"]
        
        # Check model settings
        assert config.bge_model == "BAAI/bge-base-en-v1.5"
        assert config.text_vector_dim == 384
        assert config.image_vector_dim == 512
        
        # Check chunking settings
        assert config.pdf_max_pages == 2
        assert config.chunk_size == 128
        assert config.chunk_overlap == 25
        
        # Check indexing settings
        assert config.batch_size == 32
        
        # Check HNSW settings
        assert config.hnsw_m == 16
        assert config.hnsw_ef_construction == 200
        assert config.hnsw_ef_search == 100
        
        # Check search settings
        assert config.top_k == 20
        assert config.num_query_variations == 4
        
        # Check server settings
        assert config.host == "0.0.0.0"
        assert config.port == 8000
    
    def test_config_post_init_creates_directories(self, tmp_path):
        """Test that __post_init__ creates required directories"""
        test_dir = tmp_path / "test_ottersearch"
        
        config = Config()
        config.data_dir = test_dir
        config.__post_init__()
        
        assert test_dir.exists()
        assert (test_dir / "models").exists()
        assert (test_dir / "cache").exists()
    
    def test_config_custom_values(self):
        """Test creating config with custom values"""
        config = Config(
            batch_size=64,
            port=9000,
            top_k=30
        )
        
        assert config.batch_size == 64
        assert config.port == 9000
        assert config.top_k == 30
    
    def test_config_paths_are_pathlib(self):
        """Test that directory paths are Path objects"""
        config = Config()
        
        assert isinstance(config.data_dir, Path)
        assert all(isinstance(p, Path) for p in config.watch_dirs)
    
    def test_config_watch_dirs_mutable(self):
        """Test that watch_dirs can be modified"""
        config = Config()
        
        new_dir = Path.home() / "Downloads"
        config.watch_dirs.append(new_dir)
        
        assert new_dir in config.watch_dirs
        assert len(config.watch_dirs) >= 2
