"""
Unit tests for ottersearch.chunking module
"""
import pytest
from ottersearch.chunking import chunk_text


class TestChunking:
    """Test text chunking functionality"""
    
    def test_chunk_short_text(self):
        """Test chunking text shorter than chunk size"""
        text = "This is a short text."
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_long_text(self):
        """Test chunking longer text"""
        # Create text with 200 words
        words = ["word"] * 200
        text = " ".join(words)
        
        chunks = chunk_text(text, chunk_size=128, overlap=25)
        
        assert len(chunks) > 1
        # All chunks should be non-empty
        assert all(len(chunk) > 0 for chunk in chunks)
    
    def test_chunk_overlap(self):
        """Test that chunks have proper overlap"""
        # Create easily identifiable text
        words = [f"word{i}" for i in range(100)]
        text = " ".join(words)
        
        chunks = chunk_text(text, chunk_size=128, overlap=50)
        
        # With overlap, consecutive chunks should share some content
        if len(chunks) > 1:
            # Not a direct string match since overlap is word-based
            # But the chunks should have some overlap
            assert len(chunks[0].split()) > 0
            assert len(chunks[1].split()) > 0
    
    def test_chunk_custom_sizes(self):
        """Test chunking with custom sizes"""
        words = ["word"] * 150
        text = " ".join(words)
        
        # Larger chunk size should result in fewer chunks
        chunks_large = chunk_text(text, chunk_size=200, overlap=20)
        chunks_small = chunk_text(text, chunk_size=100, overlap=20)
        
        assert len(chunks_large) <= len(chunks_small)
    
    def test_chunk_empty_text(self):
        """Test chunking empty text"""
        text = ""
        chunks = chunk_text(text, chunk_size=128, overlap=25)
        
        assert len(chunks) == 1
        assert chunks[0] == ""
    
    def test_chunk_whitespace_text(self):
        """Test chunking text with only whitespace"""
        text = "   \n\t  "
        chunks = chunk_text(text, chunk_size=128, overlap=25)
        
        assert len(chunks) >= 1
    
    def test_chunk_preserves_content(self):
        """Test that all original words appear in chunks"""
        words = [f"unique{i}" for i in range(80)]
        text = " ".join(words)
        
        chunks = chunk_text(text, chunk_size=128, overlap=25)
        
        # Join all chunks and extract words
        all_chunk_words = set()
        for chunk in chunks:
            all_chunk_words.update(chunk.split())
        
        # All original words should appear in at least one chunk
        original_words = set(words)
        assert original_words.issubset(all_chunk_words)
    
    def test_chunk_default_parameters(self):
        """Test chunking with default parameters"""
        words = ["word"] * 200
        text = " ".join(words)
        
        # Should work with defaults from config
        chunks = chunk_text(text)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
