"""
Unit tests for ottersearch.models module
"""
import pytest
from pathlib import Path
from datetime import datetime
from ottersearch.models import Document, SearchResult


class TestDocument:
    """Test Document dataclass"""
    
    def test_document_creation(self, tmp_path):
        """Test creating a Document instance"""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("content")
        
        doc = Document.from_path(test_file, "test content", "pdf")
        
        assert doc.path == test_file
        assert doc.content == "test content"
        assert doc.doc_type == "pdf"
        assert isinstance(doc.modified_at, datetime)
        assert doc.size_bytes > 0
    
    def test_document_hash(self, tmp_path):
        """Test document hash generation"""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("content")
        
        doc = Document.from_path(test_file, "test content", "pdf")
        hash1 = doc.hash
        
        # Hash should be consistent
        assert hash1 == doc.hash
        assert len(hash1) == 64  # SHA256 produces 64 hex chars
    
    def test_document_id_single_chunk(self, tmp_path):
        """Test document ID for single chunk"""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("content")
        
        doc = Document.from_path(test_file, "test content", "pdf")
        doc_id = doc.id
        
        assert len(doc_id) == 16  # First 16 chars of hash
        assert "_c" not in doc_id
    
    def test_document_id_multiple_chunks(self, tmp_path):
        """Test document ID for chunked documents"""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("content")
        
        doc = Document.from_path(test_file, "chunk 0", "pdf")
        doc.chunk_index = 0
        doc.total_chunks = 3
        
        doc_id = doc.id
        assert "_c0" in doc_id
        
        doc.chunk_index = 2
        doc_id2 = doc.id
        assert "_c2" in doc_id2
    
    def test_document_to_dict(self, tmp_path):
        """Test converting document to dictionary"""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("content")
        
        doc = Document.from_path(test_file, "test content", "pdf")
        doc.page_count = 5
        
        doc_dict = doc.to_dict()
        
        assert doc_dict["path"] == str(test_file)
        assert doc_dict["doc_type"] == "pdf"
        assert doc_dict["page_count"] == 5
        assert doc_dict["size_bytes"] > 0
        assert "modified_at" in doc_dict
        assert doc_dict["content"] == "test content"
    
    def test_document_image_type(self, tmp_path):
        """Test creating image document"""
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"fake image data")
        
        doc = Document.from_path(test_file, test_file, "image")
        
        assert doc.doc_type == "image"
        assert doc.content == test_file


class TestSearchResult:
    """Test SearchResult dataclass"""
    
    def test_search_result_creation(self, tmp_path):
        """Test creating SearchResult instance"""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("content")
        
        doc = Document.from_path(test_file, "test content", "pdf")
        result = SearchResult(
            document=doc,
            score=0.95,
            source="hybrid",
            snippet="test snippet"
        )
        
        assert result.document == doc
        assert result.score == 0.95
        assert result.source == "hybrid"
        assert result.snippet == "test snippet"
    
    def test_search_result_to_dict(self, tmp_path):
        """Test converting search result to dictionary"""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("content")
        
        doc = Document.from_path(test_file, "test content", "pdf")
        result = SearchResult(
            document=doc,
            score=0.95,
            source="hybrid",
            snippet="test snippet"
        )
        
        result_dict = result.to_dict()
        
        assert "document" in result_dict
        assert result_dict["score"] == 0.95
        assert result_dict["source"] == "hybrid"
        assert result_dict["snippet"] == "test snippet"
        assert result_dict["document"]["path"] == str(test_file)
    
    def test_search_result_no_snippet(self, tmp_path):
        """Test search result without snippet"""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("content")
        
        doc = Document.from_path(test_file, "test content", "pdf")
        result = SearchResult(
            document=doc,
            score=0.80,
            source="vector"
        )
        
        assert result.snippet is None
        result_dict = result.to_dict()
        assert result_dict["snippet"] is None
