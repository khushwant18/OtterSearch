"""
Unit tests for ottersearch.storage module
"""
import pytest
import numpy as np
from pathlib import Path
from datetime import datetime
from ottersearch.storage import VectorStore, MetadataStore
from ottersearch.models import Document


class TestVectorStore:
    """Test VectorStore class"""
    
    def test_vector_store_initialization(self, tmp_path):
        """Test initializing a new vector store"""
        index_path = tmp_path / "test_vectors.hnsw"
        store = VectorStore(index_path=index_path, store_type="test", dim=128)
        store.initialize()
        
        assert store.index is not None
        assert store.dim == 128
        assert store.store_type == "test"
    
    def test_add_vectors(self, tmp_path):
        """Test adding vectors to the store"""
        index_path = tmp_path / "test_vectors.hnsw"
        store = VectorStore(index_path=index_path, store_type="test", dim=128)
        store.initialize()
        
        doc_ids = ["doc1", "doc2", "doc3"]
        vectors = np.random.rand(3, 128).astype(np.float32)
        
        store.add_vectors(doc_ids, vectors)
        
        assert len(store.doc_ids) == 3
        assert store.doc_ids == doc_ids
    
    def test_search_vectors(self, tmp_path):
        """Test searching for similar vectors"""
        index_path = tmp_path / "test_vectors.hnsw"
        store = VectorStore(index_path=index_path, store_type="test", dim=128)
        store.initialize()
        
        # Add some test vectors
        doc_ids = ["doc1", "doc2", "doc3"]
        vectors = np.random.rand(3, 128).astype(np.float32)
        store.add_vectors(doc_ids, vectors)
        
        # Search with one of the vectors
        query = vectors[0].reshape(1, -1)
        results = store.search(query, k=2)
        
        assert len(results) > 0
        assert results[0][0] == "doc1"  # Should return itself as top result
        # Score should be close to 1, allow small floating point errors
        assert 0.99 <= results[0][1] <= 1.01
    
    def test_save_and_load(self, tmp_path):
        """Test saving and loading vector store"""
        index_path = tmp_path / "test_vectors.hnsw"
        
        # Create and save
        store1 = VectorStore(index_path=index_path, store_type="test", dim=128)
        store1.initialize()
        
        doc_ids = ["doc1", "doc2"]
        vectors = np.random.rand(2, 128).astype(np.float32)
        store1.add_vectors(doc_ids, vectors)
        store1.save()
        
        # Load in new instance
        store2 = VectorStore(index_path=index_path, store_type="test", dim=128)
        store2.initialize()
        
        assert len(store2.doc_ids) == 2
        assert store2.doc_ids == ["doc1", "doc2"]
    
    def test_search_empty_store(self, tmp_path):
        """Test searching in empty vector store"""
        index_path = tmp_path / "test_vectors.hnsw"
        store = VectorStore(index_path=index_path, store_type="test", dim=128)
        store.initialize()
        
        query = np.random.rand(1, 128).astype(np.float32)
        results = store.search(query, k=5)
        
        assert results == []


class TestMetadataStore:
    """Test MetadataStore class"""
    
    def test_metadata_store_creation(self, tmp_path):
        """Test creating a new metadata store"""
        db_path = tmp_path / "test.db"
        
        with MetadataStore(db_path=db_path) as store:
            assert store.conn is not None
            assert db_path.exists()
    
    def test_upsert_document(self, tmp_path):
        """Test inserting and updating documents"""
        db_path = tmp_path / "test.db"
        test_file = tmp_path / "test.pdf"
        test_file.write_text("content")
        
        doc = Document.from_path(test_file, "test content", "pdf")
        doc.page_count = 5
        
        with MetadataStore(db_path=db_path) as store:
            store.upsert_document(doc)
            
            # Retrieve and verify
            retrieved = store.get_document(doc.id)
            assert retrieved is not None
            assert retrieved.path == test_file
            assert retrieved.content == "test content"
            assert retrieved.page_count == 5
    
    def test_get_nonexistent_document(self, tmp_path):
        """Test retrieving document that doesn't exist"""
        db_path = tmp_path / "test.db"
        
        with MetadataStore(db_path=db_path) as store:
            result = store.get_document("nonexistent_id")
            assert result is None
    
    def test_document_exists(self, tmp_path):
        """Test checking if document exists"""
        db_path = tmp_path / "test.db"
        test_file = tmp_path / "test.pdf"
        test_file.write_text("content")
        
        doc = Document.from_path(test_file, "test content", "pdf")
        
        with MetadataStore(db_path=db_path) as store:
            # Should not exist initially
            assert not store.document_exists(test_file)
            
            # Insert document
            store.upsert_document(doc)
            
            # Should exist now
            assert store.document_exists(test_file)
    
    def test_get_stats(self, tmp_path):
        """Test getting statistics from metadata store"""
        db_path = tmp_path / "test.db"
        
        # Create test documents
        test_files = [
            (tmp_path / "test1.pdf", "pdf"),
            (tmp_path / "test2.pdf", "pdf"),
            (tmp_path / "test3.png", "image"),
        ]
        
        for file_path, doc_type in test_files:
            file_path.write_text("content")
        
        with MetadataStore(db_path=db_path) as store:
            for file_path, doc_type in test_files:
                doc = Document.from_path(file_path, "content", doc_type)
                store.upsert_document(doc)
            
            stats = store.get_stats()
            
            assert stats["total"] == 3
            assert stats["by_type"]["pdf"] == 2
            assert stats["by_type"]["image"] == 1
    
    def test_upsert_updates_existing(self, tmp_path):
        """Test that upsert replaces existing documents"""
        db_path = tmp_path / "test.db"
        test_file = tmp_path / "test.pdf"
        test_file.write_text("content")
        
        doc1 = Document.from_path(test_file, "original content", "pdf")
        doc1.page_count = 5
        
        with MetadataStore(db_path=db_path) as store:
            store.upsert_document(doc1)
            
            # Update with new content
            doc2 = Document.from_path(test_file, "updated content", "pdf")
            doc2.page_count = 10
            
            store.upsert_document(doc2)
            
            # Should have only one document
            stats = store.get_stats()
            assert stats["total"] == 1
            
            # Should have updated content
            retrieved = store.get_document(doc2.id)
            assert retrieved.content == "updated content"
            assert retrieved.page_count == 10
