# Changelog

All notable changes to OtterSearch will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-28

### Added
- Initial release of OtterSearch
- Semantic search for images using CLIP embeddings
- PDF text search with BGE text embeddings
- Web-based UI for searching and previewing documents
- HNSW vector indexing for fast similarity search
- Batch indexing with progress tracking
- Support for PDF chunking with configurable sizes
- Query variation generation using LLM
- File preview and reveal in finder functionality
- Incremental indexing (update mode) for new files only

### Features
- Multi-format support (PDF, PNG, JPG, JPEG)
- Local-first architecture (no data leaves your computer)
- Lightweight dependencies
- Fast indexing with parallel processing
- Hybrid search across images and PDFs
- Configurable indexing parameters
- SQLite-based metadata storage
- Thread-safe indexing operations

### Technical
- Python 3.9+ support
- CLIP for image embeddings
- Sentence Transformers for text embeddings
- HNSW for vector similarity search
- Flask web server
- PyMuPDF for PDF extraction
