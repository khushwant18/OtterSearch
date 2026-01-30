# 🦦 OtterSearch

**AI-powered image & PDF search engine for your computer**

Help you easily find PDFs and images in your laptop using semantic understanding.

## ⚡ Quick Start (30 Seconds)

```bash
# Setup with uv (fastest)
bash setup_uv.sh

# Run
python __main__.py

# Open in browser
# http://localhost:8000
```

## 🎯 Features

- 🖼️ **Image Search** - Find images using text descriptions
- 📄 **PDF Search** - Search across PDF documents (indexes first 3 pages)
- 🤖 **AI-Powered** - Semantic understanding, not just keywords
- 🔍 **Query Expansion** - Uses SLM (Small Language Model) to enhance search queries
- ⚡ **Fast** - HNSW vector indexing
- 🏠 **Local** - Everything stays on your computer
- 💻 **Lightweight** - Minimal dependencies, runs anywhere

## 📦 Requirements

- Python 3.9+
- ~1GB disk space (for ML models)
- `uv` (or standard pip)

## 🚀 Setup Options

### Option 1: uv (Recommended - Fastest)
```bash
bash setup_uv.sh
python __main__.py
```

### Option 2: Standard pip
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python __main__.py
```

## 📖 Usage

1. Open **http://localhost:8000**
2. Click **Index** button
3. Select folders to search
4. Wait for indexing
5. Search using the text box

## 🛠️ Configuration

Edit `ottersearch/config.py` to customize:

```python
data_dir: Path = Path.home() / ".ottersearch"
port: int = 8000
batch_size: int = 32
```

## 📁 Architecture

```
ottersearch/
├── config.py       - Settings
├── models.py       - Data structures
├── extractors.py   - PDF/image extraction
├── storage.py      - Vector indexing
├── ml_models.py    - AI models (CLIP, transformers)
├── indexer.py      - Indexing pipeline
├── searcher.py     - Semantic search
└── api.py          - Web server
```

## 🔧 Troubleshooting

**Port 8000 in use?**
```python
# Edit ottersearch/config.py
port: int = 8001
```

**Out of memory?**
```python
# Edit ottersearch/config.py
batch_size: int = 16
```

**Models not downloading?**
- Check disk space (need ~1GB)
- Models go to: `~/.ottersearch/models/`

## 🧪 Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=ottersearch --cov-report=html

# Run specific test file
pytest tests/test_models.py
```

## 🤝 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 📊 Performance

- Indexing: ~30 docs/second
- Search: <2 sec for 11000+ documents
- Memory: 2-4GB with batch processing

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

## 📄 License

MIT License - See [LICENSE](LICENSE)

## 🙋 Support

- Check [README.md](README.md) for details
- See [DEVELOPMENT.md](DEVELOPMENT.md) for extending
- [STRUCTURE.md](STRUCTURE.md) explains file organization

---

**Find anything in your images and PDFs!** 🦦

Built with CLIP, transformers, and HNSW vector search.
