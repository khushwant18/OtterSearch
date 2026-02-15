# 🦦 OtterSearch

**AI-powered multimodal search engine for your computer**

Find PDFs and images in your laptop using semantic understanding with image embeddings and AI-powered multimodal search.

## 📥 Download (macOS)

**Ready-to-use desktop app - no installation required!**

- [OtterSearch-1.0.0-arm64.dmg](https://github.com/khushwant18/OtterSearch/releases/download/v1.0.0/OtterSearch-1.0.0-arm64.dmg) - For Apple Silicon (M1/M2/M3)
- [OtterSearch-1.0.0.dmg](https://github.com/khushwant18/OtterSearch/releases/download/v1.0.0/OtterSearch-1.0.0.dmg) - For Intel Macs

Just download the right version for your Mac, open the DMG, and drag to Applications!

**Windows:** Coming soon! For now, use Python setup below.

### ✨ What's New in v1.0.0

- 📑 **Scanned PDF Detection** - Automatically detects pages with minimal text and indexes them as image embeddings using CLIP
- ⏸️ **Pause/Resume/Stop** - Full control over indexing with progress persistence across app restarts
- 💾 **Data Durability** - SQLite WAL mode with automatic checkpoints ensures no data loss
- 🔄 **Smart Updates** - Only indexes new/modified files, dramatically faster reindexing

## 🎯 Features

- 🖼️ **Image Search** - Find images using text descriptions with CLIP embeddings
- 📄 **PDF Search** - Search across PDF documents (indexes first 2 pages) with MiniLM embeddings
- 📑 **Scanned PDF Support** - Automatically detects and indexes scanned pages as images using CLIP
- ⏸️ **Pause/Resume Indexing** - Control indexing progress, pause and resume anytime
- 🛑 **Smart Updates** - Only indexes new/modified files, skips already-indexed ones
- 🤖 **AI-Powered** - Multimodal semantic understanding using image embeddings
- 🔍 **Query Expansion** - Uses LFM2-350M (Small Language Model) to enhance search queries
- ⚡ **Fast** - HNSW vector indexing with WAL-mode SQLite for durability
- 🏠 **Local** - Everything stays on your computer
- 💻 **Lightweight** - Minimal dependencies, runs anywhere

## 📖 Usage

1. Launch OtterSearch from Applications 
2. Click **Index Settings** button
3. Index folders (quick: Documents/Desktop/Downloads, or custom path)
4. Use **Pause/Resume/Stop** buttons to control indexing progress
5. Progress is saved - you can close the app and resume later
6. Search anything using the search box

**Note:** Scanned PDFs are automatically detected and indexed as images for better search accuracy!

---

## 🐍 Python Setup (Alternative)

If you prefer to run from source or don't want the DMG:

### ⚡ Quick Start (30 Seconds)

```bash
# Setup with uv (fastest)
bash setup_uv.sh

# Run
python __main__.py

# Open in browser
# http://localhost:8000
```

### 📦 Requirements

- Python 3.9+
- ~1GB disk space (for ML models)
- `uv` (or standard pip)

### 🚀 Setup Options

**Option 1: uv (Recommended - Fastest)**
```bash
bash setup_uv.sh
python __main__.py
```

**Option 2: Standard pip**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python __main__.py
```

---

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
- Search: <2 sec for 30000+ documents
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

Built with CLIP embeddings, MiniLM embeddings, LFM2-350M query expansion, and HNSW vector search.
