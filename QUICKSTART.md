# 🦦 OtterSearch - Quick Start (30 Seconds)

## ⚡ Setup with uv (Recommended - Fastest)

`uv` is blazing-fast Python installer. One binary, no venv management headaches!

```bash
# 1. Install uv (one-time only)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Setup OtterSearch
uv venv
source .venv/bin/activate    # macOS/Linux
# or: .venv\Scripts\activate  # Windows

# 3. Install dependencies
uv pip install -r requirements.txt
```

## 🚀 Run

```bash
python __main__.py
```

Open browser: **http://localhost:8000**

Done! 🦦

---

## 🎨 Using OtterSearch

1. **Open** http://localhost:8000
2. **Click Index** button
3. **Select folders** to search (Documents, Desktop, etc.)
4. **Wait** for indexing (first time: 5-10 min)
5. **Search** using the search bar at top
6. **Click results** to open files or preview

---

## 🔧 Quick Config Changes

**Port already in use?**
```python
# Edit: ottersearch/config.py
port: int = 8001
```

**Out of memory?**
```python
# Edit: ottersearch/config.py
batch_size: int = 16
```

---

## 📚 More Help

- **README.md** - Full documentation
- **DEVELOPMENT.md** - For developers
- **INDEX.txt** - File structure overview

---

That's it! Enjoy searching! 🦦
