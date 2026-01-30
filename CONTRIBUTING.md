# Contributing to OtterSearch

Thank you for your interest in contributing to OtterSearch! 🦦

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment (OS, Python version)

### Suggesting Features

Feature requests are welcome! Please:
- Check existing issues first
- Describe the use case
- Explain why this would be useful

### Code Contributions

1. **Fork the repository**
2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the existing code style
   - Add docstrings to new functions
   - Update tests if needed
   - Keep commits focused and atomic

4. **Test your changes**
   ```bash
   pytest tests/
   ```

5. **Submit a Pull Request**
   - Reference any related issues
   - Describe what changed and why
   - Ensure all tests pass

## Code Style Guidelines

- Use Python 3.9+ features
- Follow PEP 8 conventions
- Add type hints to function signatures
- Write descriptive docstrings
- Keep functions focused and concise

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/ottersearch.git
cd ottersearch

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8

# Run tests
pytest tests/
```

## Project Structure

```
ottersearch/
├── config.py       - Configuration settings
├── models.py       - Data structures
├── extractors.py   - PDF/image extraction
├── storage.py      - Vector indexing and metadata
├── ml_models.py    - AI model management
├── indexer.py      - Document indexing pipeline
├── searcher.py     - Semantic search
└── api.py          - Flask web server
```

## Questions?

Feel free to open an issue for any questions about contributing!

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
