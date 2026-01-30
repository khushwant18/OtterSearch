#!/usr/bin/env bash
# OtterSearch - Quick Setup with uv (Lightning-fast)

set -e

echo "🦦 OtterSearch - Ultra-lightweight Setup"
echo "========================================"
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📥 Installing uv (Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "✓ Using uv for lightning-fast setup"
echo ""

# Create virtual environment
echo "🔧 Creating virtual environment..."
uv venv

# Activate venv
source .venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
uv pip install -r requirements.txt

echo ""
echo "✅ Setup complete! Ready to go 🚀"
echo ""
echo "Next: python __main__.py"
echo "Then open: http://localhost:8000"
echo ""
