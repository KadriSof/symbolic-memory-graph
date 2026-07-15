#!/bin/bash
# setup.sh - Complete setup from scratch

set -e # Exit on error

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

echo "MemoryGraph Setup"
echo "═══════════════════════════════════════════════════════════"

# 1. Check prerequisites
echo "Checking prerequisites..."
command -v cmake >/dev/null 2>&1 || {
  echo "[ERROR] cmake not found"
  exit 1
}
command -v uv >/dev/null 2>&1 || {
  echo "[ERROR] uv not found"
  exit 1
}
command -v git >/dev/null 2>&1 || {
  echo "[ERROR] git not found"
  exit 1
}

# 2. Initialize submodules
echo "Initializing submodules..."
if [ -f .gitmodules ]; then
  git submodule update --init --recursive
else
  echo "[ERROR] No .gitmodules found, skipping..."
fi

# 3. Set up Python virtual environment
echo "Setting up Python environment..."
cd "$PROJECT_ROOT/python"
uv venv --python 3.12 --clear
source .venv/bin/activate
uv pip install -e .

# 4. Build C++ library
echo "Building C++ library..."
cd "$PROJECT_ROOT"
mkdir -p build
cd build

# Get Python executable from venv
PYTHON_EXEC="$PROJECT_ROOT/python/.venv/bin/python"

cmake -DBUILD_PYTHON_BINDINGS=ON \
  -DPython_EXECUTABLE="$PYTHON_EXEC" \
  ..
cmake --build . --target memory_graph_core

# 5. Verify installation
echo "Verifying installation..."
cd "$PROJECT_ROOT/python"
source .venv/bin/activate
python -c "from memory_graph import MemoryGraph, Node, Edge; print('Import successful!')"

cd "$PROJECT_ROOT"

echo "═══════════════════════════════════════════════════════════"
echo "Setup complete!"
echo ""
echo "To activate the virtual environment:"
echo "  source python/.venv/bin/activate"
echo ""
echo "To run Python with the environment:"
echo "  cd python && source .venv/bin/activate && python"
echo ""
echo "To rebuild the C++ bindings:"
echo "  cd build && cmake --build . --target memory_graph_core"
