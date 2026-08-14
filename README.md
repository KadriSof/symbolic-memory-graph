# MemoryGraph

## Requirements

- CMake 3.15+
- Python 3.12+
- uv (Python package manager)
- C++17 compiler
- Git (for submodules)

## Quick Start

### Setup

```
# Full setup with tests
./setup.sh --ctest --pytest

# Clean rebuild with verbose output
./setup.sh --clean --verbose --ctest --pytest

# Run only C++ tests with filter
./setup.sh --ctest --test-filter MemoryGraphTest.*

# Run only Python tests
./setup.sh --pytest

# Fast rebuild (skip venv, run tests)
./setup.sh --skip-venv --ctest --pytest

# Just build (skip tests)
./setup.sh --skip-tests
```

### One-Command Setup

```bash
git clone --recursive https://github.com/your-repo/memory_graph.git
cd memory_graph
./setup.sh
```
