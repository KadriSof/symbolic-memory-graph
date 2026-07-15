# justfile - Dev commands
# install: cargo install just

# Show all commands
default:
  @just --list

# Full setup from scratch (clone -> build -> test)
setup:
  @echo "Setting up MemoryGrap.."
  @echo init-submodules
  @just install
  @just build
  @echo "Setup complete! Run 'just test' to verify."

# Initialize git submodules
init-submodules:
  @echo "initializing submodules..."
  git submodules update --init --recursive

# Install Python package in editable mode
install:
  @echo "Installing Python package..."
  cd python && uv venv --python 3.12 --clear
  cd python && source .venv/bin/activate && uv pip install -e .

# Build C++ bindings
build:
  @echo "Building C++ bindings..."
  mkdir -p build
  cd build && cmake -DBUILD_PYTHON_BINDINGS=ON -DPython_EXECUTABLE=$(realpath python/.venv/bin/python) ..
  cd build && cmake --build . --target memory_graph_core

# Rebuild only the bindings (fast)
rebuild:
  @echo "Rebuilding bindings..."
  cd build && cmake --build . --target memory_graph_core

# Test Python import
test:
  @echo "Testing import..."
  cd python && source .venv/bin/activate && python -c "from memory_graph import MemoryGrap, Node, Edge; print('Import works!')"

# Run a Python script
run:
  @echo "Running Python..."
  cd python && source .venv/bin/activate && python

# Clean build artifacts
clean:
  @echo "Cleaning..."
  rm -rf build
  rm -rf python/.venv
  rm -rf python/src/memory_graph.egg-info
  rm -f python/src/memory_graph/memory_graph_core.so

# Format C++ code
format:
    @echo "Formatting C++..."
    clang-format -i include/**/*.hpp src/**/*.cpp bindings/**/*.cpp bindings/**/*.hpp

# Lint Python code
lint:
    @echo "Linting Python..."
    cd python && source .venv/bin/activate && black src/

# Run tests (C++)
test-cpp:
    @echo "Running C++ tests..."
    cd build && ctest --output-on-failure

# Run tests (Python)
test-python:
    @echo "Running Python tests..."
    cd python && source .venv/bin/activate && pytest

# Full test suite
test-all: test-cpp test-python

# Build documentation
docs:
    @echo "Building documentation..."
    # Placeholder for Doxygen/Sphinx

# Development watch mode (auto-rebuild on changes)
watch:
    @echo "Watching for changes..."
    @just rebuild
    @echo " Done"

