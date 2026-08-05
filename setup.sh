#!/bin/bash
# setup.sh - Complete setup, build, and test script

set -e # Exit on error

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
  echo ""
  echo "═══════════════════════════════════════════════════════════"
  echo -e "${BLUE}$1${NC}"
  echo "═══════════════════════════════════════════════════════════"
}

print_success() {
  echo -e "${GREEN}[SUCCESS] $1${NC}"
}

print_error() {
  echo -e "${RED}[ERROR] $1${NC}"
}

print_warning() {
  echo -e "${YELLOW}[WARNING]  $1${NC}"
}

print_info() {
  echo -e "${BLUE}[INFO]  $1${NC}"
}

# Parse command line arguments
SKIP_SUBMODULES=false
SKIP_VENV=false
SKIP_BUILD=false
SKIP_TESTS=false
CLEAN_BUILD=false
RUN_CTESTS=false
RUN_PYTESTS=false
VERBOSE=false
TEST_FILTER=""

while [[ $# -gt 0 ]]; do
  case $1 in
  --skip-submodules) SKIP_SUBMODULES=true ;;
  --skip-venv) SKIP_VENV=true ;;
  --skip-build) SKIP_BUILD=true ;;
  --skip-tests) SKIP_TESTS=true ;;
  --clean) CLEAN_BUILD=true ;;
  --ctest) RUN_CTESTS=true ;;
  --pytest) RUN_PYTESTS=true ;;
  --test-filter)
    TEST_FILTER="$2"
    shift
    ;;
  --verbose) VERBOSE=true ;;
  --help)
    echo "Usage: ./setup.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --skip-submodules   Skip git submodule initialization"
    echo "  --skip-venv         Skip Python virtual environment setup"
    echo "  --skip-build        Skip building C++ library"
    echo "  --skip-tests        Skip running tests"
    echo "  --clean             Clean build directory before building"
    echo "  --ctest             Run C++ tests"
    echo "  --pytest            Run Python tests"
    echo "  --test-filter       Filter for tests (e.g., 'MemoryGraphTest.*')"
    echo "  --verbose           Verbose output"
    echo "  --help              Show this help message"
    exit 0
    ;;
  *)
    print_error "Unknown option: $1"
    exit 1
    ;;
  esac
  shift
done

# 1. Check prerequisites
print_header "Checking prerequisites"

command -v cmake >/dev/null 2>&1 || {
  print_error "cmake not found"
  exit 1
}
command -v uv >/dev/null 2>&1 || {
  print_error "uv not found"
  exit 1
}
command -v git >/dev/null 2>&1 || {
  print_error "git not found"
  exit 1
}
command -v g++ >/dev/null 2>&1 || {
  print_warning "g++ not found, trying clang++"
  command -v clang++ >/dev/null 2>&1 || {
    print_error "No C++ compiler found"
    exit 1
  }
}

print_success "All prerequisites found"

# 2. Initialize submodules
if [ "$SKIP_SUBMODULES" = false ]; then
  print_header "Initializing submodules"
  if [ -f .gitmodules ]; then
    git submodule update --init --recursive
    print_success "Submodules initialized"
  else
    print_warning "No .gitmodules found, skipping..."
  fi
else
  print_info "Skipping submodules (--skip-submodules)"
fi

# 3. Set up Python virtual environment
if [ "$SKIP_VENV" = false ]; then
  print_header "Setting up Python environment"
  cd "$PROJECT_ROOT/python"

  # Remove existing venv if it exists
  if [ -d ".venv" ]; then
    print_info "Removing existing virtual environment..."
    rm -rf .venv
  fi

  uv venv --python 3.12 --clear
  source .venv/bin/activate
  uv pip install -e .

  print_success "Python environment setup complete"
else
  print_info "Skipping Python venv (--skip-venv)"
fi

# 4. Build C++ library
if [ "$SKIP_BUILD" = false ]; then
  print_header "Building C++ library"

  # Clean build if requested
  if [ "$CLEAN_BUILD" = true ] && [ -d "$PROJECT_ROOT/build" ]; then
    print_info "Cleaning build directory..."
    rm -rf "$PROJECT_ROOT/build"
  fi

  cd "$PROJECT_ROOT"
  mkdir -p build
  cd build

  # Get Python executable from venv
  PYTHON_EXEC="$PROJECT_ROOT/python/.venv/bin/python"

  if [ "$VERBOSE" = true ]; then
    print_info "Running cmake with verbose output..."
  fi

  cmake -DBUILD_PYTHON_BINDINGS=ON \
    -DPython_EXECUTABLE="$PYTHON_EXEC" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    ..

  # Build the core library
  if [ "$VERBOSE" = true ]; then
    cmake --build . --target memory_graph_core --verbose
    cmake --build . --target test_memory_graph --verbose
  else
    cmake --build . --target memory_graph_core
    cmake --build . --target test_memory_graph
  fi

  print_success "C++ library built"
else
  print_info "Skipping C++ build (--skip-build)"
fi

# 5. Verify installation
print_header "Verifying installation"
cd "$PROJECT_ROOT/python"
source .venv/bin/activate

if python -c "from memory_graph import MemoryGraph, Node, Edge; print('[X] Import successful!')" 2>/dev/null; then
  print_success "Python import successful"
else
  print_error "[!] Python import failed"
  exit 1
fi

# ============================================================================
# 6. Run tests
# ============================================================================
if [ "$SKIP_TESTS" = false ]; then
  print_header "Running tests"

  # Run C++ tests
  if [ "$RUN_CTESTS" = true ] || [ -z "$RUN_CTESTS" ] && [ -z "$RUN_PYTESTS" ]; then
    cd "$PROJECT_ROOT/build"

    if [ -f "./bin/test_memory_graph" ]; then
      print_header "Running C++ tests"

      if [ "$VERBOSE" = true ]; then
        print_info "Verbose output enabled"
      fi

      if [ -n "$TEST_FILTER" ]; then
        print_info "Filter: $TEST_FILTER"
        ./bin/test_memory_graph --gtest_filter="$TEST_FILTER"
      else
        ctest -V
      fi

      print_success "C++ tests complete"
    else
      print_warning "test_memory_graph executable not found. Build may have failed."
    fi
  fi

  # Run Python tests
  if [ "$RUN_PYTESTS" = true ] || [ -z "$RUN_CTESTS" ] && [ -z "$RUN_PYTESTS" ]; then
    cd "$PROJECT_ROOT/python"
    source .venv/bin/activate

    if [ -f "tests/test_cogito_agent.py" ] || [ -d "tests" ]; then
      print_header "Running Python tests"

      if [ -n "$TEST_FILTER" ]; then
        print_info "Filter: $TEST_FILTER"
        uv run pytest tests/ -k "$TEST_FILTER" -v
      else
        uv run pytest tests/ -v
      fi

      print_success "Python tests complete"
    else
      print_warning "No Python tests found in tests/ directory"
    fi
  fi
else
  print_info "Skipping tests (--skip-tests)"
fi

# 7. Summary
cd "$PROJECT_ROOT"

print_header "Setup Complete!"

echo ""
echo "To activate the virtual environment:"
echo "  source python/.venv/bin/activate"
echo ""
echo "To run Python with the environment:"
echo "  cd python && source .venv/bin/activate && python"
echo ""
echo "To rebuild the C++ bindings:"
echo "  cd build && cmake --build . --target memory_graph_core"
echo ""
echo "To run C++ tests:"
echo "  cd build && ./bin/test_memory_graph"
echo ""
echo "To run Python tests:"
echo "  cd python && uv run pytest tests/ -v"
echo ""
echo "To run this script again with options:"
echo "  ./setup.sh --ctest --pytest --clean"
echo ""
echo "To run with a test filter:"
echo "  ./setup.sh --ctest --test-filter MemoryGraphTest.*"
echo ""

if [ "$RUN_CTESTS" = true ] || [ "$RUN_PYTESTS" = true ] || [ "$SKIP_TESTS" = false ]; then
  print_success "All tasks completed successfully!"
else
  print_info "Setup completed. Use --ctest and/or --pytest to run tests."
fi
