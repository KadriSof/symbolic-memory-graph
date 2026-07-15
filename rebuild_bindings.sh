#!/bin/bash
# rebuild_bindings.sh - Quick rebuild for C++ bindings

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "PROJECT_ROOT"

echo "Building C++ bindings..."
cd build
cmake --build . --target memory_graph_core

echo ".so file auto-copied to python/src/memory_graph/"
echo "Done!"
