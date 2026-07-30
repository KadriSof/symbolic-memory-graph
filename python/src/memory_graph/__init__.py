"""
MemoryGraph - Python bindings for C++ memory graph library
"""

import sys
from pathlib import Path

# Add the current directory to Python's path for the .so file
_current_dir = Path(__file__).parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

try:
    from memory_graph_core import (
        AsymmetricConnections,
        Edge,
        EdgeType,
        MemoryGraph,
        Node,
        SymmetricConnections,
    )
except ImportError as e:
    raise ImportError(
        "MemoryGraph core module not found. Please build the C++ bindings:\n"
        "  cd build && cmake -DBUILD_PYTHON_BINDINGS=ON .. && cmake --build . --target memory_graph_core\n"
        "  cp build/bindings/memory_graph_core*.so python/src/memory_graph/\n"
        f"  Original error: {e}"
    ) from e

# [!] Import order must be kept this way to avoid circulair dependency
from .core.representation import GraphRepresentation  # noqa: E402

# Import core utilities (still in progress..)
# from .core import (
# GraphBridge,
# GraphRepresentation,
# GraphConstructor,
# GraphSerializer,
# )


__all__ = [
    # C++ bindings
    "MemoryGraph",
    "Node",
    "Edge",
    "EdgeType",
    "SymmetricConnections",
    "AsymmetricConnections",
    # Core utilities
    # "GraphBridge",
    "GraphRepresentation",
    # "GraphConstructor",
    # "GraphSerializer",
]

__version__ = "0.1.0"
