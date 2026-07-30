"""
Core graph utilities for MemoryGraph.

Provides bidirectional conversion between MemoryGraph and:
- JSON (structured data)
- Linearized text (LLM-readable)
- Triples (subject → predicate → object)
- Cypher-like queries

Also provides serialization for persistence.
"""

# In progress..
# from .bridge import GraphBridge
from .representation import GraphRepresentation

# from .construction import GraphConstructor
from .serialization import GraphSerializer

__all__ = [
    # "GraphBridge",
    "GraphRepresentation",
    # "GraphConstructor",
    "GraphSerializer",
]
