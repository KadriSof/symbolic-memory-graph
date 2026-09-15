"""
Graph Serialization Utilities for MemoryGraph.

Provides high-level wrappers around the C++ binary serialization and delta
computation, enabling efficient persistence and state synchronization.
"""

from typing import Any, Dict

from memory_graph.memory_graph_core import (
    MemoryGraph,
    CompressionType,
    SerializationOptions,
    to_binary as _cpp_to_binary,
    from_binary as _cpp_from_binary,
    compute_delta as _cpp_compute_delta,
    apply_delta as _cpp_apply_delta,
    apply_delta_binary as _cpp_apply_delta_binary,
    compress as _cpp_compress,
    decompress as _cpp_decompress,
    get_version as _cpp_get_version,
    is_valid_format as _cpp_is_valid_format,
)


class GraphSerializer:
    """
    High-level wrapper for MemoryGraph serialization and delta operations.

    This class provides a clean, Pythonic API for saving, loading, and
    synchronizing graph state, leveraging optimized C++ binary formats.
    """

    @staticmethod
    def save_to_file(graph: MemoryGraph, filepath: str, compress: bool = True) -> None:
        """
        Save a MemoryGraph to a binary file.

        Args:
            graph: The MemoryGraph to save.
            filepath: The destination file path.
            compress: Whether to compress the binary data (uses ZLIB by default).
        """
        options = SerializationOptions()
        options.include_metadata = True
        options.include_edges = True
        options.compression = CompressionType.ZLIB if compress else CompressionType.NONE
        options.version = 1

        binary_data = _cpp_to_binary(graph, options)

        with open(filepath, "wb") as f:
            f.write(binary_data)

    @staticmethod
    def load_from_file(filepath: str) -> MemoryGraph:
        """
        Load a MemoryGraph from a binary file.

        Args:
            filepath: The source file path.

        Returns:
            A reconstructed MemoryGraph.
        """
        with open(filepath, "rb") as f:
            binary_data = f.read()

        if not _cpp_is_valid_format(binary_data):
            raise ValueError(f"Invalid or corrupted graph file: {filepath}")

        return _cpp_from_binary(binary_data)

    @staticmethod
    def to_binary(graph: MemoryGraph, compress: bool = True) -> bytes:
        """
        Convert graph to binary bytes (convenience wrapper).

        Args:
            graph: The MemoryGraph to serialize.
            compress: Whether to compress the binary data.

        Returns:
            Binary data as bytes.

        Raises:
            TypeError: If the C++ function returns an unsupported type.
        """
        options = SerializationOptions()
        options.include_metadata = True
        options.include_edges = True
        options.compression = CompressionType.ZLIB if compress else CompressionType.NONE
        options.version = 1

        result = _cpp_to_binary(graph, options)

        # Ensure we return bytes, not a list
        if isinstance(result, list):
            return bytes(result)
        if isinstance(result, bytearray):
            return bytes(result)
        if isinstance(result, memoryview):
            return bytes(result)
        if isinstance(result, bytes):
            return result

        raise TypeError(f"Unexpected type from C++ to_binary: {type(result)}")

    @staticmethod
    def from_binary(data: bytes) -> MemoryGraph:
        """
        Convert binary bytes to graph.

        Args:
            data: Binary data to deserialize.

        Returns:
            Reconstructed MemoryGraph.

        Raises:
            ValueError: If the data is invalid or corrupted.
        """
        # Ensure we have bytes
        if isinstance(data, list):
            data = bytes(data)
        elif isinstance(data, bytearray):
            data = bytes(data)
        elif isinstance(data, memoryview):
            data = bytes(data)
        elif not isinstance(data, bytes):
            raise TypeError(f"Expected bytes-like object, got {type(data)}")

        # Convert bytes to list for compatibility
        data_list = list(data)

        # Validate
        try:
            if not _cpp_is_valid_format(data_list):
                raise ValueError("Invalid or corrupted binary data")
        except TypeError:
            if not _cpp_is_valid_format(data):
                raise ValueError("Invalid or corrupted binary data")

        return _cpp_from_binary(data_list)

    @staticmethod
    def compute_delta(before: MemoryGraph, after: MemoryGraph) -> Dict[str, Any]:
        """
        Compute the difference between two graph states.

        This is highly efficient and returns a structured dictionary representing
        added, modified, and removed nodes/edges.

        Args:
            before: The previous state of the graph.
            after: The new state of the graph.

        Returns:
            A dictionary representing the delta.
        """
        return _cpp_compute_delta(before, after)

    @staticmethod
    def apply_delta(graph: MemoryGraph, delta: Dict[str, Any]) -> None:
        """
        Apply a computed delta to a graph in-place.

        Args:
            graph: The graph to update.
            delta: The delta dictionary (from compute_delta).
        """
        _cpp_apply_delta(graph, delta)

    @staticmethod
    def apply_delta_binary(graph: MemoryGraph, delta_data: bytes) -> None:
        """
        Apply a binary-encoded delta to a graph in-place.

        Useful for network synchronization or loading incremental updates.

        Args:
            graph: The graph to update.
            delta_data: The binary delta payload.
        """
        _cpp_apply_delta_binary(graph, delta_data)

    @staticmethod
    def compress_data(
        data: bytes, compression_type: int = CompressionType.ZLIB
    ) -> bytes:
        """Compress arbitrary binary data."""
        return _cpp_compress(data, compression_type)

    @staticmethod
    def decompress_data(data: bytes) -> bytes:
        """Decompress arbitrary binary data."""
        return _cpp_decompress(data)

    @staticmethod
    def get_file_version(filepath: str) -> int:
        """Get the serialization version of a binary file without fully loading it."""
        with open(filepath, "rb") as f:
            # Read just enough bytes to check the header
            header = f.read(16)
            return _cpp_get_version(header)
