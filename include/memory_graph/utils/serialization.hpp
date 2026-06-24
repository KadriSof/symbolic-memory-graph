#pragma once

#include "memory_graph/memory_graph.hpp"
#include <cstddef>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <vector>

namespace memory_graph::utils {

/**
 * @brief Serialization version for format compatibility
 */
constexpr uint32_t SERIALIZATION_VERSION = 1;

/**
 * @brief Compression alogrithms supported
 */
enum class CompressionType {
  NONE = 0, // No compression
  ZLIB = 1, // zlib/deflate (good balance)
  LZ4 = 2,  // LZ4 (very fast, good compression)
};

/**
 * @brief Check if binary data is compressed
 * @param data Binary data to check
 * @return true if data appears to be compressed
 */
bool isCompressed(const std::vector<uint8_t> &data);

/**
 * @brief Options for serialization
 */
struct SerializationOptions {
  bool include_metadata = true;
  bool include_nodes = true;
  bool include_edges = true;
  CompressionType compression = CompressionType::NONE;
  uint32_t version = SERIALIZATION_VERSION;
};

// Core Serialization
/**
 * @brief Serialize a MemoryGraph to binary format
 * @param options Serialization options
 * @param Binary data as vector of bytes
 */
std::vector<uint8_t> toBinary(const MemoryGraph &graph,
                              const SerializationOptions &options = {});

/**
 * @brief Deserialize a MemoryGraph from binary format
 * @param data Binary data to deserialize
 * @return Deserialize graph
 * @throws std::runtime_error if data is corrupted or version mismatch
 */
MemoryGraph fromBinary(const std::vector<uint8_t> &data);

// Incremental Serialization (Deltas)
/**
 * @brief Compute the difference between two graphs (changes since last save)
 * @param before The previous state (last saved)
 * @param after The current state
 * @return JSON delta describing changes
 */
nlohmann::json computeDelta(const MemoryGraph &before,
                            const MemoryGraph &after);

/**
 * @brief Apply a delta to a graph (update from saved state to current)
 * @param graph The graph to update
 * @param delta The delta to Apply
 * @throws std::runtime_error if delta is invalid
 */
void applyDelta(MemoryGraph &graph, const nlohmann::json &delta);

/**
 * @brief Apply a delta from binary format
 * @param graph The graph to update
 * @param deltaData Binary delta data
 */
void applyDeltaBinary(MemoryGraph &graph,
                      const std::vector<uint8_t> &deltaData);

// Compression Helpers
/**
 * @brief Compress binary data
 * @param data Data to compress
 * @param type Compression algorithm to use
 * @return Compressed data
 */
std::vector<uint8_t> compress(const std::vector<uint8_t> &data,
                              CompressionType type = CompressionType::ZLIB);

/**
 * @brief Decompress binary data
 * @param data Compressed data
 * @return Decompressed data
 * @throws std::runtime_error if data is corrupted
 */
std::vector<uint8_t> decompress(const std::vector<uint8_t> &data);

/**
 * @brief Get compression ratio
 * @param original Original data size
 * @param compressed Compressed data size
 * @return Compression ratio (0.0 - 1.0, lower is better)
 */
inline float compressionRatio(size_t original, size_t compressed) {
  return compressed / static_cast<float>(original);
}

// Version Management
/**
 * @brief Get the serialization version from binary data
 * @param data Binary data
 * @return Version number
 * @throws std::runtime_error if data is invalid
 */
uint32_t getVersion(const std::vector<uint8_t> &data);

/**
 * @brief Check if binary data is valid for this version
 * @param data Binary data
 * @return true if valid
 */
bool isValidFormat(const std::vector<uint8_t> &data);

} // namespace memory_graph::utils
