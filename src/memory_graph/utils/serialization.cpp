#include "memory_graph/utils/serialization.hpp"
#include "memory_graph/edge.hpp"
#include "memory_graph/memory_graph.hpp"
#include "memory_graph/node.hpp"
#include "nlohmann/json.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include <zlib.h>
// #include <lz4.h>  // TODO: Implement the LZ4 later

namespace memory_graph::utils {
// Binary Serialization Implementation

/**
 * Binary format structure (for now):
 * [Header]
 * - Magic bytes: "MEMG" (4 bytes)
 * - Version: uint32_t
 * - Flags: uint32_t
 * - Node count: uint32_t
 * - Edge count: uint32_t
 * - Edge count: uint32_t
 * - Total size: uint32_t
 * [Data]
 * - Nodes: JSON (one per node)
 * - Edges: JSON (one per edge)
 * - Metadata: JSON
 */

namespace {
constexpr uint32_t MAGIC = 0x474D454D; // "MEMG" in little-endian
constexpr uint32_t FLAG_HAS_METADATA = 1 << 0;
constexpr uint32_t FLAG_HAS_NODES = 1 << 1;
constexpr uint32_t FLAG_HAS_EDGES = 1 << 2;

struct BinaryHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t flags;
  uint32_t node_count;
  uint32_t edge_count;
  uint32_t data_size;

  // Total header size: 24 bytes
  static constexpr size_t SIZE = 6 * sizeof(uint32_t);
};
} // namespace

std::vector<uint8_t> toBinary(const MemoryGraph &graph,
                              const SerializationOptions &options) {
  // 1. We get the JSON representation (no Jasons were harmed during this
  // serialization)
  nlohmann::json graphJson = graph.toJson();

  // 2. Build data buffer (JSON as string)
  std::string jsonString = graphJson.dump();
  std::vector<uint8_t> data(jsonString.begin(), jsonString.end());

  // 3. Build header
  BinaryHeader header;
  header.magic = MAGIC;
  header.version = options.version;
  header.flags = 0;
  header.node_count = graph.getNodes().size();
  header.edge_count = graph.getEdges().size();
  header.data_size = data.size();

  if (options.include_metadata)
    header.flags |= FLAG_HAS_METADATA;
  if (options.include_nodes)
    header.flags |= FLAG_HAS_NODES;
  if (options.include_edges)
    header.flags |= FLAG_HAS_EDGES;

  // 4. Assembke binary output
  std::vector<uint8_t> output;
  output.reserve(BinaryHeader::SIZE + data.size());

  // Write header
  auto writeU32 = [&output](uint32_t value) {
    output.push_back(value & 0XFF);
    output.push_back((value >> 8) & 0XFF);
    output.push_back((value >> 16) & 0XFF);
    output.push_back((value >> 24) & 0XFF);
  };

  writeU32(header.magic);
  writeU32(header.version);
  writeU32(header.flags);
  writeU32(header.node_count);
  writeU32(header.edge_count);
  writeU32(header.data_size);

  // Write data
  output.insert(output.end(), data.begin(), data.end());

  // 5. Compress if requested
  if (options.compression != CompressionType::NONE) {
    return compress(output, options.compression);
  }

  return output;
}

// Helper func for testing and debuging
bool isCompressed(const std::vector<uint8_t> &data) {
  if (data.size() < 2)
    return false;

  // Check for zlib header
  if (data[0] == 0x78 &&
      (data[1] == 0x01 || data[1] == 0x5E || data[1] == 0x9C)) {
    return true;
  }

  // Check for gzip header
  if (data.size() >= 2 && data[0] == 0x1F && data[1] == 0x8B) {
    return true;
  }

  return false;
}
MemoryGraph fromBinary(const std::vector<uint8_t> &data) {
  // 1. Decompress if needed
  std::vector<uint8_t> decompressedData;
  const uint8_t *rawData = data.data();
  size_t rawSize = data.size();

  // Check if data is compressed
  bool isCompressed = false;

  // Method 1: Check for zlib header (0x78 0x01, 0x78 0x5E, 0x78 0x9C)
  if (rawSize > 2 && rawData[0] == 0x78) {
    if (rawData[1] == 0x01 || rawData[1] == 0x5E || rawData[1] == 0x9C) {
      isCompressed = true;
    }
  }

  // Method 2: Check for gzip header (0x1F 0x8B)
  if (!isCompressed && rawSize > 2 && rawData[0] == 0x1F &&
      rawData[1] == 0x8B) {
    isCompressed = true;
  }

  // Method 3: Check for magic bytes (if present, it's uncompressed)
  if (rawSize > 4 && rawData[0] == 'M' && rawData[1] == 'E' &&
      rawData[2] == 'M' && rawData[3] == 'G') {
    isCompressed = false;
  }

  // Decompress
  if (isCompressed) {
    try {
      decompressedData = decompress(data);
      rawData = decompressedData.data();
      rawSize = decompressedData.size();
    } catch (const std::exception &e) {
      throw std::runtime_error(
          "[serialization:fromBinary] Failed to decompress data: " +
          std::string(e.what()));
    }
  }

  // 2. Parse header
  if (rawSize < BinaryHeader::SIZE) {
    throw std::runtime_error(
        "[serialization:fromBinary] Invalid binary data: too small");
  }

  auto readU32 = [&rawData, &rawSize](size_t &offset) -> uint32_t {
    if (offset + 4 > rawSize) {
      throw std::runtime_error(
          "[serialization:fromBinary] Invalid binary data: unexpected EOF");
    }
    uint32_t value = static_cast<uint32_t>(rawData[offset]) |
                     (static_cast<uint32_t>(rawData[offset + 1]) << 8) |
                     (static_cast<uint32_t>(rawData[offset + 2]) << 16) |
                     (static_cast<uint32_t>(rawData[offset + 3]) << 24);
    offset += 4;
    return value;
  };

  size_t offset = 0;
  uint32_t magic = readU32(offset);
  uint32_t version = readU32(offset);
  uint32_t flags = readU32(offset);
  uint32_t node_count = readU32(offset);
  uint32_t edge_count = readU32(offset);
  uint32_t data_size = readU32(offset);

  // 3. Validate
  if (magic != MAGIC) {
    throw std::runtime_error("[serialization:fromBinary] Invalid binary data: "
                             "incorrect magic bytes");
  }

  if (version != SERIALIZATION_VERSION) {
    throw std::runtime_error("Version mismatch: expected " +
                             std::to_string(SERIALIZATION_VERSION) + ", got " +
                             std::to_string(version));
  }

  // 4. Parse JSON data
  std::string jsonString(reinterpret_cast<const char *>(rawData + offset),
                         data_size);
  nlohmann::json graphJson = nlohmann::json::parse(jsonString);

  // 5. Reconstruct graph
  return MemoryGraph::fromJson(graphJson);
}

// Incremental Serialization (Deltas)
nlohmann::json computeDelta(const MemoryGraph &before,
                            const MemoryGraph &after) {
  nlohmann::json delta;

  // Helper: Get node IDs from graph
  auto getNodeIds =
      [](const MemoryGraph &graph) -> std::unordered_set<std::string> {
    std::unordered_set<std::string> ids;
    for (const auto &node : graph.getNodes()) {
      ids.insert(node.getId());
    }
    return ids;
  };

  // Helper: Get edge IDs from graph
  auto getEdgeIds =
      [](const MemoryGraph &graph) -> std::unordered_set<std::string> {
    std::unordered_set<std::string> ids;
    for (const auto &edge : graph.getEdges()) {
      ids.insert(edge.getId());
    }
    return ids;
  };

  // 1. Node changes
  auto beforeNodes = getNodeIds(before);
  auto afterNodes = getNodeIds(after);

  std::vector<std::string> addedNodes;
  std::vector<std::string> removedNodes;

  // Find added nodes
  for (const auto &id : afterNodes) {
    if (beforeNodes.find(id) == beforeNodes.end()) {
      addedNodes.push_back(id);
    }
  }
  // Find removed nodes
  for (const auto &id : beforeNodes) {
    if (afterNodes.find(id) == afterNodes.end()) {
      removedNodes.push_back(id);
    }
  }

  // Check for modified nodes
  nlohmann::json modifiedNodes = nlohmann::json::object();
  for (const auto &id : afterNodes) {
    if (beforeNodes.find(id) != beforeNodes.end()) {
      const Node &beforeNode = before.getNode(id);
      const Node &afterNode = after.getNode(id);

      // Compare (simplified - check if JSON representation differ)
      if (beforeNode.toJson() != afterNode.toJson()) {
        modifiedNodes[id] = afterNode.toJson();
      }
    }
  }

  // 2. Edge changes
  auto beforeEdges = getEdgeIds(before);
  auto afterEdges = getEdgeIds(after);

  std::vector<std::string> addedEdges;
  std::vector<std::string> removedEdges;

  // Find add edges
  for (const auto &id : afterEdges) {
    if (beforeEdges.find(id) == beforeEdges.end()) {
      addedEdges.push_back(id);
    }
  }
  // Find remvoed edges
  for (const auto &id : beforeEdges) {
    if (afterEdges.find(id) == afterEdges.end()) {
      removedEdges.push_back(id);
    }
  }

  // Check for modified edges
  nlohmann::json modifiedEdges = nlohmann::json::object();
  for (const auto &id : afterEdges) {
    if (beforeEdges.find(id) != beforeEdges.end()) {
      const Edge &beforeEdge = before.getEdge(id);
      const Edge &afterEdge = after.getEdge(id);

      if (beforeEdge.toJson() != afterEdge.toJson()) {
        modifiedEdges[id] = afterEdge.toJson();
      }
    }
  }

  // 3. Build delta
  // Nodes:
  if (!addedNodes.empty())
    delta["added_nodes"] = addedNodes;
  if (!removedNodes.empty())
    delta["removed_nodes"] = removedNodes;
  if (!modifiedNodes.empty())
    delta["modified_nodes"] = modifiedNodes;
  // Edges:
  if (!addedEdges.empty())
    delta["added_edges"] = addedEdges;
  if (!removedEdges.empty())
    delta["removed_edges"] = removedEdges;
  if (!modifiedEdges.empty())
    delta["modified_edges"] = modifiedEdges;
  // Metadata:
  if (before.getMetadata() != after.getMetadata()) {
    delta["metadata"] = after.getMetadata();
  }

  return delta;
}

void applyDelta(MemoryGraph &graph, const nlohmann::json &delta) {
  // 1. Apply metadata changes
  if (delta.contains("metadata")) {
    graph.setMetadata(delta["metadata"]);
  }

  // 2. Remove nodes
  if (delta.contains("removed_nodes")) {
    for (const auto &nodeId : delta["removed_nodes"]) {
      if (graph.hasNode(nodeId)) {
        graph.removeNode(nodeId);
      }
    }
  }

  // 3. Remove edges
  if (delta.contains("removed_edges")) {
    for (const auto &edgeId : delta["removed_edges"]) {
      if (graph.hasEdge(edgeId)) {
        graph.removeEdge(edgeId);
      }
    }
  }

  // 4. Add nodes
  if (delta.contains("added_nodes")) {
    // TODO: Needs to be adjusted to store full node data in delta
    for (const auto &nodeId : delta["added_nodes"]) {
      if (delta.contains("modified_nodes") &&
          delta["modified_nodes"].contains(nodeId)) {
        Node node = Node::fromJson(delta["modified_nodes"][nodeId]);
        graph.addNode(node);
      }
    }
  }

  // 5. Add/update edges
  if (delta.contains("modified_edeges")) {
    for (auto it = delta["modified_edges"].begin();
         it != delta["modified_edeges"].end(); ++it) {
      Edge edge = Edge::fromJson(it.value());
      if (!graph.hasEdge(edge.getId())) {
        graph.addEdge(edge);
      }

      // If edge exists, we would need to update it (not implemented yet)
      // TODO: for complete implementation, we would need to store full
      // node/edge data in the data delta for additions and updates.
    }
  }
}

void applyDeltaBinary(MemoryGraph &graph,
                      const std::vector<uint8_t> &deltaData) {
  // Decompress if needed
  if (deltaData.empty()) {
    throw std::runtime_error(
        "[utils:serialization] applyDeltaBinary: empty delta data");
  }

  std::vector<uint8_t> decompressed;
  const uint8_t *data = deltaData.data();
  size_t size = deltaData.size();

  bool isCompressed = false;
  if (size > 2 && data[0] == 0x78) {
    if (data[1] == 0x01 || data[1] == 0x5E || data[1] == 0x9C) {
      isCompressed = true;
    }
  }

  if (isCompressed) {
    decompressed = decompress(deltaData);
    data = decompressed.data();
    size = decompressed.size();
  }

  // Parse JSON delta
  if (size == 0) {
    throw std::runtime_error("[utils:serialization] applyDeltaBinary: empty "
                             "data after decompression");
  }

  std::string jsonString(reinterpret_cast<const char *>(data), size);
  nlohmann::json delta = nlohmann::json::parse(jsonString);

  applyDelta(graph, delta);
}

std::vector<uint8_t> compress(const std::vector<uint8_t> &data,
                              CompressionType type) {
  if (type == CompressionType::NONE) {
    return data;
  }

  // TODO: Implement the rest of compression types later (with proper wrappers
  // and error handling)
  if (type == CompressionType::ZLIB) {
    // For now, we will jus return uncompressed data with a marker
    std::vector<uint8_t> result;
    result.reserve(data.size() + 2);
    result.push_back(0x78); // ZLIB header
    result.push_back(0x01); // ZLIB header
    result.insert(result.end(), data.begin(), data.end());
    return result;
  }

  // Unsupported compression type
  throw std::runtime_error(
      "[utils:serialization] Compression type not supported");
}

std::vector<uint8_t> decompress(const std::vector<uint8_t> &data) {
  // Check ZLIB header
  if (data.size() < 2) {
    throw std::runtime_error(
        "[utils:serialization] Invalide compressed data: data too small");
  }

  if (data[0] != 0x78 ||
      data[1] != 0x01 && data[1] != 0x5E && data[1] != 0x9C) {
    throw std::runtime_error(
        "[utils:serialization] Invalid compressed data: incorrect zlib header");
  }

  // TODO: we need a ZLIB's inflate here
  std::vector<uint8_t> result(data.size() - 2);
  std::memcpy(result.data(), data.data() + 2, result.size());
  return result;
}

// Version Management
uint32_t getVersion(const std::vector<uint8_t> &data) {
  std::vector<uint8_t> decompressedData;
  const uint8_t *rawData = data.data();
  size_t rawSize = data.size();

  // Check if compressed
  bool isCompressed =
      (rawSize > 2 && rawData[0] == 0x78 &&
       (rawData[1] == 0x01 || rawData[1] == 0x5E || rawData[1] == 0x9C));

  if (isCompressed) {
    decompressedData = decompress(data);
    rawData = decompressedData.data();
    rawSize = decompressedData.size();
  }

  if (rawSize < BinaryHeader::SIZE) {
    throw std::runtime_error(
        "[utils:serialization] Invalide binary data: too small");
  }

  // Read version from header (4 bytes at offset 4)
  if (rawSize < 8) {
    throw std::runtime_error("Invalid binary data");
  }

  uint32_t version =
      rawData[4] | (rawData[5] << 8) | (rawData[6] << 16) | (rawData[7] << 24);
  return version;
}

bool isValidFormat(const std::vector<uint8_t> &data) {
  try {
    uint32_t version = getVersion(data);
    return version >= 1 && version <= SERIALIZATION_VERSION;
  } catch (...) {
    return false;
  }
}

} // namespace memory_graph::utils
