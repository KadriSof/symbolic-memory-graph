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
#include <variant>
#include <vector>
#include <zconf.h>
#include <zlib.h>

namespace memory_graph::utils {

namespace {
constexpr uint32_t MAGIC = 0x474D454D; // "MEMG" in little-endian
constexpr uint32_t FLAG_HAS_METADATA = 1 << 0;
constexpr uint32_t FLAG_HAS_NODES = 1 << 1;
constexpr uint32_t FLAG_HAS_EDGES = 1 << 2;
constexpr uint32_t FLAG_MODE_KNOWLEDGE = 1 << 3;

struct BinaryHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t flags;
  uint32_t node_count;
  uint32_t edge_count;
  uint32_t data_size;

  static constexpr size_t SIZE = 6 * sizeof(uint32_t);
};

void writeU32(std::vector<uint8_t> &output, uint32_t value) {
  output.push_back(value & 0xFF);
  output.push_back((value >> 8) & 0xFF);
  output.push_back((value >> 16) & 0xFF);
  output.push_back((value >> 24) & 0xFF);
}

uint32_t readU32(const std::uint8_t *rawData, size_t rawSize, size_t &offset) {
  if (offset + 4 > rawSize) {
    throw std::runtime_error(
        "[serialization:readU32] Invalid binary data: unexpected EOF");
  }
  uint32_t value = static_cast<uint32_t>(rawData[offset]) |
                   (static_cast<uint32_t>(rawData[offset + 1]) << 8) |
                   (static_cast<uint32_t>(rawData[offset + 2]) << 16) |
                   (static_cast<uint32_t>(rawData[offset + 3]) << 24);
  offset += 4;
  return value;
}

CompressionType detectCompressionType(const std::vector<uint8_t> &data) {
  if (data.size() < 2)
    return CompressionType::NONE;

  if (data.size() >= 4 && data[0] == 'M' && data[1] == 'E' && data[2] == 'M' &&
      data[3] == 'G') {
    return CompressionType::NONE;
  }

  if (data[0] == 0x78 && (data[1] == 0x01 || data[1] == 0x5E ||
                          data[1] == 0x9C || data[1] == 0xDA)) {
    return CompressionType::ZLIB;
  }

  return CompressionType::NONE;
}
} // namespace

bool isCompressed(const std::vector<uint8_t> &data) {
  return detectCompressionType(data) != CompressionType::NONE;
}

std::vector<uint8_t> toBinary(const MemoryGraph &graph,
                              const SerializationOptions &options) {
  nlohmann::json graphJson = graph.toJson();
  BinaryHeader header;

  header.magic = MAGIC;
  header.version = options.version;
  header.flags = 0;

  header.flags |= FLAG_HAS_NODES;
  header.node_count = graph.getNodesRef().size();

  if (options.include_edges) {
    header.flags |= FLAG_HAS_EDGES;
    header.edge_count = graph.getEdgesRef().size();
  } else {
    header.edge_count = 0;
    if (graphJson.contains("edges"))
      graphJson["edges"] = nlohmann::json::array();
  }

  if (options.include_metadata) {
    header.flags |= FLAG_HAS_METADATA;
  } else {
    if (graphJson.contains("metadata"))
      graphJson.erase("metadata");
  }

  if (graph.getMode() == GraphMode::KNOWLEDGE) {
    header.flags |= FLAG_MODE_KNOWLEDGE;
  }

  std::string jsonString = graphJson.dump();
  std::vector<uint8_t> data(jsonString.begin(), jsonString.end());
  header.data_size = data.size();

  std::vector<uint8_t> output;
  output.reserve(BinaryHeader::SIZE + data.size());

  writeU32(output, header.magic);
  writeU32(output, header.version);
  writeU32(output, header.flags);
  writeU32(output, header.node_count);
  writeU32(output, header.edge_count);
  writeU32(output, header.data_size);

  output.insert(output.end(), data.begin(), data.end());

  if (options.compression != CompressionType::NONE) {
    return compress(output, options.compression);
  }

  return output;
}

MemoryGraph fromBinary(const std::vector<uint8_t> &data) {
  std::vector<uint8_t> decompressedData;
  const uint8_t *rawData = data.data();
  size_t rawSize = data.size();

  CompressionType compType = detectCompressionType(data);
  if (compType != CompressionType::NONE) {
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

  if (rawSize < BinaryHeader::SIZE) {
    throw std::runtime_error(
        "[serialization:fromBinary] Invalid binary data: too small");
  }

  size_t offset = 0;
  uint32_t magic = readU32(rawData, rawSize, offset);
  uint32_t version = readU32(rawData, rawSize, offset);
  uint32_t flags = readU32(rawData, rawSize, offset);
  uint32_t node_count = readU32(rawData, rawSize, offset);
  uint32_t edge_count = readU32(rawData, rawSize, offset);
  uint32_t data_size = readU32(rawData, rawSize, offset);

  if (magic != MAGIC) {
    throw std::runtime_error("[serialization:fromBinary] Invalid binary data: "
                             "incorrect magic bytes");
  }

  if (version != SERIALIZATION_VERSION) {
    throw std::runtime_error("Version mismatch: expected " +
                             std::to_string(SERIALIZATION_VERSION) + ", got " +
                             std::to_string(version));
  }

  if (data_size > rawSize - offset) {
    throw std::runtime_error(
        "[serialization:fromBinary] Invalid binary data: data_size (" +
        std::to_string(data_size) + ") exceeds remaining buffer size (" +
        std::to_string(rawSize - offset) + ")");
  }

  const size_t MAX_DATA_SIZE = 100 * 1024 * 1024;
  if (data_size > MAX_DATA_SIZE) {
    throw std::runtime_error(
        "[serialization:binary] Invalid binary data: data_size (" +
        std::to_string(data_size) + ") exceeds maximum allowed (" +
        std::to_string(MAX_DATA_SIZE) + ")");
  }

  std::string jsonString(reinterpret_cast<const char *>(rawData + offset),
                         data_size);
  nlohmann::json graphJson;

  try {
    graphJson = nlohmann::json::parse(jsonString);
  } catch (const nlohmann::json::parse_error &e) {
    throw std::runtime_error(
        "[serialization:fromBinary] Failed to parse JSON: " +
        std::string(e.what()));
  }

  MemoryGraph graph = MemoryGraph::fromJson(graphJson);

  if (!(flags & FLAG_MODE_KNOWLEDGE)) {
    graph.setMode(GraphMode::SYMBOLIC);
  } else {
    graph.setMode(GraphMode::KNOWLEDGE);
  }

  return graph;
}

// Incremental Serialization (Deltas)
nlohmann::json computeDelta(const MemoryGraph &before,
                            const MemoryGraph &after) {
  nlohmann::json delta;

  // Helper: Get node IDs from graph
  auto getNodeIds =
      [](const MemoryGraph &graph) -> std::unordered_set<std::string> {
    std::unordered_set<std::string> ids;
    for (const auto &[id, _] : graph.getNodesRef()) {
      ids.insert(id);
    }
    return ids;
  };

  // Helper: Get edge IDs from graph
  auto getEdgeIds =
      [](const MemoryGraph &graph) -> std::unordered_set<std::string> {
    std::unordered_set<std::string> ids;
    for (const auto &[id, _] : graph.getEdgesRef()) {
      ids.insert(id);
    }
    return ids;
  };

  auto beforeNodes = getNodeIds(before);
  auto afterNodes = getNodeIds(after);

  // Find added nodes
  nlohmann::json addedNodes = nlohmann::json::object();
  for (const auto &id : afterNodes) {
    if (beforeNodes.find(id) == beforeNodes.end()) {
      addedNodes[id] = after.getNode(id).toJson();
    }
  }

  // Find removed nodes
  std::vector<std::string> removedNodes;
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
      if (beforeNode.toJson() != afterNode.toJson()) {
        modifiedNodes[id] = afterNode.toJson();
      }
    }
  }

  auto beforeEdges = getEdgeIds(before);
  auto afterEdges = getEdgeIds(after);

  // Find added edges
  nlohmann::json addedEdges = nlohmann::json::object();
  for (const auto &id : afterEdges) {
    if (beforeEdges.find(id) == beforeEdges.end()) {
      addedEdges[id] = after.getEdge(id).toJson();
    }
  }

  // Find removed edges
  std::vector<std::string> removedEdges;
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

  if (!addedNodes.empty())
    delta["added_nodes"] = addedNodes;
  if (!removedNodes.empty())
    delta["removed_nodes"] = removedNodes;
  if (!modifiedNodes.empty())
    delta["modified_nodes"] = modifiedNodes;

  if (!addedEdges.empty())
    delta["added_edges"] = addedEdges;
  if (!removedEdges.empty())
    delta["removed_edges"] = removedEdges;
  if (!modifiedEdges.empty())
    delta["modified_edges"] = modifiedEdges;

  if (before.getMode() != after.getMode()) {
    delta["mode"] =
        (after.getMode() == GraphMode::KNOWLEDGE) ? "KNOWLEDGE" : "SYMBOLIC";
  }

  if (before.getMetadata() != after.getMetadata()) {
    delta["metadata"] = after.getMetadata();
  }

  return delta;
}

void applyDelta(MemoryGraph &graph, const nlohmann::json &delta) {
  if (delta.contains("mode")) {
    std::string modeStr = delta["mode"].get<std::string>();
    graph.setMode(modeStr == "KNOWLEDGE" ? GraphMode::KNOWLEDGE
                                         : GraphMode::SYMBOLIC);
  }

  if (delta.contains("metadata")) {
    graph.setMetadata(delta["metadata"]);
  }

  if (delta.contains("removed_nodes")) {
    for (const auto &nodeId : delta["removed_nodes"]) {
      if (graph.hasNode(nodeId)) {
        graph.removeNode(nodeId);
      }
    }
  }

  if (delta.contains("removed_edges")) {
    for (const auto &edgeId : delta["removed_edges"]) {
      if (graph.hasEdge(edgeId)) {
        graph.removeEdge(edgeId);
      }
    }
  }

  if (delta.contains("added_nodes")) {
    for (auto it = delta["added_nodes"].begin();
         it != delta["added_nodes"].end(); ++it) {
      std::string nodeId = it.key();
      nlohmann::json nodeJson = it.value();
      Node node = Node::fromJson(nodeJson);
      graph.addNode(node);
    }
  }

  if (delta.contains("modified_nodes")) {
    for (auto it = delta["modified_nodes"].begin();
         it != delta["modified_nodes"].end(); ++it) {
      std::string nodeId = it.key();
      nlohmann::json nodeJson = it.value();

      if (graph.hasNode(nodeId)) {
        // Save incident edges before removing node
        std::vector<Edge> incidentEdges;
        for (const auto &[edgeId, edge] : graph.getEdgesRef()) {
          if (std::holds_alternative<SymmetricConnections>(
                  edge.getConnections())) {
            const auto &conn_set =
                std::get<SymmetricConnections>(edge.getConnections());
            if (conn_set.find(nodeId) != conn_set.end()) {
              incidentEdges.push_back(edge);
            }
          } else {
            const auto &conn_pair =
                std::get<AsymmetricConnections>(edge.getConnections());
            if (conn_pair.first == nodeId || conn_pair.second == nodeId) {
              incidentEdges.push_back(edge);
            }
          }
        }

        graph.removeNode(nodeId);
        Node node = Node::fromJson(nodeJson);
        graph.addNode(node);

        for (const auto &edge : incidentEdges) {
          graph.addEdge(edge);
        }
      }
    }
  }

  if (delta.contains("added_edges")) {
    for (auto it = delta["added_edges"].begin();
         it != delta["added_edges"].end(); ++it) {
      std::string edgeId = it.key();
      nlohmann::json edgeJson = it.value();

      if (!graph.hasEdge(edgeId)) {
        Edge edge = Edge::fromJson(edgeJson);
        graph.addEdge(edge);
      }
    }
  }

  if (delta.contains("modified_edges")) {
    for (auto it = delta["modified_edges"].begin();
         it != delta["modified_edges"].end(); ++it) {
      std::string edgeId = it.key();
      nlohmann::json edgeJson = it.value();

      if (graph.hasEdge(edgeId)) {
        graph.removeEdge(edgeId);
      }
      Edge edge = Edge::fromJson(edgeJson);
      graph.addEdge(edge);
    }
  }
}

void applyDeltaBinary(MemoryGraph &graph,
                      const std::vector<uint8_t> &deltaData) {
  if (deltaData.empty()) {
    throw std::runtime_error(
        "[utils:serialization] applyDeltaBinary: empty delta data");
  }

  std::vector<uint8_t> decompressed;
  const uint8_t *data = deltaData.data();
  size_t size = deltaData.size();

  CompressionType compType = detectCompressionType(deltaData);
  if (compType != CompressionType::NONE) {
    decompressed = decompress(deltaData);
    data = decompressed.data();
    size = decompressed.size();
  }

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

  if (type == CompressionType::ZLIB) {
    uLongf compressedSize = compressBound(data.size());
    std::vector<uint8_t> result(compressedSize);
    int ret = compress2(result.data(), &compressedSize, data.data(),
                        data.size(), Z_DEFAULT_COMPRESSION);
    if (ret != Z_OK) {
      throw std::runtime_error(
          "[serialization:compress] ZLIB compression failed: " +
          std::to_string(ret));
    }
    result.resize(compressedSize);
    return result;
  }

  throw std::runtime_error(
      "[utils:serialization] Compression type not supported");
}

std::vector<uint8_t> decompress(const std::vector<uint8_t> &data) {
  if (data.size() < 2) {
    throw std::runtime_error("[serialization:decompress] Invalid compressed "
                             "data: data size too small");
  }

  if (data[0] != 0x78 || (data[1] != 0x01 && data[1] != 0x5E &&
                          data[1] != 0x9C && data[1] != 0XDA)) {
    throw std::runtime_error("serialization:decompress] Invalid compressed "
                             "data: incorrect ZLIB header");
  }

  uLongf uncompressedSize = data.size() * 4;
  if (uncompressedSize < 1024)
    uncompressedSize = 1024;

  std::vector<uint8_t> result(uncompressedSize);
  int ret;
  const size_t MAX_UNCOMPRESSED_SIZE = 100 * 1024 * 1024;

  while ((ret = uncompress(result.data(), &uncompressedSize, data.data(),
                           data.size())) == Z_BUF_ERROR) {
    if (uncompressedSize > MAX_UNCOMPRESSED_SIZE) {
      throw std::runtime_error("[serialization:decompress] Uncompressed size "
                               "exceeds maximum allowed");
    }
    uncompressedSize *= 2;
    result.resize(uncompressedSize);
  }

  if (ret != Z_OK) {
    throw std::runtime_error(
        "[serialization:decompress] ZLIB decompression failed: " +
        std::to_string(ret));
  }

  result.resize(uncompressedSize);
  return result;
}

static void parseAndValidateHeader(const std::vector<uint8_t> &data,
                                   uint32_t *out_version = nullptr,
                                   uint32_t *out_node_count = nullptr,
                                   uint32_t *out_edge_count = nullptr,
                                   uint32_t *out_data_size = nullptr) {
  std::vector<uint8_t> decompressedData;
  const uint8_t *rawData = data.data();
  size_t rawSize = data.size();

  CompressionType compType = detectCompressionType(data);
  if (compType != CompressionType::NONE) {
    decompressedData = decompress(data);
    rawData = decompressedData.data();
    rawSize = decompressedData.size();
  }

  if (rawSize < BinaryHeader::SIZE) {
    throw std::runtime_error("[serialization:parseAndValidateHeader] Invalid "
                             "binary data: too small");
  }

  size_t offset = 0;
  uint32_t magic = readU32(rawData, rawSize, offset);
  uint32_t version = readU32(rawData, rawSize, offset);
  uint32_t flags = readU32(rawData, rawSize, offset);
  uint32_t node_count = readU32(rawData, rawSize, offset);
  uint32_t edge_count = readU32(rawData, rawSize, offset);
  uint32_t data_size = readU32(rawData, rawSize, offset);

  if (magic != MAGIC) {
    throw std::runtime_error("[serialization:parseAndValidateHeader] Invalid "
                             "binary data: Incorrect magic bytes");
  }

  if (version != SERIALIZATION_VERSION) {
    throw std::runtime_error(
        "[serialization:parseAndValidateHeader] Version mismatch: expected " +
        std::to_string(SERIALIZATION_VERSION) + ", got " +
        std::to_string(version));
  }

  if (offset + data_size > rawSize) {
    throw std::runtime_error("serialization:parseAndValidateHeader] Invalid "
                             "binary data: data_size exceeds buffer");
  }

  if (out_version)
    *out_version = version;
  if (out_node_count)
    *out_node_count = node_count;
  if (out_edge_count)
    *out_edge_count = edge_count;
  if (out_data_size)
    *out_data_size = data_size;
}

uint32_t getVersion(const std::vector<uint8_t> &data) {
  uint32_t version = 0;
  parseAndValidateHeader(data, &version);
  return version;
}

bool isValidFormat(const std::vector<uint8_t> &data) {
  try {
    parseAndValidateHeader(data);
    return true;
  } catch (const std::exception &) {
    return false;
  }
}

} // namespace memory_graph::utils
