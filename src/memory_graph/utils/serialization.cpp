#include "memory_graph/utils/serialization.hpp"
#include "memory_graph/memory_graph.hpp"
#include "nlohmann/json.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
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

MemoryGraph fromBinary(const std::vector<uint8_t> &data) {
  // 1. Decompress if needed
  std::vector<uint8_t> decompressedData;
  const uint8_t *rawData = data.data();
  size_t rawSize = data.size();

  // Check if data is compressed
  bool isCompressed =
      (rawSize > 2 && rawData[0] == 0x78 &&
       (rawData[1] == 0X01 || rawData[1] == 0x5E || rawData[1] == 0x9C));

  if (isCompressed) {
    decompressedData = decompress(data);
    rawData = decompressedData.data();
    rawSize = decompressedData.size();
  }

  // 2. Parse header
  if (rawSize < BinaryHeader::SIZE) {
    throw std::runtime_error(
        "[serialization:fromBinary] Invalid binary data: too small");
  }

  auto readU32 = [&rawData, &rawSize](size_t &offset) -> uint32_t {
    if (offset + 4 > rawSize) {
      throw std::runtime_error("Invalid binary data: unexpected EOF");
    }
    uint32_t value = rawData[offset] |
                     (rawData[offset + 1] << 8 | (rawData[offset + 2] << 16 |
                                                  (rawData[offset + 3] << 24)));
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

} // namespace memory_graph::utils
