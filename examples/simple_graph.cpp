#include "memory_graph/edge.hpp"
#include "memory_graph/memory_graph.hpp"
#include "memory_graph/node.hpp"
#include "memory_graph/utils/serialization.hpp"
#include "nlohmann/json.hpp"
#include <cstdint>
#include <exception>
#include <iostream>
#include <vector>

using namespace memory_graph;
using namespace memory_graph::utils;
using json = nlohmann::json;

int main() {
  // Create a memory graph
  MemoryGraph graph(
      json{{"name", "One Piece Memory Graph"}, {"version", "1.0"}});

  // Add nodes
  Node luffy("luffy", "Monkey D. Luffy",
             json{{"type", "Character"}, {"bounty", 3000000000}});
  Node shanks("shanks", "Shanks",
              json{{"type", "Character"}, {"bounty", 4048900000}});
  Node strawHats("straw_hats", "Straw Hat Pirates", json{{"type", "Crew"}});

  graph.addNode(luffy);
  graph.addNode(shanks);
  graph.addNode(strawHats);

  // Add edges
  AsymmetricConnections luffyShanksConn = {"luffy", "shanks"};
  Edge inspirationEdge("luffy_shanks_inspiration", "inspired by",
                       EdgeType::ASYMMETRIC, luffyShanksConn, 0.95f,
                       json{{"since", "Age 7"}, {"location", "East Blue"}});
  graph.addEdge(inspirationEdge);

  AsymmetricConnections luffyStrawHatsConn = {"luffy", "straw_hats"};
  Edge captainOf("luffy_captai_straw_hats", "is captain of",
                 EdgeType::ASYMMETRIC, luffyStrawHatsConn, 1.0f,
                 json{{"since", "151st year of the Greate Pirate Era"}});
  graph.addEdge(captainOf);

  // Query the graph
  std::cout << "=== luffy's Neighbors ===" << std::endl;
  for (const auto &neighbor : graph.getNeighbors("luffy")) {
    std::cout << "- " << neighbor.getLabel() << std::endl;
  }

  // Test query with depth
  std::cout << "\n === Query Result (Depth 1) ===" << std::endl;
  auto queryResult = graph.query("luffy", 1);
  std::cout << queryResult.dump(2) << std::endl;

  // Serialize to JSON
  std::cout << "\n Graph JSON ===" << std::endl;
  json graphJson = graph.toJson();
  std::cout << graphJson.dump(2) << std::endl;

  // Deserialize from JSON
  MemoryGraph newGraph = MemoryGraph::fromJson(graphJson);
  std::cout << "\n === Deserailization Graph Node ===" << std::endl;
  for (const auto &node : newGraph.getNodes()) {
    std::cout << "- " << node.getId() << ": " << node.getLabel() << std::endl;
  }

  std::cout << "\n=== Binary Serialization & Compression Test ===" << std::endl;

  // 1. Create Uncompressed Binary
  memory_graph::utils::SerializationOptions uncompressedOpts;
  uncompressedOpts.compression = CompressionType::NONE;
  std::vector<uint8_t> uncompressedBinary = toBinary(graph, uncompressedOpts);

  // 2. Create Compressed Binary (ZLIB)
  memory_graph::utils::SerializationOptions compressedOpts;
  compressedOpts.compression = CompressionType::ZLIB;
  std::vector<uint8_t> compressedBinary = toBinary(graph, compressedOpts);

  // 3. Compare Sizes
  std::cout << "Uncompressed Binary Size: " << uncompressedBinary.size()
            << " bytes" << std::endl;
  std::cout << "Compressed Binary Size:   " << compressedBinary.size()
            << " bytes" << std::endl;

  // 4. Verify Compression Headers using our helper function
  std::cout << "Is Uncompressed data marked as compressed? "
            << (isCompressed(uncompressedBinary) ? "Yes" : "No") << std::endl;
  std::cout << "Is Compressed data marked as compressed?   "
            << (isCompressed(compressedBinary) ? "Yes" : "No") << std::endl;

  // 5. Round-Trip Test: Decompress the ZLIB data and verify it matches the
  // original
  try {
    MemoryGraph decompressedGraph = fromBinary(compressedBinary);
    std::cout << "\n[SUCCESS] Round-trip decompression worked!" << std::endl;
    std::cout << " -> Nodes: " << decompressedGraph.getNodes().size()
              << " (Original: " << graph.getNodes().size() << ")" << std::endl;
    std::cout << " -> Edges: " << decompressedGraph.getEdges().size()
              << " (Original: " << graph.getEdges().size() << ")" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "\n[FAILED] Decompression threw an error: " << e.what()
              << std::endl;
  }

  return 0;
}
