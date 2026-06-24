#include "memory_graph/edge.hpp"
#include "memory_graph/memory_graph.hpp"
#include "memory_graph/node.hpp"
#include "memory_graph/utils/serialization.hpp"
#include "test_utils.hpp"
#include <cstdint>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <vector>

using namespace test_utils;
using namespace memory_graph;
using namespace memory_graph::utils;
using json = nlohmann::json;

// Helper Functions
/**
 * @brief Create a modified version of the test graph
 */
MemoryGraph createModifiedGraph() {
  MemoryGraph graph(json{{"name", "Modified Graph Test"}, {"version", "2.0"}});

  // Add some same nodes
  Node luffy("luffy", "Monkey D. Luffy",
             json{{"type", "character"}, {"bounty", 3000000000}});
  Node zoro("zoro", "Roronoa Zoro",
            json{{"type", "character"}, {"bounty", 1111000000}});
  Node usopp("usopp", "Usopp",
             json{{"type", "character"}, {"bounty", 500000000}});
  Node sanji("sanji", "Sanji",
             json{{"type", "character"}, {"bounty", 1032000000}});

  graph.addNode(luffy);
  graph.addNode(zoro);
  graph.addNode(usopp);
  graph.addNode(sanji);

  // SYMMETRIC: Crew relationship (all crew members)
  SymmetricConnections crewConnections{"luffy", "zoro", "usopp", "sanji"};
  Edge crewEdge("straw_hat_crew", "crew_members", EdgeType::SYMMETRIC,
                crewConnections, 1.0f);
  graph.addEdge(crewEdge);

  SymmetricConnections crew1{"luffy", "zoro"};
  Edge edge1("luffy_zoro", "crew_member", EdgeType::SYMMETRIC, crew1, 1.0f);
  graph.addEdge(edge1);

  SymmetricConnections crew2{"luffy", "nami"};
  Edge edge2("luffy_nami", "crew_member", EdgeType::SYMMETRIC, crew2, 1.0f);
  graph.addEdge(edge2);

  SymmetricConnections crew3{"zoro", "usopp"};
  Edge edge3("zoro_usopp", "crew_member", EdgeType::SYMMETRIC, crew3, 1.0f);
  graph.addEdge(edge1);

  SymmetricConnections crew4{"zoro", "nami"};
  Edge edge4("zoro_nami", "crew_member", EdgeType::SYMMETRIC, crew3, 1.0f);
  graph.addEdge(edge1);

  // ASYMMETRIC: Captain relationship
  AsymmetricConnections captainConn{"luffy", "straw_hats"};
  Edge captain("luffy_captain", "is_captain_of", EdgeType::ASYMMETRIC,
               captainConn, 1.0f);
  graph.addEdge(captain);

  AsymmetricConnections followsConn{"zoro", "luffy"};
  Edge follows("zoro_follows", "follows", EdgeType::ASYMMETRIC, followsConn,
               1.0f);
  graph.addEdge(follows);

  return graph;
}

// Binary Serialization Tests
TEST(SerializationTest, ToBinaryBasic) {
  MemoryGraph original = createTestGraph();

  std::vector<uint8_t> binary = toBinary(original);

  EXPECT_FALSE(binary.empty());
  EXPECT_GT(binary.size(), 24); // at least header size

  // Check magic bytes
  EXPECT_EQ(binary[0], 'M');
  EXPECT_EQ(binary[1], 'E');
  EXPECT_EQ(binary[2], 'M');
  EXPECT_EQ(binary[3], 'G');
}

TEST(SerializationTest, CompressionRoundTrip) {
  MemoryGraph original = createTestGraph();

  // Enable compression
  SerializationOptions options;
  options.compression = CompressionType::ZLIB;
  std::vector<uint8_t> compressed = toBinary(original, options);

  MemoryGraph deserialized = fromBinary(compressed);

  EXPECT_EQ(original.getNodes().size(), deserialized.getNodes().size());
  EXPECT_EQ(original.getEdges().size(), deserialized.getEdges().size());
}

TEST(SerializationTest, AutoDetectCompression) {
  MemoryGraph original = createTestGraph();

  // Since we changed the compression options default to NONE
  std::vector<uint8_t> uncompressed = toBinary(original);
  EXPECT_FALSE(isCompressed(uncompressed));

  SerializationOptions options;
  options.compression = CompressionType::ZLIB;
  std::vector<uint8_t> compressed = toBinary(original, options);
  EXPECT_TRUE(isCompressed(compressed));

  MemoryGraph fromUncompressed = fromBinary(uncompressed);
  MemoryGraph fromCompressed = fromBinary(compressed);

  EXPECT_EQ(fromUncompressed.getNodes().size(),
            fromCompressed.getNodes().size());
}

TEST(SerializationTest, FromBinaryBasic) {
  MemoryGraph original = createTestGraph();

  std::vector<uint8_t> binary = toBinary(original);
  MemoryGraph deserialized = fromBinary(binary);

  // Compare metadata
  EXPECT_EQ(original.getMetadata()["name"], deserialized.getMetadata()["name"]);
  EXPECT_EQ(original.getMetadata()["version"],
            deserialized.getMetadata()["version"]);

  // Compare nodes
  EXPECT_EQ(original.getNodes().size(), deserialized.getNodes().size());
  for (const auto &node : original.getNodes()) {
    EXPECT_TRUE(deserialized.hasNode(node.getId()));
    const Node &desNode = deserialized.getNode(node.getId());
    EXPECT_EQ(node.getMetadata(), desNode.getMetadata());
    EXPECT_EQ(node.getConnections().size(), desNode.getConnections().size());
  }

  // Compare edges
  EXPECT_EQ(original.getEdges().size(), deserialized.getEdges().size());
  for (const auto &edge : original.getEdges()) {
    EXPECT_TRUE(deserialized.hasEdge(edge.getId()));
    const Edge &desEdge = deserialized.getEdge(edge.getId());
    EXPECT_EQ(edge.getLabel(), desEdge.getLabel());
    EXPECT_EQ(edge.getType(), desEdge.getType());
    EXPECT_NEAR(edge.getWeight(), desEdge.getWeight(), 1e-6);
  }
}
