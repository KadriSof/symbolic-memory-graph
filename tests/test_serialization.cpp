#include "memory_graph/edge.hpp"
#include "memory_graph/memory_graph.hpp"
#include "memory_graph/node.hpp"
#include "memory_graph/utils/serialization.hpp"
#include "test_utils.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

using namespace test_utils;
using namespace memory_graph;
using namespace memory_graph::utils;
using json = nlohmann::json;

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

TEST(SerializationTest, SerializationRoundTrip) {
  MemoryGraph original = createTestGraph();

  std::vector<uint8_t> binary = toBinary(original);
  MemoryGraph deserialized = fromBinary(binary);
  std::vector<uint8_t> rebinary = toBinary(deserialized);

  EXPECT_EQ(original.getNodes().size(), deserialized.getNodes().size());
  EXPECT_EQ(original.getEdges().size(), deserialized.getEdges().size());

  // Verify symmetric connections preserved
  const auto &originalConn = original.getNode("luffy").getConnections();
  const auto &deserializedConn = deserialized.getNode("luffy").getConnections();
  EXPECT_EQ(originalConn.size(), deserializedConn.size());

  // Luffy should have connections to Zoro and Nami (symmetric crew)
  EXPECT_TRUE(std::find(originalConn.begin(), originalConn.end(), "zoro") !=
              originalConn.end());
  EXPECT_TRUE(std::find(originalConn.begin(), originalConn.end(), "nami") !=
              originalConn.end());

  // Zoro should have connection to Luffy
  const auto &zoroConn = deserialized.getNode("zoro").getConnections();
  EXPECT_TRUE(std::find(zoroConn.begin(), zoroConn.end(), "luffy") !=
              zoroConn.end());
}

TEST(SerializationTest, SymmetricConnectionsPreserved) {
  MemoryGraph original = createTestGraph();

  // Use sets to handle potential duplicates
  const auto &luffyConn = original.getNode("luffy").getConnections();
  EXPECT_TRUE(std::find(luffyConn.begin(), luffyConn.end(), "zoro") !=
              luffyConn.end());
  EXPECT_TRUE(std::find(luffyConn.begin(), luffyConn.end(), "nami") !=
              luffyConn.end());

  EXPECT_TRUE(std::find(luffyConn.begin(), luffyConn.end(), "shanks") !=
              luffyConn.end());
  EXPECT_TRUE(std::find(luffyConn.begin(), luffyConn.end(), "straw_hats") !=
              luffyConn.end());

  const auto &zoroConn = original.getNode("zoro").getConnections();
  EXPECT_TRUE(std::find(luffyConn.begin(), luffyConn.end(), "shanks") !=
              luffyConn.end());
  EXPECT_TRUE(std::find(luffyConn.begin(), luffyConn.end(), "straw_hats") !=
              luffyConn.end());

  const auto &namiConn = original.getNode("nami").getConnections();
  EXPECT_TRUE(std::find(namiConn.begin(), namiConn.end(), "luffy") !=
              namiConn.end());
  EXPECT_TRUE(std::find(namiConn.begin(), namiConn.end(), "zoro") !=
              namiConn.end());

  // Serialize and deserialize
  std::vector<uint8_t> binary = toBinary(original);
  MemoryGraph deserialized = fromBinary(binary);

  // Verify symmetric connections preserved in deserialized
  const auto &desLuffyConn = deserialized.getNode("luffy").getConnections();
  EXPECT_TRUE(std::find(desLuffyConn.begin(), desLuffyConn.end(), "zoro") !=
              desLuffyConn.end());
  EXPECT_TRUE(std::find(desLuffyConn.begin(), desLuffyConn.end(), "nami") !=
              desLuffyConn.end());
}

TEST(SerializationTest, AsymmetricConnectionsPreserved) {
  MemoryGraph original = createTestGraph();

  std::vector<uint8_t> binary = toBinary(original);
  MemoryGraph deserialized = fromBinary(binary);

  // Verify asymmetric connections preserved
  const auto &luffyConn = deserialized.getNode("luffy").getConnections();
  EXPECT_TRUE(std::find(luffyConn.begin(), luffyConn.end(), "shanks") !=
              luffyConn.end());

  EXPECT_TRUE(std::find(luffyConn.begin(), luffyConn.end(), "straw_hats") !=
              luffyConn.end());

  const auto &zoroConn = deserialized.getNode("zoro").getConnections();
  EXPECT_TRUE(std::find(zoroConn.begin(), zoroConn.end(), "luffy") !=
              zoroConn.end());

  const auto &shanksConn = deserialized.getNode("shanks").getConnections();
  EXPECT_TRUE(shanksConn.empty());
}

TEST(SerializationTest, ComputeDeltaSymmetricAddedNodes) {
  MemoryGraph before = createTestGraph();
  MemoryGraph after = createTestGraph();

  // Add a new node to the symmetric crew
  Node usopp("usopp", "Usopp",
             json{{"type", "character"}, {"bounty", 500000000}});
  after.addNode(usopp);

  // Add new pairwise edges for usopp
  SymmetricConnections crew1{"luffy", "usopp"};
  Edge edge1("luffy_usopp", "crew_member", EdgeType::SYMMETRIC, crew1, 1.0f);
  after.addEdge(edge1);

  SymmetricConnections crew2{"zoro", "usopp"};
  Edge edge2("zoro_usopp", "crew_member", EdgeType::SYMMETRIC, crew2, 1.0f);
  after.addEdge(edge2);

  SymmetricConnections crew3{"nami", "usopp"};
  Edge edge3("nami_usopp", "crew_member", EdgeType::SYMMETRIC, crew3, 1.0f);
  after.addEdge(edge3);

  nlohmann::json delta = computeDelta(before, after);

  EXPECT_TRUE(delta.contains("added_nodes"));
  EXPECT_EQ(delta["added_nodes"].size(), 1);
  EXPECT_TRUE(delta["added_nodes"].contains("usopp"));        // Check by key
  EXPECT_EQ(delta["added_nodes"]["usopp"]["label"], "Usopp"); // Verify data

  EXPECT_TRUE(delta.contains("added_edges"));
  EXPECT_EQ(delta["added_edges"].size(), 3);
  EXPECT_TRUE(delta["added_edges"].contains("luffy_usopp"));
  EXPECT_TRUE(delta["added_edges"].contains("zoro_usopp"));
  EXPECT_TRUE(delta["added_edges"].contains("nami_usopp"));
}

TEST(SerializationTest, ApplyDeltaWithSymmetricConnections) {
  MemoryGraph base = createTestGraph();
  MemoryGraph target = createModifiedGraph();

  nlohmann::json delta = computeDelta(base, target);
  applyDelta(base, delta);

  // Check nodes
  EXPECT_FALSE(base.hasNode("nami"));
  EXPECT_FALSE(base.hasNode("shanks"));
  EXPECT_TRUE(base.hasNode("usopp"));
  EXPECT_TRUE(base.hasNode("sanji"));

  // Check luffy's connections using set
  const auto &luffyConn = base.getNode("luffy").getConnections();
  std::unordered_set<std::string> luffySet(luffyConn.begin(), luffyConn.end());

  EXPECT_TRUE(luffySet.count("zoro") > 0);
  EXPECT_TRUE(luffySet.count("usopp") > 0);
  EXPECT_TRUE(luffySet.count("sanji") > 0);
  EXPECT_TRUE(luffySet.count("straw_hats") > 0);
  EXPECT_FALSE(luffySet.count("nami") > 0);
  EXPECT_FALSE(luffySet.count("shanks") > 0);

  // Check usopp's connections using set
  const auto &usoppConn = base.getNode("usopp").getConnections();
  std::unordered_set<std::string> usoppSet(usoppConn.begin(), usoppConn.end());
  EXPECT_TRUE(usoppSet.count("luffy") > 0);
  EXPECT_TRUE(usoppSet.count("zoro") > 0);
  EXPECT_TRUE(usoppSet.count("sanji") > 0);
}

TEST(SerializationTest, EdgeTypePreserved) {
  MemoryGraph graph;

  // Add nodes
  Node alice("alice", "Alice");
  Node bob("bob", "Bob");
  Node charlie("charlie", "Charlie");
  graph.addNode(alice);
  graph.addNode(bob);
  graph.addNode(charlie);

  // Add SYMMETRIC edge
  SymmetricConnections symConn{"alice", "bob"};
  Edge symEdge("friends", "friends", EdgeType::SYMMETRIC, symConn, 0.8f);
  graph.addEdge(symEdge);

  // Add ASYMMETRIC edge
  AsymmetricConnections asymConn{"alice", "charlie"};
  Edge asymEdge("follows", "follows", EdgeType::ASYMMETRIC, asymConn, 1.0f);
  graph.addEdge(asymEdge);

  // Serialize and deserialize
  std::vector<uint8_t> binary = toBinary(graph);
  MemoryGraph deserialized = fromBinary(binary);

  // Verify edge types preserved
  const Edge &symDes = deserialized.getEdge("friends");
  EXPECT_EQ(symDes.getType(), EdgeType::SYMMETRIC);

  const Edge &asymDes = deserialized.getEdge("follows");
  EXPECT_EQ(asymDes.getType(), EdgeType::ASYMMETRIC);

  // Verify symmetric connections are bidirectional
  const auto &aliceConn = deserialized.getNode("alice").getConnections();
  EXPECT_TRUE(std::find(aliceConn.begin(), aliceConn.end(), "bob") !=
              aliceConn.end());

  const auto &bobConn = deserialized.getNode("bob").getConnections();
  EXPECT_TRUE(std::find(bobConn.begin(), bobConn.end(), "alice") !=
              bobConn.end());

  // Verify asymmetric connections are directed
  const auto &charlieConn = deserialized.getNode("charlie").getConnections();
  EXPECT_TRUE(charlieConn.empty()); // Charlie doesn't follow Alice back
}

TEST(SerializationTest, SerializationOptionsExcludeNodes) {
  MemoryGraph original = createTestGraph();
  SerializationOptions opts;
  opts.include_nodes = false;

  std::vector<uint8_t> binary = toBinary(original, opts);
  MemoryGraph deserialized = fromBinary(binary);

  // Node should be exclude, mate!
  EXPECT_EQ(deserialized.getNodes().size(), 0);
  // Edges and metadata should still be intact?
  EXPECT_GT(deserialized.getEdges().size(), 0);
  EXPECT_FALSE(deserialized.getMetadata().empty());
}

TEST(SerializationTest, SerializationOptionsExcludeEdges) {
  MemoryGraph original = createTestGraph();
  SerializationOptions opts;
  opts.include_edges = false;

  std::vector<uint8_t> binary = toBinary(original, opts);
  MemoryGraph deserialized = fromBinary(binary);

  // Edges should be exculded, cuhh!
  EXPECT_EQ(deserialized.getEdges().size(), 0);
  // Nodes and metadata should still be intact or I would spit in your general
  // direction!
  EXPECT_GT(deserialized.getNodes().size(), 0);
  EXPECT_FALSE(deserialized.getMetadata().empty());
}

TEST(SerializationTest, ApplyDeltaPreservesEdgesOnNodeModification) {
  MemoryGraph base = createTestGraph();

  // 1. Verify Luffy has connections before we modify him
  size_t originalConnectionCount =
      base.getNode("luffy").getConnections().size();
  EXPECT_GT(originalConnectionCount, 0);

  // 2. Create a manual delta that ONLY modifies Luffy's metadata
  nlohmann::json delta;
  nlohmann::json modifiedLuffyJson = base.getNode("luffy").toJson();
  modifiedLuffyJson["metadata"]["bounty"] = 0; // Sorry, Oda Sensei..

  delta["modified_nodes"] = nlohmann::json::object();
  delta["modified_nodes"]["luffy"] = modifiedLuffyJson;

  // 3. Apply the delta
  applyDelta(base, delta);

  // 4. Check luffy's connections (should StILL be there!)
  size_t newConnectionCount = base.getNode("luffy").getConnections().size();
  EXPECT_EQ(originalConnectionCount, newConnectionCount);
}

TEST(SerializationTest, ComputeDeltaCorrectlyIdentifiesRemoveEdges) {
  MemoryGraph before = createTestGraph();
  MemoryGraph after = createTestGraph();

  // Get an existing edge ID to remove
  std::string edgeToRemove = before.getEdges().front().getId();

  // Remove it from the 'after' graph
  after.removeEdge(edgeToRemove);

  // Compute delta
  nlohmann::json delta = computeDelta(before, after);

  // The delta should contain exactly ONE removed edge
  EXPECT_TRUE(delta.contains("removed_edges"));
  EXPECT_EQ(delta["removed_edges"].size(), 1);
  EXPECT_EQ(delta["removed_edges"][0], edgeToRemove);
}

TEST(SerializationTest, ApplyDeltaBinaryRoundTrip) {
  MemoryGraph base = createTestGraph();
  MemoryGraph target = createModifiedGraph();

  // Create a standard JSON delta
  nlohmann::json deltaJson = computeDelta(base, target);
  std::string deltaStr = deltaJson.dump();
  std::vector<uint8_t> deltaData(deltaStr.begin(), deltaStr.end());

  // Test 1: Uncompressed binary delta
  MemoryGraph baseCopy1 = createTestGraph();
  applyDeltaBinary(baseCopy1, deltaData);
  EXPECT_EQ(baseCopy1.getNodes().size(), target.getNodes().size());
  EXPECT_EQ(baseCopy1.getEdges().size(), target.getEdges().size());

  // Test 2: Compressed (ZLIB) binary delta
  std::vector<uint8_t> compressedDelta =
      compress(deltaData, CompressionType::ZLIB);
  MemoryGraph baseCopy2 = createTestGraph();
  applyDeltaBinary(baseCopy2, compressedDelta);
  EXPECT_EQ(baseCopy2.getNodes().size(), target.getNodes().size());
  EXPECT_EQ(baseCopy2.getEdges().size(), target.getEdges().size());
}

TEST(SerializationTest, FromBinaryInvalidMagicBytes) {
  // Create 24 bytes of garbage data (enough to pass the size check, but fail
  // magic check)
  std::vector<uint8_t> garbage(24, 0x00);
  EXPECT_THROW(fromBinary(garbage), std::runtime_error);
}

TEST(SerializationTest, FromBinaryTruncatedData) {
  MemoryGraph original = createTestGraph();
  std::vector<uint8_t> binary = toBinary(original);

  // Truncate it to just 10 bytes (less than 24-bytes header size)
  std::vector<uint8_t> truncated(binary.begin(), binary.begin() + 10);
  EXPECT_THROW(fromBinary(truncated), std::runtime_error);
}

TEST(SerializationTest, EmptyGraphRoundTrip) {
  MemoryGraph emptyGraph;

  // Test uncompressed
  std::vector<uint8_t> binary = toBinary(emptyGraph);
  MemoryGraph deserialized = fromBinary(binary);
  EXPECT_EQ(deserialized.getNodes().size(), 0);
  EXPECT_EQ(deserialized.getEdges().size(), 0);

  // Test compressed
  SerializationOptions opts;
  opts.compression = CompressionType::ZLIB;
  std::vector<uint8_t> compressedBinary = toBinary(emptyGraph, opts);
  MemoryGraph deserializedCompressed = fromBinary(compressedBinary);
  EXPECT_EQ(deserializedCompressed.getNodes().size(), 0);
  EXPECT_EQ(deserializedCompressed.getEdges().size(), 0);
}

TEST(SerializationTest, HeaderValidationHelpers) {
  MemoryGraph original = createTestGraph();
  std::vector<uint8_t> binary = toBinary(original);

  // Valid format checks
  EXPECT_TRUE(isValidFormat(binary));
  EXPECT_EQ(getVersion(binary), SERIALIZATION_VERSION);

  // Corrupt the magic bytes (oh, no! not the magic bytes!!)
  std::vector<uint8_t> corrupted = binary;
  corrupted[0] = 'X';
  EXPECT_FALSE(isValidFormat(corrupted));
  EXPECT_THROW(getVersion(corrupted), std::runtime_error);
}
