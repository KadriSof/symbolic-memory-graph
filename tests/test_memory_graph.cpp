#include "memory_graph/edge.hpp"
#include "memory_graph/exceptions.hpp"
#include "memory_graph/memory_graph.hpp"
#include "memory_graph/node.hpp"
#include "test_utils.hpp"
#include <algorithm>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_set>
#include <vector>

using namespace test_utils;
using namespace memory_graph;
using json = nlohmann::json;

// Constructor Tests
TEST(MemoryGraphTest, ConstructorWithMetadata) {
  json metadata = {{"name", "One Piece Graph"}, {"author", "Edward Newgate"}};
  MemoryGraph graph(metadata);

  EXPECT_EQ(graph.getMetadata()["name"], "One Piece Graph");
  EXPECT_EQ(graph.getMetadata()["author"], "Edward Newgate");
  EXPECT_TRUE(graph.getNodes().empty());
  EXPECT_TRUE(graph.getEdges().empty());
}

TEST(MemoryGraphTest, ConstructorEmptyMetadata) {
  MemoryGraph graph;
  EXPECT_TRUE(graph.getMetadata().is_object());
  EXPECT_TRUE(graph.getMetadata().empty());
}

// Node Management Tests
TEST(MemoryGraphTest, AddNode) {
  MemoryGraph graph;
  Node node("luffy", "Monkey D. Luffy");

  EXPECT_NO_THROW(graph.addNode(node));
  EXPECT_TRUE(graph.hasNode("luffy"));
  EXPECT_EQ(graph.getNodes().size(), 1);
}

TEST(MemoryGraphTest, AddDuplicateNodeThrows) {
  MemoryGraph graph;
  Node node1("luffy", "Monkey D. Luffy");
  Node node2("luffy", "Demalo Black");

  graph.addNode(node1);
  EXPECT_THROW(graph.addNode(node2), DuplicateIdError);
  EXPECT_EQ(graph.getNodes().size(), 1);
}

TEST(MemoryGraphTest, GetNode) {
  MemoryGraph graph;
  Node node("luffy", "Monkey D. Luffy", json{{"bounty", 3000000000}});
  graph.addNode(node);

  const Node &retrieved = graph.getNode("luffy");
  EXPECT_EQ(retrieved.getId(), "luffy");
  EXPECT_EQ(retrieved.getLabel(), "Monkey D. Luffy");
  EXPECT_EQ(retrieved.getMetadata()["bounty"], 3000000000);
}

TEST(MemoryGraphTest, RemoveNode) {
  MemoryGraph graph = createTestGraph();

  EXPECT_TRUE(graph.hasNode("zoro"));
  EXPECT_NO_THROW(graph.removeNode("zoro"));
  EXPECT_FALSE(graph.hasNode("zoro"));
}

TEST(MemoryGraphTest, RemoveNodeWithEdges) {
  MemoryGraph graph = createTestGraph();

  // Luffy has connections to zoro, nami, shanks, straw_hats
  EXPECT_EQ(graph.getNeighbors("luffy").size(), 4);

  // Remove nami (sorry Nami :p)
  graph.removeNode("nami");

  // Luffy should now have 3 connections (zoro, shanks, straw_hats)
  EXPECT_EQ(graph.getNeighbors("luffy").size(), 3);

  // Edge to nami should be gone
  EXPECT_FALSE(graph.hasEdge("luffy_nami"));
}

TEST(MemoryGraphTest, RemoveNonExistentNodeThrows) {
  MemoryGraph graph;
  EXPECT_THROW(graph.removeNode("nonexistent"), NodeNotFoundError);
}

TEST(MemoryGraphTest, GetNodes) {
  MemoryGraph graph = createTestGraph();
  std::vector<Node> nodes = graph.getNodes();

  EXPECT_EQ(nodes.size(), 5);

  // Verify all nodes are present
  std::unordered_set<std::string> nodeIds;
  for (const auto &node : nodes) {
    nodeIds.insert(node.getId());
  }

  EXPECT_TRUE(nodeIds.count("luffy"));
  EXPECT_TRUE(nodeIds.count("zoro"));
  EXPECT_TRUE(nodeIds.count("nami"));
  EXPECT_TRUE(nodeIds.count("shanks"));
  EXPECT_TRUE(nodeIds.count("straw_hats"));
}

// Edge Management Tests
TEST(MemoryGraphTest, AddAsymmetricEdge) {
  MemoryGraph graph;
  Node luffy("luffy", "Luffy");
  Node shanks("shanks", "Shanks");
  graph.addNode(luffy);
  graph.addNode(shanks);

  AsymmetricConnections conn{"luffy", "shanks"};
  Edge edge("e1", "inspired_by", EdgeType::ASYMMETRIC, conn, 0.95f);

  EXPECT_NO_THROW(graph.addEdge(edge));
  EXPECT_TRUE(graph.hasEdge("e1"));
  EXPECT_EQ(graph.getEdges().size(), 1);

  // Check that connections was added to nodes
  const auto &luffyNode = graph.getNode("luffy");
  const auto &connections = luffyNode.getConnections();
  EXPECT_EQ(connections.size(), 1);
  EXPECT_EQ(connections[0], "shanks");
}

TEST(MemoryGraphTest, AddSymmetricEdge) {
  MemoryGraph graph;
  Node luffy("luffy", "Luffy");
  Node zoro("zoro", "Zoro");
  graph.addNode(luffy);
  graph.addNode(zoro);

  std::unordered_set<std::string> nodes = {"luffy", "zoro"};
  SymmetricConnections conn(nodes);

  Edge edge("e1", "friends", EdgeType::SYMMETRIC, conn, 1.0f);

  EXPECT_NO_THROW(graph.addEdge(edge));
  EXPECT_TRUE(graph.hasEdge("e1"));

  // Check bidirectional connections
  const auto &luffyConnections = graph.getNode("luffy").getConnections();
  const auto &zoroConnections = graph.getNode("zoro").getConnections();
  EXPECT_EQ(luffyConnections.size(), 1);
  EXPECT_EQ(zoroConnections.size(), 1);
  EXPECT_EQ(luffyConnections[0], "zoro");
  EXPECT_EQ(zoroConnections[0], "luffy");
}

TEST(MemoryGraphTest, AddDuplicateEdgeThrows) {
  MemoryGraph graph = createTestGraph();

  SymmetricConnections conn{"luffy", "zoro"};
  Edge duplicate("luffy_zoro", "duplicate", EdgeType::SYMMETRIC, conn);

  EXPECT_THROW(graph.addEdge(duplicate), DuplicateIdError);
}

TEST(MemoryGraphTest, AddEdgeWithNonExistentNodeThrows) {
  MemoryGraph graph;
  Node luffy("luffy", "luffy");
  graph.addNode(luffy);

  AsymmetricConnections conn{"luffy", "imu"};
  Edge edge("e1", "bad", EdgeType::SYMMETRIC, conn);

  EXPECT_THROW(graph.addEdge(edge), InvalidConnectionError);
}

TEST(MemoryGraphTest, AddSymmetricEdgeWithWrongNodeCountThrows) {
  MemoryGraph graph;

  Node luffy("luffy", "Luffy");
  Node zoro("zoro", "Zoro");
  Node nami("nami", "Nami");
  graph.addNode(luffy);
  graph.addNode(zoro);
  graph.addNode(nami);

  SymmetricConnections conn({"luffy", "zoro", "nami"});
  Edge edge("e1", "group", EdgeType::SYMMETRIC, conn);

  EXPECT_THROW(graph.addEdge(edge), InvalidConnectionError);
}

TEST(MemoryGraphTest, RemoveEdge) {
  MemoryGraph graph = createTestGraph();

  EXPECT_TRUE(graph.hasEdge("luffy_zoro"));
  graph.removeEdge("luffy_zoro");
  EXPECT_FALSE(graph.hasEdge("luffy_zoro"));

  // Check connections removed from nodes
  const auto &luffyConnections = graph.getNode("luffy").getConnections();
  EXPECT_FALSE(std::find(luffyConnections.begin(), luffyConnections.end(),
                         "zoro") != luffyConnections.end());
}

TEST(MemoryGraphTest, RemoveNonExistentEdgeThrows) {
  MemoryGraph graph;
  EXPECT_THROW(graph.removeEdge("nonexistent"), EdgeNotFoundError);
}

TEST(MemoryGraphTest, GetEdge) {
  MemoryGraph graph = createTestGraph();

  const Edge &edge = graph.getEdge("luffy_zoro");
  EXPECT_EQ(edge.getId(), "luffy_zoro");
  EXPECT_EQ(edge.getLabel(), "crew_member");
  EXPECT_EQ(edge.getType(), EdgeType::SYMMETRIC);
}

TEST(MemoryGraphTest, GetNonExistentEdgeThrows) {
  MemoryGraph graph;
  EXPECT_THROW(graph.getEdge("nonexistent"), EdgeNotFoundError);
}

TEST(MemoryGraphTest, GetEdges) {
  MemoryGraph graph = createTestGraph();
  std::vector<Edge> edges = graph.getEdges();

  EXPECT_EQ(edges.size(), 5);

  std::unordered_set<std::string> edgeIds;
  for (const auto &edge : edges) {
    edgeIds.insert(edge.getId());
  }

  EXPECT_TRUE(edgeIds.count("luffy_zoro"));
  EXPECT_TRUE(edgeIds.count("luffy_nami"));
  EXPECT_TRUE(edgeIds.count("luffy_shanks"));
  EXPECT_TRUE(edgeIds.count("luffy_captain"));
}

// Neighbors Tests
TEST(MemoryGraphTest, getNeighbors) {
  MemoryGraph graph = createTestGraph();

  std::vector<Node> neighbors = graph.getNeighbors("luffy");
  EXPECT_EQ(neighbors.size(), 4);

  std::unordered_set<std::string> neighborsIds;
  for (const auto &node : neighbors) {
    neighborsIds.insert(node.getId());
  }

  EXPECT_TRUE(neighborsIds.count("zoro"));
  EXPECT_TRUE(neighborsIds.count("nami"));
  EXPECT_TRUE(neighborsIds.count("shanks"));
  EXPECT_TRUE(neighborsIds.count("straw_hats"));
}

TEST(MemoryGraphTest, getNeighborsEmpty) {
  MemoryGraph graph = createTestGraph();

  std::vector<Node> neighbors = graph.getNeighbors("shanks");
  EXPECT_EQ(neighbors.size(), 0);
}

TEST(MemoryGraphTest, getNeighborsNonExistentNodeThrows) {
  MemoryGraph graph;
  EXPECT_THROW(graph.getNeighbors("imu"), NodeNotFoundError);
}

// Query Tests (BFS Traversal)
TEST(MemoryGraphTest, QueryDepthO) {
  MemoryGraph graph = createTestGraph();

  json result = graph.query("luffy", 0);
  EXPECT_TRUE(result.contains("depth_0"));
  EXPECT_EQ(result["depth_0"].size(), 1);
  EXPECT_EQ(result["depth_0"][0], "luffy");
  EXPECT_FALSE(result.contains("depth_1"));
}

TEST(MemoryGraphTest, QueryDepth1) {
  MemoryGraph graph = createTestGraph();

  json result = graph.query("luffy", 1);
  EXPECT_TRUE(result.contains("depth_0"));
  EXPECT_TRUE(result.contains("depth_1"));
  EXPECT_EQ(result["depth_0"][0], "luffy");
  EXPECT_EQ(result["depth_1"].size(), 4);

  // Verify all neighbors are present
  std::unordered_set<std::string> neighbors;
  for (const auto &id : result["depth_1"]) {
    neighbors.insert(id.get<std::string>());
  }

  EXPECT_TRUE(neighbors.count("zoro"));
  EXPECT_TRUE(neighbors.count("nami"));
  EXPECT_TRUE(neighbors.count("shanks"));
  EXPECT_TRUE(neighbors.count("straw_hats"));
}

TEST(MemoryGraphTest, QueryWithMinWeight) {
  MemoryGraph graph = createTestGraph();

  // All edges have weight >= 0.95 except maybe some
  json result = graph.query("luffy", 1, 0.96f);

  // Only edge with weight >= 0.96
  // luffy_shanks has 0.95, so it should be excluded
  EXPECT_EQ(result["depth_1"].size(), 3);

  std::unordered_set<std::string> neighbors;
  for (const auto &id : result["depth_1"]) {
    neighbors.insert(id.get<std::string>());
  }

  EXPECT_TRUE(neighbors.count("zoro"));
  EXPECT_TRUE(neighbors.count("nami"));
  EXPECT_TRUE(neighbors.count("straw_hats"));
  EXPECT_FALSE(neighbors.count("shanks"));
}

TEST(MemoryGraphTest, QueryNonExistentNodeThrows) {
  MemoryGraph graph;
  EXPECT_THROW(graph.query("imu", 1), NodeNotFoundError);
}

// Serialization Tests
TEST(MemoryGraphTest, ToJson) {
  MemoryGraph graph = createTestGraph();
  json graphJson = graph.toJson();

  EXPECT_TRUE(graphJson.contains("metadata"));
  EXPECT_TRUE(graphJson.contains("nodes"));
  EXPECT_TRUE(graphJson.contains("edges"));
  EXPECT_EQ(graphJson["metadata"]["name"], "Test Graph");
  EXPECT_EQ(graphJson["nodes"].size(), 5);
  EXPECT_EQ(graphJson["edges"].size(), 5);

  // Check a specific node
  EXPECT_TRUE(graphJson["nodes"].contains("luffy"));
  EXPECT_EQ(graphJson["nodes"]["luffy"]["label"], "Monkey D. Luffy");
  EXPECT_EQ(graphJson["nodes"]["luffy"]["metadata"]["bounty"], 3000000000);

  // Check a specific edge
  EXPECT_TRUE(graphJson["edges"].contains("luffy_shanks"));
  EXPECT_EQ(graphJson["edges"]["luffy_shanks"]["label"], "inspired_by");
  EXPECT_NEAR(graphJson["edges"]["luffy_shanks"]["weight"], 0.95, 1e-6);
}

TEST(MemoryGraphTest, FromJson) {
  MemoryGraph original = createTestGraph();
  json graphJson = original.toJson();

  MemoryGraph deserialized = MemoryGraph::fromJson(graphJson);

  // Check metadata
  EXPECT_EQ(deserialized.getMetadata()["name"], "Test Graph");

  // Check nodes
  EXPECT_EQ(deserialized.getNodes().size(), 5);
  EXPECT_TRUE(deserialized.hasNode("luffy"));
  EXPECT_TRUE(deserialized.hasNode("zoro"));
  EXPECT_TRUE(deserialized.hasNode("nami"));
  EXPECT_TRUE(deserialized.hasNode("shanks"));
  EXPECT_TRUE(deserialized.hasNode("straw_hats"));

  // Check node data
  const Node &luffy = deserialized.getNode("luffy");
  EXPECT_EQ(luffy.getLabel(), "Monkey D. Luffy");
  EXPECT_EQ(luffy.getMetadata()["bounty"], 3000000000);

  // Check edges
  EXPECT_EQ(deserialized.getEdges().size(), 5);
  EXPECT_TRUE(deserialized.hasEdge("luffy_zoro"));
  EXPECT_TRUE(deserialized.hasEdge("luffy_shanks"));

  // Check connections
  EXPECT_EQ(deserialized.getNeighbors("luffy").size(), 4);
}

TEST(MemoryGraphTest, SerializationRoundTrip) {
  MemoryGraph original = createTestGraph();
  json serialized = original.toJson();
  MemoryGraph deserialized = MemoryGraph::fromJson(serialized);
  json reserialized = deserialized.toJson();

  // Compare JSON objects (they should be identical)
  EXPECT_EQ(serialized["metadata"], reserialized["metadata"]);
  EXPECT_EQ(serialized["nodes"].size(), reserialized["nodes"].size());
  EXPECT_EQ(serialized["edges"].size(), reserialized["edges"].size());
}

// Metadata Tests
TEST(MemoryGraphTest, SetMetadata) {
  MemoryGraph graph;
  json newMetadata = {{"key", "value"}, {"number", 42}};

  graph.setMetadata(newMetadata);
  EXPECT_EQ(graph.getMetadata()["key"], "value");
  EXPECT_EQ(graph.getMetadata()["number"], 42);
}

TEST(MemoryGraphTest, UpdateMetadata) {
  MemoryGraph graph(json{{"name", "Graph"}, {"version", 1}});

  graph.updateMetadata("version", 2);
  graph.updateMetadata("author", "Test User");

  EXPECT_EQ(graph.getMetadata()["name"], "Graph");
  EXPECT_EQ(graph.getMetadata()["version"], 2);
  EXPECT_EQ(graph.getMetadata()["author"], "Test User");
}

// Edge Case Tests
TEST(MemoryGraphTest, EmptyGraph) {
  MemoryGraph graph;

  EXPECT_TRUE(graph.getNodes().empty());
  EXPECT_TRUE(graph.getEdges().empty());

  Node luffy("luffy", "Luffy");
  graph.addNode(luffy);
  EXPECT_EQ(graph.getNeighbors("luffy").size(), 0);
}

TEST(MemoryGraphTest, LargeNumberOfNodes) {
  MemoryGraph graph;
  const int N = 100;

  for (int i = 0; i < N; i++) {
    Node node("node_" + std::to_string(i), "Label_" + std::to_string(i));
    graph.addNode(node);
  }

  EXPECT_EQ(graph.getNodes().size(), N);

  for (int i = 0; i < N; i++) {
    EXPECT_TRUE(graph.hasNode("node_" + std::to_string(i)));
  }
}

TEST(MemoryGraphTest, QueryUsesCorrectEdgeWeights) {
  MemoryGraph graph = createTestGraph();

  // Query with minWeight high enough to filter out low-weight edges
  json result = graph.query("luffy", 1, 0.96f);

  // luffy_shanks has weight 0.95, so it should be filtered OUT
  // Other connections have weight 1.0, so they should be present
  std::unordered_set<std::string> neighbors;
  for (const auto &id : result["depth_1"]) {
    neighbors.insert(id.get<std::string>());
  }

  EXPECT_TRUE(neighbors.count("zoro"));       // weight 1.0
  EXPECT_TRUE(neighbors.count("nami"));       // weight 1.0
  EXPECT_TRUE(neighbors.count("straw_hats")); // weight 1.0
  EXPECT_FALSE(neighbors.count("shanks"));    // weight 0.95 - filtered out!
}

TEST(MemoryGraphTest, QueryFindsAllNeighborsRegardlessOfOrder) {
  MemoryGraph graph = createTestGraph();

  json result = graph.query("luffy", 1, 0.0f); // Include all weights

  std::unordered_set<std::string> neighbors;
  for (const auto &id : result["depth_1"]) {
    neighbors.insert(id.get<std::string>());
  }

  // All neighbors regardless of weight should be found
  EXPECT_EQ(neighbors.size(), 4);
  EXPECT_TRUE(neighbors.count("zoro"));
  EXPECT_TRUE(neighbors.count("nami"));
  EXPECT_TRUE(neighbors.count("shanks"));
  EXPECT_TRUE(neighbors.count("straw_hats"));
}

TEST(MemoryGraphTest, DuplicateIdError) {
  MemoryGraph graph;
  Node node("luffy", "Luffy");
  graph.addNode(node);

  try {
    graph.addNode(node);
    FAIL() << "Expected DuplicateIdError";
  } catch (const DuplicateIdError &e) {
    EXPECT_STREQ(e.what(), "Node with ID 'luffy' already exists.");
  }
}

TEST(MemoryGraphTest, NodeNotFoundError) {
  MemoryGraph graph;

  try {
    graph.getNode("nonexistent");
    FAIL() << "Expected NodeNotFoundError";
  } catch (const NodeNotFoundError &e) {
    EXPECT_STREQ(e.what(), "Node with ID 'nonexistent' not found.");
  }
}

// Integration Tests
TEST(MemoryGraphTest, ComplexGraphOperations) {
  MemoryGraph graph;

  // Create a small networks
  Node geralt("geralt", "Geralt");
  Node vesemir("vesemir", "Vesemir");
  Node triss("triss", "Triss");
  Node yennefer("yennefer", "Yennefer");

  graph.addNode(vesemir);
  graph.addNode(geralt);
  graph.addNode(triss);
  graph.addNode(yennefer);

  // Add edges
  AsymmetricConnections vesemirGeralt{"vesemir", "geralt"};
  Edge edge1("vesemir_geralt", "mentored_by", EdgeType::ASYMMETRIC,
             vesemirGeralt, 1.0f);
  graph.addEdge(edge1);

  SymmetricConnections geraltTriss{"geralt", "triss"};
  Edge edge2("geralt_triss", "lovers", EdgeType::SYMMETRIC, geraltTriss,
             1.0f); // I like Triss, fight me!
  graph.addEdge(edge2);

  SymmetricConnections trissYennefer{"triss", "yennefer"};
  Edge edge3("triss_yennefer", "friends", EdgeType::SYMMETRIC, trissYennefer,
             0.5f);
  graph.addEdge(edge3);

  // Query from Vesemir
  json result = graph.query("vesemir", 2);

  EXPECT_EQ(result["depth_0"][0], "vesemir");
  EXPECT_EQ(result["depth_1"].size(), 1);
  EXPECT_EQ(result["depth_1"][0], "geralt");
  EXPECT_EQ(result["depth_2"].size(), 1);
  EXPECT_EQ(result["depth_2"][0], "triss");

  // Remove Gerlat and verify
  graph.removeNode("geralt");
  EXPECT_FALSE(graph.hasNode("geralt"));
  EXPECT_FALSE(graph.hasEdge("geralt_triss"));
  EXPECT_FALSE(graph.hasEdge("vesemir_geralt"));

  // Vesemir should now have no connections
  EXPECT_EQ(graph.getNeighbors("vesemir").size(), 0);
}
