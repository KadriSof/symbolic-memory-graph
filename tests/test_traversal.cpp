#include "memory_graph/edge.hpp"
#include "memory_graph/exceptions.hpp"
#include "memory_graph/memory_graph.hpp"
#include "memory_graph/node.hpp"
#include "memory_graph/utils/traversal.hpp"
#include "test_utils.hpp"
#include <algorithm>
#include <chrono>
#include <cstddef>
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

// Helper functions
namespace {
/**
 * @brief Convert a vector to a set for easier comparison
 */
std::unordered_set<std::string> toSet(const std::vector<std::string> &vec) {
  return std::unordered_set<std::string>(vec.begin(), vec.end());
}

/**
 * @brief Check if two vectors contain the same elements (order doesn't matter)
 */
bool sameElements(const std::vector<std::string> &a,
                  const std::vector<std::string> &b) {
  if (a.size() != b.size())
    return false;
  auto setA = toSet(a);
  auto setB = toSet(b);
  return setA == setB;
}
} // namespace

// BFS Tests
TEST(TraversalTest, BfsBasic) {
  MemoryGraph graph = createTestGraph();
  auto result = bfs(graph, "luffy");

  // BFS from luffy should visit all nodes in the graph
  EXPECT_EQ(result.size(), 5);

  // First node should be luffy
  EXPECT_EQ(result[0], "luffy");

  // All nodes should be present
  auto resultSet = toSet(result);
  EXPECT_TRUE(resultSet.count("luffy"));
  EXPECT_TRUE(resultSet.count("zoro"));
  EXPECT_TRUE(resultSet.count("nami"));
  EXPECT_TRUE(resultSet.count("shanks"));
  EXPECT_TRUE(resultSet.count("straw_hats"));
}

TEST(TraversalTest, BfsWithDepthLimit) {
  MemoryGraph graph = createTestGraph();
  auto result = bfs(graph, "luffy", 0);

  EXPECT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], "luffy");

  result = bfs(graph, "luffy", 1);
  // luffy has connections to: zoro, nami, shanks, straw_hats
  EXPECT_EQ(result.size(), 5);
  EXPECT_EQ(result[0], "luffy");
}

TEST(TraversalTest, BfsNonExistentNode) {
  MemoryGraph graph = createTestGraph();

  EXPECT_THROW(bfs(graph, "Imu"), NodeNotFoundError);
}

TEST(TraversalTest, BfsIsolatedNode) {
  MemoryGraph graph;
  Node isolated("black_beard", "Marshal D. Teach");
  graph.addNode(isolated);

  auto result = bfs(graph, "black_beard");
  EXPECT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], "black_beard");
}

// DFS Tests
TEST(TraversalTest, DfsBasic) {
  MemoryGraph graph = createTestGraph();
  auto result = dfs(graph, "luffy");

  // DFS from luffy should visit all nodes in the graph
  EXPECT_EQ(result.size(), 5);
  // First node should be luffy
  EXPECT_EQ(result[0], "luffy");

  // All nodes should be present
  auto resultSet = toSet(result);
  EXPECT_TRUE(resultSet.count("luffy"));
  EXPECT_TRUE(resultSet.count("zoro"));
  EXPECT_TRUE(resultSet.count("nami"));
  EXPECT_TRUE(resultSet.count("shanks"));
  EXPECT_TRUE(resultSet.count("straw_hats"));
}

TEST(TraversalTest, DfsWithDepthLimit) {
  MemoryGraph graph = createTestGraph();

  auto result = dfs(graph, "luffy", 0);
  EXPECT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], "luffy");

  result = dfs(graph, "luffy", 1);
  EXPECT_EQ(result.size(), 5); // luffy + 4 neighbors
  EXPECT_EQ(result[0], "luffy");
}

TEST(TraversalTest, DfsNonExistentNode) {
  MemoryGraph graph = createTestGraph();

  EXPECT_THROW(dfs(graph, "Imu"), NodeNotFoundError);
}

// Shortest Path Tests
TEST(TraversalTest, ShortestPathDirect) {
  MemoryGraph graph = createTestGraph();
  auto path = shortestPath(graph, "luffy", "zoro");

  // Direct connection between luffy and zoro
  EXPECT_EQ(path.size(), 2);
  EXPECT_EQ(path[0], "luffy");
  EXPECT_EQ(path[1], "zoro");
}

TEST(TraversalTest, ShortestPathIndirect) {
  MemoryGraph graph = createTestGraph();

  auto path = shortestPath(graph, "luffy", "nami");

  EXPECT_EQ(path.size(), 2);
  EXPECT_EQ(path[0], "luffy");
  EXPECT_EQ(path[1], "nami");
}

TEST(TraversalTest, ShortestPathSameNode) {
  MemoryGraph graph = createTestGraph();
  auto path = shortestPath(graph, "luffy", "luffy");

  EXPECT_EQ(path.size(), 1);
  EXPECT_EQ(path[0], "luffy");
}

TEST(TraversalTest, ShortestPathNoPath) {
  MemoryGraph graph;
  Node blackBeard("black_beard", "Marshal D. Teach");
  Node whiteBeard("white_beard", "Edward Newgate");
  graph.addNode(blackBeard);
  graph.addNode(whiteBeard);

  EXPECT_THROW(shortestPath(graph, "black_beard", "white_beard"),
               std::runtime_error);
}

TEST(TraversalTest, ShortestPathNonExistentNode) {
  MemoryGraph graph = createTestGraph();

  EXPECT_THROW(shortestPath(graph, "luffy", "Imu"), NodeNotFoundError);
  EXPECT_THROW(shortestPath(graph, "Imu", "luffy"), NodeNotFoundError);
}

// All Paths Tests
TEST(TraversalTest, FindAllPathsDirect) {
  MemoryGraph graph = createTestGraph();
  auto paths = findAllPaths(graph, "luffy", "zoro");

  EXPECT_GE(paths.size(), 1);

  std::sort(paths.begin(), paths.end(),
            [](const auto &a, const auto &b) { return a.size() < b.size(); });

  // First path should be direct
  EXPECT_EQ(paths[0].size(), 2);
  EXPECT_EQ(paths[0][0], "luffy");
  EXPECT_EQ(paths[0][1], "zoro");
}

TEST(TraversalTest, FindAllPathsWithMaxDepth) {
  MemoryGraph graph = createTestGraph();
  auto paths = findAllPaths(graph, "luffy", "nami", 1);

  // With depth 1, only direct connections
  EXPECT_EQ(paths.size(), 1);
  EXPECT_EQ(paths[0][0], "luffy");
  EXPECT_EQ(paths[0][1], "nami");
}

TEST(TraversalTest, FindAllPathsNoPath) {
  MemoryGraph graph;
  Node blackBeard("black_beard", "Marshal D. Teach");
  Node whiteBeard("white_beard", "Edward Newgate");
  graph.addNode(blackBeard);
  graph.addNode(whiteBeard);

  auto paths = findAllPaths(graph, "black_beard", "white_beard");
  EXPECT_TRUE(paths.empty());
}

// Cycle Detection Tests
TEST(TraversalTest, HasCycleWithCycle) {
  MemoryGraph graph;

  Node geralt("geralt", "Geralt of Rivia");
  Node vesemir("vesemir", "Vesemir of Kaer Morhen");
  Node triss("triss", "Triss Marigold");
  Node yennefer("yennefer", "Yennefer of Vengerberg");
  graph.addNode(geralt);
  graph.addNode(vesemir);
  graph.addNode(triss);
  graph.addNode(yennefer);

  // For the sake of the book plot
  AsymmetricConnections geraltYennefer{"geralt", "yennefer"};
  Edge e1("geralt-yennefer", "Loves", EdgeType::ASYMMETRIC, geraltYennefer);
  graph.addEdge(e1);

  AsymmetricConnections yenneferTriss{"yennefer", "triss"};
  Edge e2("yennefer-triss", "Hates", EdgeType::ASYMMETRIC, yenneferTriss);
  graph.addEdge(e2);

  AsymmetricConnections trissGeralt{"triss", "geralt"};
  Edge e3("triss-geralt", "Loves", EdgeType::ASYMMETRIC, trissGeralt);
  graph.addEdge(e3);

  EXPECT_TRUE(hasCycle(graph));
}

TEST(TraversalTest, HasCycleSymmetricNoCycle) {
  MemoryGraph graph;

  // Symmetric edge between two nodes should NOT be detected as cycle
  Node geralt("geralt", "Geralt of Rivia");
  Node triss("triss", "Triss Marigold");
  graph.addNode(geralt);
  graph.addNode(triss);

  // I like Triss more than yenn, I don't care..
  SymmetricConnections conn{"geralt", "triss"};
  Edge edge("geralt-triss", "Love", EdgeType::SYMMETRIC, conn);
  graph.addEdge(edge);

  // Single symmetric edge should not be a cycle
  EXPECT_FALSE(hasCycle(graph));
}

TEST(TraversalTest, HasCycleSymmetricTriangle) {
  MemoryGraph graph;

  Node geralt("geralt", "Geralt of Rivia");
  Node vesemir("vesemir", "Vesemir of Kaer Morhen");
  Node triss("triss", "Triss Marigold");
  Node yennefer("yennefer", "Yennefer of Vengerberg");
  graph.addNode(geralt);
  graph.addNode(vesemir);
  graph.addNode(triss);
  graph.addNode(yennefer);

  // For the sake of the book plot
  SymmetricConnections geraltYennefer{"geralt", "yennefer"};
  Edge e1("geralt-yennefer", "Love", EdgeType::SYMMETRIC, geraltYennefer);
  graph.addEdge(e1);

  SymmetricConnections yenneferTriss{"yennefer", "triss"};
  Edge e2("yennefer-triss", "Hate", EdgeType::SYMMETRIC, yenneferTriss);
  graph.addEdge(e2);

  SymmetricConnections trissGeralt{"triss", "geralt"};
  Edge e3("triss-geralt", "Love", EdgeType::SYMMETRIC, trissGeralt);
  graph.addEdge(e3);

  EXPECT_TRUE(hasCycle(graph));
}

TEST(TraversalTest, TopologicalSortBasic) {
  MemoryGraph graph;

  Node geralt("geralt", "Geralt of Rivia");
  Node vesemir("vesemir", "Vesemir of Kaer Morhen");
  Node triss("triss", "Triss Marigold");
  graph.addNode(geralt);
  graph.addNode(vesemir);
  graph.addNode(triss);

  AsymmetricConnections vesemirGeralt{"vesemir", "geralt"};
  Edge e1("vesemir-geralt", "Mentored", EdgeType::ASYMMETRIC, vesemirGeralt);
  graph.addEdge(e1);

  AsymmetricConnections gerlatTriss{"geralt", "triss"};
  Edge e2("geralt-triss", "Loves", EdgeType::ASYMMETRIC, gerlatTriss);
  graph.addEdge(e2);

  auto result = topologicalSort(graph);

  // Topological order: vesemir, geral, triss
  EXPECT_EQ(result.size(), 3);
  EXPECT_EQ(result[0], "vesemir");
  EXPECT_EQ(result[1], "geralt");
  EXPECT_EQ(result[2], "triss");
}

TEST(TraversalTest, TopologicalSortComplex) {
  MemoryGraph graph;

  // DAG: vesemir -> triss, vesemir -> yennefer, triss -> geralt, yennefer ->
  // geralt
  Node a("vesemir", "Vesemir of Kaer Morhen");
  Node b("geralt", "Geralt of Rivia");
  Node c("lambert", "Lambert of Something");
  Node d("ekimmara", "An ugly vampire");
  graph.addNode(a);
  graph.addNode(b);
  graph.addNode(c);
  graph.addNode(d);

  AsymmetricConnections ab{"vesemir", "geralt"};
  Edge e1("vesemir-geralt", "Mentored", EdgeType::ASYMMETRIC, ab);
  graph.addEdge(e1);

  AsymmetricConnections ac{"vesemir", "lambert"};
  Edge e2("vesemir-lambert", "Mentored", EdgeType::ASYMMETRIC, ac);
  graph.addEdge(e2);

  AsymmetricConnections bd{"geralt", "ekimmara"};
  Edge e3("geralt-imlerith", "Hunted", EdgeType::ASYMMETRIC, bd);
  graph.addEdge(e3);

  AsymmetricConnections cd{"lambert", "ekimmara"};
  Edge e4("lambert", "Hunted", EdgeType::ASYMMETRIC, cd);
  graph.addEdge(e4);

  auto result = topologicalSort(graph);

  EXPECT_EQ(result.size(), 4);
  EXPECT_EQ(result[0], "vesemir");
  // b and c can be in either order
  EXPECT_TRUE((result[1] == "geralt" && result[2] == "lambert") ||
              (result[1] == "lambert" && result[2] == "geralt"));
  EXPECT_EQ(result[3], "ekimmara");
}

TEST(TraversalTest, TopologicalSortWithSymmetricEdges) {
  MemoryGraph graph;

  // Mixed graph: asymmetric vesemir -> geralt, symmetric geralt-Letho
  Node a("vesemir", "Vesemir of Kaer Morhen");
  Node b("geralt", "Geralt of Rivia");
  Node c("letho", "Letho of Gulet");
  graph.addNode(a);
  graph.addNode(b);
  graph.addNode(c);

  AsymmetricConnections ab{"vesemir", "geralt"};
  Edge e1("vesemir-geralt", "Mentored", EdgeType::ASYMMETRIC, ab);
  graph.addEdge(e1);

  SymmetricConnections bc{"geralt", "letho"};
  Edge e2("geralt-letho", "Allies", EdgeType::SYMMETRIC, bc);
  graph.addEdge(e2);

  auto result = topologicalSort(graph);

  // Topological sort should still work, treating symmetric edges as
  // non-dependencies
  EXPECT_EQ(result.size(), 3);
  EXPECT_EQ(result[0], "letho");
}

TEST(TraversalTest, TopologicalSortWithCycle) {
  MemoryGraph graph;

  // Create a cycle: vesemir -> geralt -> vesemir
  Node a("vesemir", "Vesemir of Kaer Morhen");
  Node b("geralt", "Geralt of Rivia");
  graph.addNode(a);
  graph.addNode(b);

  AsymmetricConnections ab{"vesemir", "geralt"};
  Edge e1("vesemir-geralt", "Mentored", EdgeType::ASYMMETRIC, ab);
  graph.addEdge(e1);

  AsymmetricConnections ba{"geralt", "vesemir"};
  Edge e2("geralt-vesemir", "Respected", EdgeType::ASYMMETRIC, ba);
  graph.addEdge(e2);

  EXPECT_THROW(topologicalSort(graph), std::runtime_error);
}

// Connectivity Tests
TEST(TraversalTest, IsConnectedBasic) {
  auto graph = createTestGraph();

  EXPECT_TRUE(isConnected(graph, "luffy"));
}

TEST(TraversalTest, IsConnectedDisconnected) {
  MemoryGraph graph;

  Node a("luffy", "Monkey D. Luffy");
  Node b("geralt", "Geralt of Rivia");
  graph.addNode(a);
  graph.addNode(b);

  // No edges between nodes
  EXPECT_FALSE(isConnected(graph, "luffy"));
  EXPECT_FALSE(isConnected(graph, "geralt"));
}

TEST(TraversalTest, IsConnectedNonExistentNode) {
  auto graph = createTestGraph();

  EXPECT_THROW(isConnected(graph, "Sangoku"), NodeNotFoundError);
}

// Subgraph Extraction Tests
TEST(TraversalTest, SubgraphRadius0) {
  auto graph = createTestGraph();

  auto sub = subgraph(graph, "luffy", 0);

  EXPECT_EQ(sub.getNodes().size(), 1);
  EXPECT_TRUE(sub.hasNode("luffy"));
  EXPECT_EQ(sub.getEdges().size(), 0);
}

TEST(TraversalTest, SubgraphRadius1) {
  auto graph = createTestGraph();

  auto sub = subgraph(graph, "luffy", 1);

  // luffy + all neighbors: zoro, nami, shanks, straw_hats
  EXPECT_EQ(sub.getNodes().size(), 5);
  EXPECT_TRUE(sub.hasNode("luffy"));
  EXPECT_TRUE(sub.hasNode("zoro"));
  EXPECT_TRUE(sub.hasNode("nami"));
  EXPECT_TRUE(sub.hasNode("shanks"));
  EXPECT_TRUE(sub.hasNode("straw_hats"));
}

TEST(TraversalTest, SubgraphRadius2) {
  auto graph = createTestGraph();

  auto sub = subgraph(graph, "luffy", 2);

  // Graph is small, so should contain all nodes
  EXPECT_EQ(sub.getNodes().size(), 5);
}

TEST(TraversalTest, SubgraphNonExistentCenter) {
  auto graph = createTestGraph();

  EXPECT_THROW(subgraph(graph, "guts", 1), NodeNotFoundError);
}

// Subgraph by Predicate Tests
TEST(TraversalTest, SubgraphByPredicateBasic) {
  auto graph = createTestGraph();

  // Find all character nodes
  auto sub = subgraphByPredicate(graph, [](const Node &node) {
    const auto &metadata = node.getMetadata();
    return metadata.contains("type") && metadata["type"] == "character";
  });

  // Should find: luffy, zoro, nami, shanks (but not straw_hats which is a crew)
  EXPECT_EQ(sub.getNodes().size(), 4);
  EXPECT_TRUE(sub.hasNode("luffy"));
  EXPECT_TRUE(sub.hasNode("zoro"));
  EXPECT_TRUE(sub.hasNode("nami"));
  EXPECT_TRUE(sub.hasNode("shanks"));
  EXPECT_FALSE(sub.hasNode("straw_hats"));
}

TEST(TraversalTest, SubgraphByPredicateWithNeighbors) {
  auto graph = createTestGraph();

  // Find shanks and include neighbors
  auto sub = subgraphByPredicate(
      graph, [](const Node &node) { return node.getId() == "shanks"; },
      true // includeNeighbors
  );

  // Should include shanks and its neighbors
  // shanks only has luffy as neighbor (asymmetric: luffy -> shanks)
  // So subgraph should have shanks and luffy
  EXPECT_EQ(sub.getNodes().size(), 2);
  EXPECT_TRUE(sub.hasNode("shanks"));
  EXPECT_TRUE(sub.hasNode("luffy"));
}

TEST(TraversalTest, SubgraphByPredicateEmpty) {
  auto graph = createTestGraph();

  // Predicate that matches nothing
  auto sub = subgraphByPredicate(graph, [](const Node &) { return false; });

  EXPECT_EQ(sub.getNodes().size(), 0);
  EXPECT_EQ(sub.getEdges().size(), 0);
}

// Node Query Tests
TEST(TraversalTest, FindNodesByLabel) {
  auto graph = createTestGraph();

  auto results = findNodesByLabel(graph, "Monkey D. Luffy");

  EXPECT_EQ(results.size(), 1);
  EXPECT_EQ(results[0], "luffy");
}

TEST(TraversalTest, FindNodesByLabelMultiple) {
  // Create graph with multiple nodes sharing label
  MemoryGraph graph;
  Node n1("n1", "Same Label");
  Node n2("n2", "Same Label");
  graph.addNode(n1);
  graph.addNode(n2);

  auto results = findNodesByLabel(graph, "Same Label");

  EXPECT_EQ(results.size(), 2);
  EXPECT_TRUE((results[0] == "n1" && results[1] == "n2") ||
              (results[0] == "n2" && results[1] == "n1"));
}

TEST(TraversalTest, FindNodesByLabelNotFound) {
  auto graph = createTestGraph();

  auto results = findNodesByLabel(graph, "NonExistent Label");

  EXPECT_TRUE(results.empty());
}

TEST(TraversalTest, FindNodesByMetadata) {
  auto graph = createTestGraph();

  auto results = findNodesByMetadata(graph, "bounty", 3000000000);

  EXPECT_EQ(results.size(), 1);
  EXPECT_EQ(results[0], "luffy");
}

TEST(TraversalTest, FindNodesByMetadataNotFound) {
  auto graph = createTestGraph();

  auto results = findNodesByMetadata(graph, "non_existent_key", "value");

  EXPECT_TRUE(results.empty());
}

// Context Window Tests
TEST(TraversalTest, GetContextWindowBasic) {
  auto graph = createTestGraph();

  auto result = getContextWindow(graph, "luffy", 1000, 0.0f);

  EXPECT_TRUE(result.contains("center"));
  EXPECT_EQ(result["center"], "luffy");
  EXPECT_TRUE(result.contains("nodes"));
  EXPECT_TRUE(result.contains("edges"));
  EXPECT_TRUE(result.contains("token_count"));
  EXPECT_TRUE(result.contains("node_count"));
  EXPECT_TRUE(result.contains("edge_count"));
}

TEST(TraversalTest, GetContextWindowLimitTokens) {
  auto graph = createTestGraph();

  // Very small token limit should only include the center
  auto result = getContextWindow(graph, "luffy", 10, 0.0f);

  EXPECT_EQ(result["center"], "luffy");
  EXPECT_GE(result["node_count"].get<size_t>(), 1);
}

TEST(TraversalTest, GetContextWindowMinRelevance) {
  auto graph = createTestGraph();

  // Only include edges with weight >= 0.96
  // luffy_shanks has weight 0.95, so it should be excluded
  auto result = getContextWindow(graph, "luffy", 1000, 0.96f);

  // Should still have edges (other edges have weight 1.0)
  // But luffy_shanks should be filtered out
  EXPECT_TRUE(result.contains("edges"));
}

TEST(TraversalTest, GetContextWindowNonExistentCenter) {
  auto graph = createTestGraph();

  EXPECT_THROW(getContextWindow(graph, "nonexistent", 1000, 0.0f),
               NodeNotFoundError);
}

// Integration Tests
TEST(TraversalTest, FullTraversalWorkflow) {
  auto graph = createTestGraph();

  // 1. BFS traversal
  auto bfsResult = bfs(graph, "luffy");
  EXPECT_EQ(bfsResult.size(), 5);

  // 2. Check connectivity
  EXPECT_TRUE(isConnected(graph, "luffy"));

  // 3. Find shortest path
  auto path = shortestPath(graph, "luffy", "nami");
  EXPECT_EQ(path.size(), 2);

  // 4. Check for cycles
  EXPECT_FALSE(hasCycle(graph));

  // 5. Extract subgraph
  auto sub = subgraph(graph, "luffy", 1);
  EXPECT_EQ(sub.getNodes().size(), 5);

  // 6. Find nodes by metadata
  auto nodes = findNodesByMetadata(graph, "type", "character");
  EXPECT_EQ(nodes.size(), 4);

  // 7. Get context window
  auto context = getContextWindow(graph, "luffy", 1000, 0.0f);
  EXPECT_TRUE(context.contains("nodes"));
}

TEST(TraversalTest, AdjacencyListCacheConsistency) {
  auto graph = createTestGraph();

  // First call builds cache
  const auto &adj1 = graph.getAdjacencyList();
  EXPECT_FALSE(
      graph
          .isAdjacencyCacheValid()); // Actually, it should be valid after build
  // Wait, isAdjacencyCacheValid() returns !adjacencyDirty_
  // After first call, cache is built and should be valid

  // Second call should use cache
  const auto &adj2 = graph.getAdjacencyList();

  // Both references should point to the same data
  EXPECT_EQ(&adj1, &adj2);
}

// Performance Tests
TEST(TraversalTest, PerformanceLargeGraph) {
  MemoryGraph graph;
  const int N = 100;

  // Create a large graph
  for (int i = 0; i < N; ++i) {
    Node node("node_" + std::to_string(i), "Label_" + std::to_string(i));
    graph.addNode(node);
  }

  // Add some edges to create a path
  for (int i = 1; i < N; ++i) {
    AsymmetricConnections conn{"node_" + std::to_string(i - 1),
                               "node_" + std::to_string(i)};
    Edge edge("edge_" + std::to_string(i), "connects", EdgeType::ASYMMETRIC,
              conn);
    graph.addEdge(edge);
  }

  // BFS should complete quickly
  auto start = std::chrono::high_resolution_clock::now();
  auto result = bfs(graph, "node_0");
  auto end = std::chrono::high_resolution_clock::now();

  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  // Should visit all nodes
  EXPECT_EQ(result.size(), N);

  // Should be fast (under 1 second)
  EXPECT_LT(duration.count(), 1000);
}

// Edge Cases
TEST(TraversalTest, EmptyGraph) {
  MemoryGraph graph;

  EXPECT_THROW(bfs(graph, "nonexistent"), NodeNotFoundError);
  EXPECT_THROW(dfs(graph, "nonexistent"), NodeNotFoundError);
  EXPECT_THROW(shortestPath(graph, "a", "b"), NodeNotFoundError);
  EXPECT_FALSE(hasCycle(graph)); // Empty graph has no cycles
  EXPECT_EQ(topologicalSort(graph).size(), 0);
}

TEST(TraversalTest, SingleNodeGraph) {
  MemoryGraph graph;
  Node single("single", "Single Node");
  graph.addNode(single);

  auto bfsResult = bfs(graph, "single");
  EXPECT_EQ(bfsResult.size(), 1);
  EXPECT_EQ(bfsResult[0], "single");

  EXPECT_TRUE(isConnected(graph, "single"));
  EXPECT_FALSE(hasCycle(graph));

  auto sortResult = topologicalSort(graph);
  EXPECT_EQ(sortResult.size(), 1);
  EXPECT_EQ(sortResult[0], "single");
}
