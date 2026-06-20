#pragma once

#include "memory_graph/memory_graph.hpp"
#include "memory_graph/node.hpp"
#include "nlohmann/json.hpp"
#include <cstddef>
#include <functional>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <vector>

namespace memory_graph::utils {
// Basic Traversals
/**
 * @brief Breadth-First Search Traversal starting from a node
 * @param graph The graph to traverse
 * @param start Start node ID
 * @param maxDepth Maximum depth to traverse (-1 for unlimited)
 * @return Ordered list of node IDs in BFS order
 * @throws NodeNotFoundError if start node doesn't exist
 */
std::vector<std::string> bfs(const MemoryGraph &graph, const std::string &start,
                             int maxDepth = -1);

/**
 * @brief Depth-First Search traversal starting from a node
 * @param graph The graph to traverse
 * @param start Start node ID
 * @param maxDepth Maximum depth to traverse (-1 for unlimited)
 * @return Ordered list of node IDs in DFS order
 * @throws NodeNotFoundError if start node doesn't exist
 */
std::vector<std::string> dfs(const MemoryGraph &graph, const std::string &start,
                             int maxDepth = -1);

// Path Finding
/**
 * @brief Find the shortest path between two nodes (unweighted)
 * @param graph The graph to Search
 * @param from Start node ID
 * @param to Target node ID
 * @return Vector of node IDs representing the path (including start and end)
 * @throws NodeNotFoundError if either node doesn't exist
 * @throws std::runtime_error if no path exists
 */
std::vector<std::string> shortestPath(const MemoryGraph &graph,
                                      const std::string &from,
                                      const std::string &to);

/**
 * @brief Find all paths between two nodes (up to maxDepth)
 * @param graph The graph to search
 * @param from Start node ID
 * @param to Target node ID
 * @param maxDepth Maximum depth to search (-1 for unlimited)
 * @return Vector of paths (each path is a vector of node IDs)
 * @throws NodeNotFoundError if either node doesn't exist
 */
std::vector<std::vector<std::string>> findAllPaths(const MemoryGraph &graph,
                                                   const std::string &from,
                                                   const std::string &to,
                                                   int maxDepth = -1);

// Graph Properties
/**
 * @brief Check if the graph contains a cycle
 * @param graph The graph to check
 * @return true if a cycle exists
 */
bool hasCycle(const MemoryGraph &graph);

/**
 * @brief Perform topological sort on a directed acyclic graph
 * @param graph The graph to sort
 * @return Ordered list of node IDs (dependencies first)
 * @throws std::runtime_error if graph has a cycle
 */
std::vector<std::string> topologicalSort(const MemoryGraph &graph);

/**
 * @brief Check if graph is connected (from a given start node)
 * @param graph The graph to check
 * @param start Start node to check connectivity from
 * @return true if all nodes are reacheable from start
 */
bool isConnected(const MemoryGraph &graph, const std::string &start);

// Subgraph Extraction
/**
 * @brief Extract a subgraph centered at a node within a radius
 * @param graph The source graph
 * @param center Center node ID
 * @param radius Maximum distance from center (0 = just center)
 * @param includeEdges Whether to include edges in subgraph
 * @return A new MemoryGraph containing the extracted subgraph
 * @throws NodeNotFoundError if center doesn't exist
 */
MemoryGraph subgraph(const MemoryGraph &graph, const std::string &center,
                     int radius, bool includeEdges = true);

/**
 * @brief Extract a subgraph containing all nodes matching a predicate
 * @param graph The source graph
 * @param predicate Function that returns true for nodes to include
 * @param includeNeighbors Whether to include neighbors of selected nodes
 * @return A new MemoryGraph containing the extracted subgraph
 */
MemoryGraph
subgraphByPredicate(const MemoryGraph &graph,
                    const std::function<bool(const Node &)> &predicate,
                    bool includeNeighbors = false);

// Node Query Helpers
/**
 * @brief Find nodes by label
 * @param graph The graph to search
 * @param label The label to match
 * @return Vector of matching node IDs
 */
std::vector<std::string> findNodesByLabel(const MemoryGraph &graph,
                                          const std::string &label);

/**
 * @brief Find nodes by metadata key-value pair
 * @param graph The graph to search
 * @param key The metadata key
 * @param value The metadata value to match
 * @return Vector of matching node IDs
 */
std::vector<std::string> findNodesByMetadata(const MemoryGraph &graph,
                                             const std::string &key,
                                             const nlohmann::json &value);

// Context Window Utilities (for LLMs)
/**
 * @brief Get a context window for an LLM
 * @param graph The graph to query
 * @param center The center node ID
 * @param maxTokens Maximum tokens to include
 * @param minRelevance Minimum relevance score
 * @return JSON with nodes and edges within the context window
 */
nlohmann::json getContextWidnow(const MemoryGraph &graph,
                                const std::string &center,
                                size_t maxTokens = 4096,
                                float minRelevance = 0.5f);
} // namespace memory_graph::utils
