#include "memory_graph/utils/traversal.hpp"
#include "memory_graph/edge.hpp"
#include "memory_graph/exceptions.hpp"
#include "memory_graph/memory_graph.hpp"
#include "memory_graph/node.hpp"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cstddef>
#include <functional>
#include <queue>
#include <stack>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace memory_graph::utils {

// Helper: get adjacency list with error checking
namespace {
/**
 * @brief Get adjacency list for a node, with proper error handling
 * @param adjList The full adjacency list
 * @param nodeId The node to look up
 * @return Reference to the adjacency vector, or empty vector if not found
 */
const std::vector<std::pair<std::string, bool>> &getNeighborList(
    const std::unordered_map<
        std::string, std::vector<std::pair<std::string, bool>>> &adjList,
    const std::string &nodeId) {
  static const std::vector<std::pair<std::string, bool>> empty;
  auto it = adjList.find(nodeId);
  if (it == adjList.end()) {
    return empty;
  }

  return it->second;
}
} // namespace

// BFS Implementation
std::vector<std::string> bfs(const MemoryGraph &graph, const std::string &start,
                             int maxDepth) {
  if (!graph.hasNode(start)) {
    throw NodeNotFoundError(start);
  }

  const auto &adjList = graph.getAdjacencyList();

  std::vector<std::string> result;
  std::unordered_set<std::string> visited;
  std::queue<std::pair<std::string, int>> queue;

  queue.push({start, 0});
  visited.insert(start);

  while (!queue.empty()) {
    auto [currentId, depth] = queue.front();
    queue.pop();

    result.push_back(currentId);

    if (maxDepth >= 0 && depth >= maxDepth) {
      continue;
    }

    for (const auto &[neighborId, isSymmetric] :
         getNeighborList(adjList, currentId)) {
      if (visited.find(neighborId) == visited.end()) {
        visited.insert(neighborId);
        queue.push({neighborId, depth + 1});
      }
    }
  }

  return result;
}

// DFS Implementation
std::vector<std::string> dfs(const MemoryGraph &graph, const std::string &start,
                             int maxDepth) {
  if (!graph.hasNode(start)) {
    throw NodeNotFoundError(start);
  }

  const auto &adjList = graph.getAdjacencyList();

  std::vector<std::string> result;
  std::unordered_set<std::string> visited;
  std::stack<std::pair<std::string, int>> stack;

  stack.push({start, 0});

  while (!stack.empty()) {
    auto [currentId, depth] = stack.top();
    stack.pop();

    if (visited.find(currentId) != visited.end()) {
      continue;
    }

    visited.insert(currentId);
    result.push_back(currentId);

    if (maxDepth >= 0 && depth >= maxDepth) {
      continue;
    }

    // Push neighbors in reverse order to maintain natural order
    const auto &neighbors = getNeighborList(adjList, currentId);
    for (auto it = neighbors.rbegin(); it != neighbors.rend(); ++it) {
      const auto &[neighborId, isSymmetric] = *it;
      if (visited.find(neighborId) == visited.end()) {
        stack.push({neighborId, depth + 1});
      }
    }
  }

  return result;
}

// Path Finding
std::vector<std::string> shortestPath(const MemoryGraph &graph,
                                      const std::string &from,
                                      const std::string &to) {
  if (!graph.hasNode(from)) {
    throw NodeNotFoundError(from);
  }
  if (!graph.hasNode(to)) {
    throw NodeNotFoundError(to);
  }

  if (from == to) {
    return {from};
  }

  const auto &adjList = graph.getAdjacencyList();

  // BFS to find shortest path
  std::unordered_set<std::string> visited;
  std::unordered_map<std::string, std::string> parent;
  std::queue<std::string> queue;

  queue.push(from);
  visited.insert(from);
  parent[from] = "";

  while (!queue.empty()) {
    std::string currentId = queue.front();
    queue.pop();

    if (currentId == to) {
      // Reconstruct path
      std::vector<std::string> path;
      std::string nodeId = to;
      while (!nodeId.empty()) {
        path.push_back(nodeId);
        nodeId = parent[nodeId];
      }
      std::reverse(path.begin(), path.end());
      return path;
    }

    for (const auto &[neighborId, isSymmetric] :
         getNeighborList(adjList, currentId)) {
      if (visited.find(neighborId) == visited.end()) {
        visited.insert(neighborId);
        parent[neighborId] = currentId;
        queue.push(neighborId);
      }
    }
  }

  throw std::runtime_error("[utils:traversal] No path found between '" + from +
                           "' and '" + to + "'");
}

std::vector<std::vector<std::string>> findAllPaths(const MemoryGraph &graph,
                                                   const std::string &from,
                                                   const std::string &to,
                                                   int maxDepth) {
  if (!graph.hasNode(from)) {
    throw NodeNotFoundError(from);
  }
  if (!graph.hasNode(to)) {
    throw NodeNotFoundError(to);
  }

  const auto &adjList = graph.getAdjacencyList();

  std::vector<std::vector<std::string>> paths;
  std::vector<std::string> currentPath;
  std::unordered_set<std::string> visited;

  std::function<void(const std::string &, int)> explore =
      [&](const std::string &currentId, int depth) {
        currentPath.push_back(currentId);
        visited.insert(currentId);

        if (currentId == to) {
          paths.push_back(currentPath);
        } else if (maxDepth < 0 || depth < maxDepth) {
          for (const auto &[neighborId, isSymmetric] :
               getNeighborList(adjList, currentId)) {
            if (visited.find(neighborId) == visited.end()) {
              explore(neighborId, depth + 1);
            }
          }
        }

        // Backtrack
        visited.erase(currentId);
        currentPath.pop_back();
      };

  explore(from, 0);
  return paths;
}

// Graph Properties
bool hasCycle(const MemoryGraph &graph) {
  const auto &adjList = graph.getAdjacencyList();

  std::unordered_set<std::string> visited;
  std::unordered_set<std::string> recursionStack;

  std::function<bool(const std::string &, const std::string &)> dfsCycle =
      [&](const std::string &nodeId, const std::string &parent) -> bool {
    visited.insert(nodeId);
    recursionStack.insert(nodeId);

    auto it = adjList.find(nodeId);
    if (it == adjList.end()) {
      recursionStack.erase(nodeId);
      return false;
    }

    for (const auto &[neighborId, isSymmetric] : it->second) {
      // Skip parent only if edge is symmetric
      if (isSymmetric && neighborId == parent)
        continue;

      if (visited.find(neighborId) == visited.end()) {
        if (dfsCycle(neighborId, nodeId)) {
          return true;
        }
      } else if (recursionStack.find(neighborId) != recursionStack.end()) {
        return true; // cycle detected
      }
    }

    recursionStack.erase(nodeId);
    return false;
  };

  // Check all nodes
  for (const auto &[nodeId, _] : adjList) {
    if (visited.find(nodeId) == visited.end()) {
      if (dfsCycle(nodeId, "")) {
        return true;
      }
    }
  }

  return false;
}

std::vector<std::string> topologicalSort(const MemoryGraph &graph) {
  if (hasCycle(graph)) {
    throw std::runtime_error(
        "[utils:traversal] Graph has a cycle - topological sort impossible");
  }

  const auto &adjList = graph.getAdjacencyList();

  std::unordered_map<std::string, int> inDegree;
  std::queue<std::string> queue;
  std::vector<std::string> result;

  // Initialize in-degree for all nodes
  for (const auto &[nodeId, _] : adjList) {
    inDegree[nodeId] = 0;
  }

  // Count incoming edges
  for (const auto &[node, neighbors] : adjList) {
    for (const auto &[neighborId, isSymmetric] : neighbors) {
      // Only count asymmetric edges since undirected connections
      // don't contribute to dependency ordering in the same way
      if (!isSymmetric) {
        inDegree[neighborId]++;
      }
    }
  }

  // Find nodes with 0 in-degree
  for (const auto &[id, count] : inDegree) {
    if (count == 0) {
      queue.push(id);
    }
  }

  // Process queue
  while (!queue.empty()) {
    std::string nodeId = queue.front();
    queue.pop();
    result.push_back(nodeId);

    auto it = adjList.find(nodeId);
    if (it != adjList.end()) {
      for (const auto &[neighbordId, isSymmetric] : it->second) {
        // Only decrement for asymmetric edges
        if (!isSymmetric) {
          inDegree[neighbordId]--;
          if (inDegree[neighbordId] == 0) {
            queue.push(neighbordId);
          }
        }
      }
    }
  }

  // Verify all nodes were processed (for diconnected nodes or cycles)
  if (result.size() != adjList.size()) {
    for (const auto &[nodeId, _] : adjList) {
      if (std::find(result.begin(), result.end(), nodeId) == result.end()) {
        result.push_back(nodeId);
      }
    }
  }

  return result;
}

bool isConnected(const MemoryGraph &graph, const std::string &start) {
  if (!graph.hasNode(start)) {
    throw NodeNotFoundError(start);
  }

  auto visited = bfs(graph, start);
  return visited.size() == graph.getNodes().size();
}

// Subgraph Extraction
MemoryGraph subgraph(const MemoryGraph &graph, const std::string &center,
                     int radius, bool includeEdges) {
  if (!graph.hasNode(center)) {
    throw NodeNotFoundError(center);
  }

  const auto &adjList = graph.getAdjacencyList();

  // Get all nodes within radius
  auto nodeIds = bfs(graph, center, radius);
  std::unordered_set<std::string> nodeSet(nodeIds.begin(), nodeIds.end());

  // Create new graph with same metadata
  MemoryGraph subgraph(graph.getMetadata());

  // Add nodes
  for (const auto &id : nodeIds) {
    subgraph.addNode(graph.getNode(id));
  }

  // Add edges
  if (includeEdges) {
    for (const auto &edge : graph.getEdges()) {
      // Check for both endpoints in the subgraph
      bool hasBothEndpoints = false;

      if (std::holds_alternative<SymmetricConnections>(edge.getConnections())) {
        const auto &conn_set =
            std::get<SymmetricConnections>(edge.getConnections());
        // For symmetric, check if ALL nodes are in subgraph
        bool allInSubgraph = true;
        for (const auto &nodeId : conn_set) {
          if (nodeSet.find(nodeId) == nodeSet.end()) {
            allInSubgraph = false;
            break;
          }
        }
        hasBothEndpoints = allInSubgraph;
      } else {
        const auto &conn_pair =
            std::get<AsymmetricConnections>(edge.getConnections());
        hasBothEndpoints = (nodeSet.find(conn_pair.first) != nodeSet.end() &&
                            nodeSet.find(conn_pair.second) != nodeSet.end());
      }

      if (hasBothEndpoints) {
        if (edge.isGroupEdge()) {
          const auto &conn_set =
              std::get<SymmetricConnections>(edge.getConnections());
          std::unordered_set<std::string> nodeIds(conn_set.begin(),
                                                  conn_set.end());
          subgraph.addGroupEdge(edge.getId(), edge.getLabel(), nodeIds,
                                edge.getWeight(), edge.getMetadata());
        } else {
          subgraph.addEdge(edge);
        }
      }
    }
  }

  return subgraph;
}

MemoryGraph
subgraphByPredicate(const MemoryGraph &graph,
                    const std::function<bool(const Node &)> &predicate,
                    bool includeNeighbors) {
  const auto &adjList = graph.getAdjacencyList();

  std::unordered_set<std::string> selectedNodes;
  std::unordered_set<std::string> finalNodes;

  // Find nodes matching predicate
  for (const auto &node : graph.getNodes()) {
    if (predicate(node)) {
      selectedNodes.insert(node.getId());
      finalNodes.insert(node.getId());
    }
  }

  // Include neighbors if requested
  if (includeNeighbors) {
    // 1. Include outgoing neighbors
    for (const auto &id : selectedNodes) {
      auto it = adjList.find(id);
      if (it != adjList.end()) {
        for (const auto &[neighborId, isSymmetric] : it->second) {
          finalNodes.insert(neighborId);
        }
      }

      // 2. Include incoming neighbors (nodes that have this node as neighbor)
      for (const auto &[nodeId, neighbors] : adjList) {
        for (const auto &[neighborId, isSymmetric] : neighbors) {
          if (neighborId == id) {
            finalNodes.insert(nodeId);
          }
        }
      }
    }
  }

  // Build subgraph
  MemoryGraph subgraph(graph.getMetadata());

  for (const auto &id : finalNodes) {
    subgraph.addNode(graph.getNode(id));
  }

  // Add edges between selected nodes
  for (const auto &edge : graph.getEdges()) {
    bool hasBothEndpoints = false;

    if (std::holds_alternative<SymmetricConnections>(edge.getConnections())) {
      const auto &conn_set =
          std::get<SymmetricConnections>(edge.getConnections());
      bool allInSubgraph = true;
      for (const auto &nodeId : conn_set) {
        if (finalNodes.find(nodeId) == finalNodes.end()) {
          allInSubgraph = false;
          break;
        }
      }
      hasBothEndpoints = allInSubgraph;
    } else {
      const auto &conn_pair =
          std::get<AsymmetricConnections>(edge.getConnections());
      hasBothEndpoints =
          (finalNodes.find(conn_pair.first) != finalNodes.end() &&
           finalNodes.find(conn_pair.second) != finalNodes.end());
    }

    if (hasBothEndpoints) {
      if (edge.isGroupEdge()) {
        const auto &conn_set =
            std::get<SymmetricConnections>(edge.getConnections());
        std::unordered_set<std::string> nodeIds(conn_set.begin(),
                                                conn_set.end());
        subgraph.addGroupEdge(edge.getId(), edge.getLabel(), nodeIds,
                              edge.getWeight(), edge.getMetadata());
      } else {
        subgraph.addEdge(edge);
      }
    }
  }

  return subgraph;
}

// Node Query Helpers
std::vector<std::string> findNodesByLabel(const MemoryGraph &graph,
                                          const std::string &label) {
  std::vector<std::string> result;
  for (const auto &node : graph.getNodes()) {
    if (node.getLabel() == label) {
      result.push_back(node.getId());
    }
  }

  return result;
}

std::vector<std::string> findNodesByMetadata(const MemoryGraph &graph,
                                             const std::string &key,
                                             const nlohmann::json &value) {
  std::vector<std::string> result;
  for (const auto &node : graph.getNodes()) {
    const auto &metadata = node.getMetadata();
    if (metadata.contains(key) && metadata[key] == value) {
      result.push_back(node.getId());
    }
  }

  return result;
}

// Context Window Utilities (for LLMs)
nlohmann::json getContextWindow(const MemoryGraph &graph,
                                const std::string &center, size_t maxTokens,
                                float minRelevance) {
  if (!graph.hasNode(center)) {
    throw NodeNotFoundError(center);
  }

  // Get nodes within radius (weighted by relevance) - simplified
  const auto &adjList = graph.getAdjacencyList();
  const int maxDepth = 5; // Default for context window

  nlohmann::json result;
  result["center"] = center;
  result["nodes"] = nlohmann::json::array();
  result["edges"] = nlohmann::json::array();

  // Get BFS traversal
  auto nodeIds = bfs(graph, center, maxDepth);
  size_t tokenCount = 0;

  for (const auto &id : nodeIds) {
    const auto &node = graph.getNode(id);

    // Estimate token count (simplified: ~4 chars per token)
    size_t nodeTokens =
        node.getLabel().length() / 4 + node.getId().length() / 4 + 10;

    if (tokenCount + nodeTokens > maxTokens && id != center) {
      break;
    }

    tokenCount += nodeTokens;
    result["nodes"].push_back({{"id", id},
                               {"label", node.getLabel()},
                               {"metadata", node.getMetadata()}});

    // Add edges from this node
    auto it = adjList.find(id);
    if (it != adjList.end()) {
      for (const auto &[neighborId, isSymmetric] : it->second) {
        // Find the actual edge between id and neighborId
        for (const auto &edge : graph.getEdges()) {
          bool isConnected = false;

          if (std::holds_alternative<SymmetricConnections>(
                  edge.getConnections())) {
            const auto &conn_set =
                std::get<SymmetricConnections>(edge.getConnections());
            if (conn_set.find(id) != conn_set.end() &&
                conn_set.find(neighborId) != conn_set.end()) {
              isConnected = true;
            }
          } else {
            const auto &conn_pair =
                std::get<AsymmetricConnections>(edge.getConnections());
            if ((conn_pair.first == id && conn_pair.second == neighborId) ||
                (conn_pair.first == neighborId && conn_pair.second == id)) {
              isConnected = true;
            }
          }

          if (isConnected && edge.getWeight() >= minRelevance) {
            result["edges"].push_back(edge.toJson());
            break; // found the edge, move to next neighbor
          }
        }
      }
    }
  }

  result["token_count"] = tokenCount;
  result["node_count"] = result["nodes"].size();
  result["edge_count"] = result["edges"].size();

  return result;
}

} // namespace memory_graph::utils
