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

    auto it = adjList.find(currentId);
    if (it != adjList.end()) {
      for (const auto &entry : it->second) {
        if (visited.find(entry.neighbor_id) == visited.end()) {
          visited.insert(entry.neighbor_id);
          queue.push({entry.neighbor_id, depth + 1});
        }
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

    auto it = adjList.find(currentId);
    if (it != adjList.end()) {
      const auto &neighbors = it->second;
      for (auto entryIt = neighbors.rbegin(); entryIt != neighbors.rend();
           ++entryIt) {
        if (visited.find(entryIt->neighbor_id) == visited.end()) {
          stack.push({entryIt->neighbor_id, depth + 1});
        }
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

    auto it = adjList.find(currentId);
    if (it != adjList.end()) {
      for (const auto &entry : it->second) {
        if (visited.find(entry.neighbor_id) == visited.end()) {
          visited.insert(entry.neighbor_id);
          parent[entry.neighbor_id] = currentId;
          queue.push(entry.neighbor_id);
        }
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
          auto it = adjList.find(currentId);
          if (it != adjList.end()) {
            for (const auto &entry : it->second) {
              if (visited.find(entry.neighbor_id) == visited.end()) {
                explore(entry.neighbor_id, depth + 1);
              }
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

    for (const auto &entry : it->second) {
      // Skip parent only if edge is symmetric
      if (entry.is_symmetric && entry.neighbor_id == parent)
        continue;

      if (visited.find(entry.neighbor_id) == visited.end()) {
        if (dfsCycle(entry.neighbor_id, nodeId)) {
          return true;
        }
      } else if (recursionStack.find(entry.neighbor_id) !=
                 recursionStack.end()) {
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
  for (const auto &[nodeId, neighbors] : adjList) {
    for (const auto &entry : neighbors) {
      // Only count asymmetric edges
      if (!entry.is_symmetric) {
        inDegree[entry.neighbor_id]++;
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
      for (const auto &entry : it->second) {
        // Only decrement for asymmetric edges
        if (!entry.is_symmetric) {
          inDegree[entry.neighbor_id]--;
          if (inDegree[entry.neighbor_id] == 0) {
            queue.push(entry.neighbor_id);
          }
        }
      }
    }
  }

  // Verify all nodes were processed
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
  return visited.size() == graph.getNodesRef().size();
}

// Subgraph Extraction
MemoryGraph subgraph(const MemoryGraph &graph, const std::string &center,
                     int radius, bool includeEdges) {
  if (!graph.hasNode(center)) {
    throw NodeNotFoundError(center);
  }

  // Get all nodes within radius
  auto nodeIds = bfs(graph, center, radius);
  std::unordered_set<std::string> nodeSet(nodeIds.begin(), nodeIds.end());

  // Create new graph with same metadata
  MemoryGraph subgraph(graph.getMetadata());

  const auto &nodesRef = graph.getNodesRef();

  // Add nodes
  for (const auto &id : nodeIds) {
    subgraph.addNode(nodesRef.at(id));
  }

  // Add edges
  if (includeEdges) {
    const auto &edgesRef = graph.getEdgesRef();
    for (const auto &[edgeId, edge] : edgesRef) {
      // Check for both endpoints in the subgraph
      bool hasBothEndpoints = false;

      if (std::holds_alternative<SymmetricConnections>(edge.getConnections())) {
        const auto &conn_set =
            std::get<SymmetricConnections>(edge.getConnections());
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

  const auto &nodesRef = graph.getNodesRef();

  // Find nodes matching predicate
  for (const auto &[nodeId, node] : nodesRef) {
    if (predicate(node)) {
      selectedNodes.insert(nodeId);
      finalNodes.insert(nodeId);
    }
  }

  // Include neighbors if requested
  if (includeNeighbors) {
    // 1. Include outgoing neighbors
    for (const auto &id : selectedNodes) {
      auto it = adjList.find(id);
      if (it != adjList.end()) {
        for (const auto &entry : it->second) {
          finalNodes.insert(entry.neighbor_id);
        }
      }
    }

    // 2. Include incoming neighbors (nodes that have this node as neighbor)
    for (const auto &[nodeId, neighbors] : adjList) {
      for (const auto &entry : neighbors) {
        if (selectedNodes.find(entry.neighbor_id) != selectedNodes.end()) {
          finalNodes.insert(nodeId);
        }
      }
    }
  }

  // Build subgraph
  MemoryGraph subgraph(graph.getMetadata());

  for (const auto &id : finalNodes) {
    subgraph.addNode(nodesRef.at(id));
  }

  // Add edges between selected nodes
  const auto &edgesRef = graph.getEdgesRef();
  for (const auto &[edgeId, edge] : edgesRef) {
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
  const auto &nodesRef = graph.getNodesRef();
  for (const auto &[nodeId, node] : nodesRef) {
    if (node.getLabel() == label) {
      result.push_back(nodeId);
    }
  }
  return result;
}

std::vector<std::string> findNodesByMetadata(const MemoryGraph &graph,
                                             const std::string &key,
                                             const nlohmann::json &value) {
  std::vector<std::string> result;
  const auto &nodesRef = graph.getNodesRef();
  for (const auto &[nodeId, node] : nodesRef) {
    const auto &metadata = node.getMetadata();
    if (metadata.contains(key) && metadata[key] == value) {
      result.push_back(nodeId);
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

  const auto &adjList = graph.getAdjacencyList();
  const int maxDepth = 5; // Default for context window

  nlohmann::json result;
  result["center"] = center;
  result["nodes"] = nlohmann::json::array();
  result["edges"] = nlohmann::json::array();

  // Get BFS traversal
  auto nodeIds = bfs(graph, center, maxDepth);
  size_t tokenCount = 0;

  const auto &nodesRef = graph.getNodesRef();
  const auto &edgesRef = graph.getEdgesRef();

  for (const auto &id : nodeIds) {
    const auto &node = nodesRef.at(id);

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
      for (const auto &entry : it->second) {
        try {
          const auto &edge = edgesRef.at(entry.edge_id);
          if (edge.getWeight() >= minRelevance) {
            result["edges"].push_back(edge.toJson());
          }
        } catch (const std::out_of_range &) {
          // Edge not found, skip
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
