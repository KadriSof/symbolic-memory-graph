#include "memory_graph/utils/traversal.hpp"
#include "memory_graph/exceptions.hpp"
#include "memory_graph/memory_graph.hpp"
#include <algorithm>
#include <functional>
#include <queue>
#include <stack>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace memory_graph::utils {
// BFS Implementation
std::vector<std::string> bfs(const MemoryGraph &graph, const std::string &start,
                             int maxDepth) {
  if (!graph.hasNode(start)) {
    throw NodeNotFoundError(start);
  }

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

    for (const auto &neighborId : graph.getNode(currentId).getConnections()) {
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
    const auto &connections = graph.getNode(currentId).getConnections();
    for (auto it = connections.rbegin(); it != connections.rend(); ++it) {
      const auto &neighborId = *it;
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

    for (const auto &neighborId : graph.getNode(currentId).getConnections()) {
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

  std::vector<std::vector<std::string>> paths;
  std::vector<std::string> currentPath;
  std::unordered_set<std::string> visited;

  std::fucntion<void(const std::string &, int)> explore =
      [&](const std::string &currentId, int depth) {
        currentPath.push_back(currentId);
        visited.insert(currentId);

        if (currentId == to) {
          paths.push_back(currentPath);
        } else if (maxDepth < 0 || depth < maxDepth) {
          for (const auto &neighborId :
               graph.getNode(currentId).getConnections()) {
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

} // namespace memory_graph::utils
