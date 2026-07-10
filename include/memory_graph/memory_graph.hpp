#pragma once

#include "memory_graph/edge.hpp"
#include "memory_graph/node.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace memory_graph {

/**
 * @class MemoryGraph
 * @brief Represents a graph of nodes and edges for LLM memory.
 */
class MemoryGraph {
public:
  MemoryGraph(const nlohmann::json &metadata = {});

  // Node operations
  void addNode(const Node &node);
  void removeNode(const std::string &nodeId);
  const Node &getNode(const std::string &nodeId) const;
  bool hasNode(const std::string &nodeId) const;
  std::vector<Node> getNodes() const;

  // Edge operations
  /**
   * @brief Add a pairwise edge (exactly 2 nodes)
   * For symmetric edges: creates bidirectinal connection
   * For asymmetric edges: creates directed connection
   * @throws InvalidConnectionError if edge has != 2 nodes for symmetric
   */
  void addEdge(const Edge &edge);

  /**
   * @brief Add a group edge (2+ nodes, symmetric only)
   * Creates a clique where ALL nodes are mutually connected
   * Useful for: teams, factions, famillies, communities, etc.
   *
   * @param id Unique identifier for the group edge
   * @param label Human-readable name (e.g., "Wolf School")
   * @param nodeIds Set of node IDs in the group (must have 2+ nodes)
   * @param weight Strength of group relationship (0.0, 1.0)
   * @param metadata Additional context (e.g., "founded", "location", etc.)
   * @throws InvalidConnectionError if nodeIds size < 2 or nodes don't exist
   *
   * @example
   * std::unordered_set<std::string> witchers = {"geralt", "vesemir",
   * "lambert"}; graph.addGroupEdge("wolf_school", "Wolf School", 1.0f,
   * json{{"founded", "1250"}});
   */
  void addGroupEdge(const std::string &id, const std::string &label,
                    const std::unordered_set<std::string> &nodeIds,
                    float weight = 1.0f, const nlohmann::json &metadata = {});

  void removeEdge(const std::string &edgeId);
  const Edge &getEdge(const std::string &edgeId) const;
  bool hasEdge(const std::string &edgeId) const;
  std::vector<Edge> getEdges() const;

  /**
   * @brief Get all group edges in the graph
   * @return Vector of group edges (symmetric edges with 2+ nodes)
   */
  std::vector<Edge> getGroupEdges() const;

  /**
   * @brief Get all nodes that are members of a group
   * @param groupId The group ID
   * @return Set of node IDs in the group
   * @throws EdgeNotFoundError if group doesn't exist
   */
  std::unordered_set<std::string>
  getGroupMembers(const std::string &groupId) const;

  // Graph operations
  std::vector<Node> getNeighbors(const std::string &nodeId) const;
  nlohmann::json query(const std::string &nodeId, int maxDepth = 1,
                       float minWeight = 0.0f) const;

  /**
   * @brief Get the adjacency list representation of the graph
   * @return Adjacency list where each entry maps nodeId -> [(neighbor,
   * isSymmetric), ...]
   *
   * The adjacency list is cached and rebuilt only when the graph changes.
   * This provides O(1) neighbor lookups for graph algorithms (cool! I know B) )
   *
   * Example usage:
   * const auto &adj = graph.getAdjacencyList();
   * for (const auto &[neighborm isSymmetric] : adj.at("luffy")) {
   *  // neighbor is connected to luffy
   *  // isSymmetric indicated if this is a bidirectinal connection:
   *    - (luffy - zoro : nakama)
   *    - (geralt - lambert: wolf school fellow witcher)
   * }
   */
  const std::unordered_map<std::string,
                           std::vector<std::pair<std::string, bool>>> &
  getAdjacencyList() const;

  /**
   * @brief Check if the adjacency list cache is valid (for debugging)
   * @return true if the cache is valid and up-to-date
   */
  bool isAdjacencyCacheValid() const { return !adjacencyDirty_; }

  // Serialization
  nlohmann::json toJson() const;
  static MemoryGraph fromJson(const nlohmann::json &graphJson);

  // Metadata
  const nlohmann::json &getMetadata() const;
  void setMetadata(const nlohmann::json &metadata);
  void updateMetadata(const std::string &key, const nlohmann::json &value);

private:
  std::unordered_map<std::string, Node> nodes_;
  std::unordered_map<std::string, Edge> edges_;
  nlohmann::json metadata_;

  // Helper to find an edge between two nodes
  std::string findEdgeId(const std::string &nodeId1,
                         const std::string &nodeId2) const;

  /**
   * @brief Cached adjacency list
   * Key: node ID
   * Value: vector of (neighbor_id, is_symmetric)
   */
  mutable std::unordered_map<std::string,
                             std::vector<std::pair<std::string, bool>>>
      adjacencyCache_;

  /**
   * @brief Flag indicating whether the cache needs to be rebuilt
   * Set to true whenever the graph is modified
   */
  mutable bool adjacencyDirty_ = true;

  /**
   * @brief Build/Rebuild the adjacency list cache
   * Called internally by getAdjacencyList() when cache is dirty
   */
  void buildAdjacencyCache() const;

  /**
   * @brief invalidate the adjacency list cache
   * Called whenever the graph structure changes
   */
  void invalidateCache() { adjacencyDirty_ = true; }
};

} // namespace memory_graph
