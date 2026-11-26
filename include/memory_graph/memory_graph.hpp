#pragma once
#include "memory_graph/edge.hpp"
#include "memory_graph/node.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace memory_graph {

enum class GraphMode { KNOWLEDGE, SYMBOLIC };

struct AdjacencyEntry {
  std::string neighbor_id;
  bool is_symmetric;
  std::string edge_id; // FIX #2: O(1) weight lookup
};

class MemoryGraph {
public:
  explicit MemoryGraph(
      const nlohmann::json &metadata = nlohmann::json::object());

  // Mode management
  GraphMode getMode() const;
  void setMode(GraphMode mode);

  // Node operations
  void addNode(const Node &node);
  bool hasNode(const std::string &nodeId) const;
  void removeNode(const std::string &nodeId);
  const Node &getNode(const std::string &nodeId) const;
  std::vector<Node> getNodes() const;
  const std::unordered_map<std::string, Node> &getNodesRef() const;

  // Edge operations
  void addEdge(const Edge &edge);
  void addGroupEdge(const std::string &id, const std::string &label,
                    const std::unordered_set<std::string> &nodeIds,
                    float weight,
                    const nlohmann::json &metadata = nlohmann::json::object());
  bool hasEdge(const std::string &edgeId) const;
  void removeEdge(const std::string &edgeId);
  const Edge &getEdge(const std::string &edgeId) const;
  std::vector<Edge> getEdges() const;
  std::vector<Edge> getGroupEdges() const;
  const std::unordered_map<std::string, Edge> &getEdgesRef() const;

  // Traversal
  std::vector<Node> getNeighbors(const std::string &nodeId) const;
  std::unordered_set<std::string>
  getGroupMembers(const std::string &groupId) const;
  nlohmann::json query(const std::string &nodeId, int maxDepth,
                       float minWeight) const;

  const std::unordered_map<std::string, std::vector<AdjacencyEntry>> &
  getAdjacencyList() const;

  // Metadata
  const nlohmann::json &getMetadata() const;
  void setMetadata(const nlohmann::json &metadata);
  void updateMetadata(const std::string &key, const nlohmann::json &value);

  // Serialization
  nlohmann::json toJson() const;
  static MemoryGraph fromJson(const nlohmann::json &graphJson);

private:
  std::unordered_map<std::string, Node> nodes_;
  std::unordered_map<std::string, Edge> edges_;
  nlohmann::json metadata_;
  GraphMode mode_;

  mutable std::unordered_map<std::string, std::vector<AdjacencyEntry>>
      adjacencyCache_;
  mutable bool adjacencyDirty_;

  void buildAdjacencyCache() const;
  void invalidateCache();
};

} // namespace memory_graph
