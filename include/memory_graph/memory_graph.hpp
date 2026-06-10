#pragma once

#include "memory_graph/edge.hpp"
#include "memory_graph/node.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
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
  void addNote(const Node &node);
  void removeNode(const std::string &nodeId);
  const Node &getNode(const std::string &nodeId) const;
  bool hasNode(const std::string &nodeId) const;
  std::vector<Node> getNodes() const;

  // Edge operations
  void addEdge(const Edge &edge);
  void removeEdge(const std::string &edgeId);
  const Edge &getEdge(const std::string &edgeId) const;
  bool hasEdge(const std::string &edgeId) const;
  std::vector<Edge> getEdges() const;

  // Graph operations
  std::vector<Node> getNeighbors(const std::string &nodeId) const;
  nlohmann::json query(const std::string &nodeId, int maxDepth = 1,
                       float minWeight = 0.0f) const;

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
};

} // namespace memory_graph
