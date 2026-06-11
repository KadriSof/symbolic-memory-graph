#include "memory_graph/memory_graph.hpp"
#include "memory_graph/edge.hpp"
#include "memory_graph/exceptions.hpp"
#include "nlohmann/json.hpp"
#include <string>
#include <unordered_set>

namespace memory_graph {

MemoryGraph::MemoryGraph(const nlohmann::json &metadata)
    : metadata_(metadata) {}

void MemoryGraph::addNode(const Node &node) {
  if (nodes_.find(node.getId()) != nodes_.end()) {
    throw DuplicateIdError("Node with ID '" + node.getId() +
                           "' already exists.");
  }
  nodes_[node.getId()] = node;
}

bool MemoryGraph::hasNode(const std::string &nodeId) const {
  return nodes_.find(nodeId) != nodes_.end();
}

void MemoryGraph::removeNode(const std::string &nodeId) {
  if (!hasNode(nodeId)) {
    throw NodeNotFoundError(nodeId);
  }

  // Remove all edges connected to this node
  std::vector<std::string> edgesToRemove;
  for (const auto &[edgeId, edge] : edges_) {
    if (std::holds_alternative<SymmetricConnections>(edge.getConnections())) {
      const auto &conn_set =
          std::get<SymmetricConnections>(edge.getConnections());
      if (conn_set.find(nodeId) != conn_set.end()) {
        edgesToRemove.push_back(edgeId);
      }
    } else {
      const auto &conn_pair =
          std::get<AsymmetricConnections>(edge.getConnections());
      if (conn_pair.first == nodeId || conn_pair.second == nodeId) {
        edgesToRemove.push_back(edgeId);
      }
    }
  }

  for (const auto &edgeId : edgesToRemove) {
    removeEdge(edgeId);
  }

  nodes_.erase(nodeId);
}

const Node &MemoryGraph::getNode(const std::string &nodeId) const {
  if (!hasNode(nodeId)) {
    throw NodeNotFoundError(nodeId);
  }
  return nodes_.at(nodeId);
}

std::vector<Node> MemoryGraph::getNodes() const {
  std::vector<Node> result;
  for (const auto &[id, node] : nodes_) {
    result.push_back(node);
  }
  return result;
}

void MemoryGraph::addEdge(const Edge &edge) {
  if (edges_.find(edge.getId()) != edges_.end()) {
    throw DuplicateIdError("Edge with ID '" + edge.getId() +
                           "' already exists.");
  }

  // Validate connections
  if (std::holds_alternative<SymmetricConnections>(edge.getConnections())) {
    const auto &conn_set =
        std::get<SymmetricConnections>(edge.getConnections());

    // Validate: must have exactly 2 nodes
    if (conn_set.size() != 2) {
      throw InvalidConnectionError(
          "Symmetric connection requires exactly 2 nodes, got " +
          std::to_string(conn_set.size()));
    }

    // Validate: all nodes must exist
    for (const auto &nodeId : conn_set) {
      if (!hasNode(nodeId)) {
        throw InvalidConnectionError(nodeId);
      }
    }

    // Update: add bidirectional connections
    auto it = conn_set.begin();
    std::string node1 = *it;
    std::string node2 = *(++it);
    nodes_.at(node1).addConnection(node2);
    nodes_.at(node2).addConnection(node1);

    // Handle asymetric connections
  } else {
    const auto &conn_pair =
        std::get<AsymmetricConnections>(edge.getConnections());

    // Validate: both nodes must exist
    if (!hasNode(conn_pair.first) || !hasNode(conn_pair.second)) {
      throw InvalidConnectionError(
          !hasNode(conn_pair.first) ? conn_pair.first : conn_pair.second);
    }

    // Update: add unidirectional connection
    nodes_.at(conn_pair.first).addConnection(conn_pair.second);
  }

  edges_[edge.getId()] = edge;
}

} // namespace memory_graph
