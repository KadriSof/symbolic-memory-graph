#include "memory_graph/memory_graph.hpp"
#include "memory_graph/edge.hpp"
#include "memory_graph/exceptions.hpp"
#include "nlohmann/json.hpp"
#include <queue>
#include <string>
#include <unordered_set>
#include <variant>

namespace memory_graph {

MemoryGraph::MemoryGraph(const nlohmann::json &metadata)
    : metadata_(metadata) {}

void MemoryGraph::addNode(const Node &node) {
  if (nodes_.find(node.getId()) != nodes_.end()) {
    throw DuplicateIdError("Node with ID '" + node.getId() +
                           "' already exists.");
  }
  nodes_.emplace(node.getId(), node);
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

  edges_.emplace(edge.getId(), edge);
}

bool MemoryGraph::hasEdge(const std::string &edgeId) const {
  return edges_.find(edgeId) != edges_.end();
}

void MemoryGraph::removeEdge(const std::string &edgeId) {
  if (!hasEdge(edgeId)) {
    throw EdgeNotFoundError(edgeId);
  }

  const Edge &edge = edges_.at(edgeId);
  if (edge.getType() == EdgeType::SYMMETRIC) {
    const auto &conn_set =
        std::get<SymmetricConnections>(edge.getConnections());
    auto it = conn_set.begin();
    std::string node1 = *it;
    std::string node2 = *(++it);
    nodes_.at(node1).removeConnection(node2);
    nodes_.at(node2).removeConnection(node1);
  } else {
    const auto &conn_pair =
        std::get<AsymmetricConnections>(edge.getConnections());
    nodes_.at(conn_pair.first).removeConnection(conn_pair.second);
  }

  edges_.erase(edgeId);
}

const Edge &MemoryGraph::getEdge(const std::string &edgeId) const {
  if (!hasEdge(edgeId)) {
    throw EdgeNotFoundError(edgeId);
  }
  return edges_.at(edgeId);
}

std::vector<Edge> MemoryGraph::getEdges() const {
  std::vector<Edge> result;
  for (const auto &[id, edge] : edges_) {
    result.push_back(edge);
  }
  return result;
}

std::vector<Node> MemoryGraph::getNeighbors(const std::string &nodeId) const {
  if (!hasNode(nodeId)) {
    throw NodeNotFoundError(nodeId);
  }

  std::vector<Node> neighbors;
  for (const auto &neighborId : nodes_.at(nodeId).getConnections()) {
    neighbors.push_back(nodes_.at(neighborId));
  }
  return neighbors;
}

nlohmann::json MemoryGraph::query(const std::string &nodeId, int maxDepth,
                                  float minWeight) const {
  if (!hasNode(nodeId)) {
    throw NodeNotFoundError(nodeId);
  }

  nlohmann::json result;
  std::unordered_set<std::string> visited;
  std::queue<std::pair<std::string, int>> queue; // (node_id, depth)

  queue.push({nodeId, 0});
  result["depth_0"] = nlohmann::json::array({nodeId});

  while (!queue.empty()) {
    auto [currentId, depth] = queue.front();
    queue.pop();

    if (depth >= maxDepth) {
      continue;
    }

    if (visited.find(currentId) != visited.end()) {
      continue;
    }

    visited.insert(currentId);

    // Explore neighbors
    for (const auto &neighborId : nodes_.at(currentId).getConnections()) {
      // Find the edge between currentId and neighborId
      float edgeWeight = 1.0f; // Default weight
      try {
        std::string edgeId = findEdgeId(currentId, neighborId);
        edgeWeight = edges_.at(edgeId).getWeight();
      } catch (const EdgeNotFoundError &) {
        // Won't probably catch this..
      }

      if (edgeWeight >= minWeight) {
        std::string depthKey = "depth_" + std::to_string(depth + 1);
        if (result.find(depthKey) == result.end()) {
          result[depthKey] = nlohmann::json::array();
        }
        result[depthKey].push_back(neighborId);
        queue.push({neighborId, depth + 1});
      }
    }
  }

  return result;
}

std::string MemoryGraph::findEdgeId(const std::string &nodeId1,
                                    const std::string &nodeId2) const {
  for (const auto &[edgeId, edge] : edges_) {
    if (std::holds_alternative<SymmetricConnections>(edge.getConnections())) {
      const auto &conn_set =
          std::get<SymmetricConnections>(edge.getConnections());
      if (conn_set.find(nodeId1) != conn_set.end() &&
          conn_set.find(nodeId2) != conn_set.end()) {
        return edgeId;
      }
    } else {
      const auto &conn_pair =
          std::get<AsymmetricConnections>(edge.getConnections());
      if ((conn_pair.first == nodeId1 && conn_pair.second == nodeId2) ||
          (conn_pair.first == nodeId2 && conn_pair.second == nodeId2)) {
        return edgeId;
      }
    }
  }

  throw EdgeNotFoundError("No edge found between '" + nodeId1 + "' and '" +
                          nodeId2 + "'.");
}

nlohmann::json MemoryGraph::toJson() const {
  nlohmann::json graphJson;
  graphJson["metadata"] = metadata_;

  // Serialize nodes
  nlohmann::json nodesJson;
  for (const auto &[nodeId, node] : nodes_) {
    nodesJson[nodeId] = node.toJson();
  }
  graphJson["nodes"] = nodesJson;

  // Serialize edges
  nlohmann::json edgesJson;
  for (const auto &[edgeId, edge] : edges_) {
    edgesJson[edgeId] = edge.toJson();
  }
  graphJson["edges"] = edgesJson;

  return graphJson;
}

MemoryGraph MemoryGraph::fromJson(const nlohmann::json &graphJson) {
  MemoryGraph graph(graphJson.value("metadata", nlohmann::json::object()));

  // Deserialize nodes
  for (const auto &[nodeId, nodesJson] : graphJson["nodes"].items()) {
    graph.addNode(Node::fromJson(nodesJson));
  }

  // Deserialize edges
  for (const auto &[edgeId, edgeJson] : graphJson["edges"].items()) {
    graph.addEdge(Edge::fromJson(edgeJson));
  }

  return graph;
}

const nlohmann::json &MemoryGraph::getMetadata() const { return metadata_; }

void MemoryGraph::setMetadata(const nlohmann::json &metadata) {
  metadata_ = metadata;
}

void MemoryGraph::updateMetadata(const std::string &key,
                                 const nlohmann::json &value) {
  metadata_[key] = value;
}

bool MemoryGraph::hasEdge(const std::string &edgeId) const {
  return edges_.find(edgeId) != edges_.end();
}

void MemoryGraph::removeEdge(const std::string &edgeId) {
  if (!hasEdge(edgeId)) {
    throw EdgeNotFoundError(edgeId);
  }

  const Edge &edge = edges_.at(edgeId);
  if (edge.getType() == EdgeType::SYMMETRIC) {
    const auto &conn_set =
        std::get<SymmetricConnections>(edge.getConnections());
    auto it = conn_set.begin();
    std::string node1 = *it;
    std::string node2 = *(++it);
    nodes_.at(node1).removeConnection(node2);
    nodes_.at(node2).removeConnection(node1);
  } else {
    const auto &conn_pair =
        std::get<AsymmetricConnections>(edge.getConnections());
    nodes_.at(conn_pair.first).removeConnection(conn_pair.second);
  }

  edges_.erase(edgeId);
}

const Edge &MemoryGraph::getEdge(const std::string &edgeId) const {
  if (!hasEdge(edgeId)) {
    throw EdgeNotFoundError(edgeId);
  }
  return edges_.at(edgeId);
}

std::vector<Edge> MemoryGraph::getEdges() const {
  std::vector<Edge> result;
  for (const auto &[id, edge] : edges_) {
    result.push_back(edge);
  }
  return result;
}

std::vector<Node> MemoryGraph::getNeighbors(const std::string &nodeId) const {
  if (!hasNode(nodeId)) {
    throw NodeNotFoundError(nodeId);
  }

  std::vector<Node> neighbors;
  for (const auto &neighborId : nodes_.at(nodeId).getConnections()) {
    neighbors.push_back(nodes_.at(neighborId));
  }
  return neighbors;
}

nlohmann::json MemoryGraph::query(const std::string &nodeId, int maxDepth,
                                  float minWeight) const {
  if (!hasNode(nodeId)) {
    throw NodeNotFoundError(nodeId);
  }

  nlohmann::json result;
  std::unordered_set<std::string> visited;
  std::queue<std::pair<std::string, int>> queue; // (node_id, depth)

  queue.push({nodeId, 0});
  result["depth_0"] = nlohmann::json::array({nodeId});

  while (!queue.empty()) {
    auto [currentId, depth] = queue.front();
    queue.pop();

    if (depth >= maxDepth) {
      continue;
    }

    if (visited.find(currentId) != visited.end()) {
      continue;
    }

    visited.insert(currentId);

    // Explore neighbors
    for (const auto &neighborId : nodes_.at(currentId).getConnections()) {
      // Find the edge between currentId and neighborId
      float edgeWeight = 1.0f; // Default weight
      try {
        std::string edgeId = findEdgeId(currentId, neighborId);
        edgeWeight = edges_.at(edgeId).getWeight();
      } catch (const EdgeNotFoundError &) {
        // Won't probably catch this..
      }

      if (edgeWeight >= minWeight) {
        std::string depthKey = "depth_" + std::to_string(depth + 1);
        if (result.find(depthKey) == result.end()) {
          result[depthKey] = nlohmann::json::array();
        }
        result[depthKey].push_back(neighborId);
        queue.push({neighborId, depth + 1});
      }
    }
  }

  return result;
}

std::string MemoryGraph::findEdgeId(const std::string &nodeId1,
                                    const std::string &nodeId2) const {
  for (const auto &[edgeId, edge] : edges_) {
    if (std::holds_alternative<SymmetricConnections>(edge.getConnections())) {
      const auto &conn_set =
          std::get<SymmetricConnections>(edge.getConnections());
      if (conn_set.find(nodeId1) != conn_set.end() &&
          conn_set.find(nodeId2) != conn_set.end()) {
        return edgeId;
      }
    } else {
      const auto &conn_pair =
          std::get<AsymmetricConnections>(edge.getConnections());
      if ((conn_pair.first == nodeId1 && conn_pair.second == nodeId2) ||
          (conn_pair.first == nodeId2 && conn_pair.second == nodeId2)) {
        return edgeId;
      }
    }
  }

  throw EdgeNotFoundError("No edge found between '" + nodeId1 + "' and '" +
                          nodeId2 + "'.");
}

nlohmann::json MemoryGraph::toJson() const {
  nlohmann::json graphJson;
  graphJson["metadata"] = metadata_;

  // Serialize nodes
  nlohmann::json nodesJson;
  for (const auto &[nodeId, node] : nodes_) {
    nodesJson[nodeId] = node.toJson();
  }
  graphJson["nodes"] = nodesJson;

  // Serialize edges
  nlohmann::json edgesJson;
  for (const auto &[edgeId, edge] : edges_) {
    edgesJson[edgeId] = edge.toJson();
  }
  graphJson["edges"] = edgesJson;

  return graphJson;
}

MemoryGraph MemoryGraph::fromJson(const nlohmann::json &graphJson) {
  MemoryGraph graph(graphJson.value("metadata", nlohmann::json::object()));

  // Deserialize nodes
  for (const auto &[nodeId, nodesJson] : graphJson["nodes"].items()) {
    graph.addNode(Node::fromJson(nodesJson));
  }

  // Deserialize edges
  for (const auto &[edgeId, edgeJson] : graphJson["edges"].items()) {
    graph.addEdge(Edge::fromJson(edgeJson));
  }

  return graph;
}

const nlohmann::json &MemoryGraph::getMetadata() const { return metadata_; }

void MemoryGraph::setMetadata(const nlohmann::json &metadata) {
  metadata_ = metadata;
}

void MemoryGraph::updateMetadata(const std::string &key,
                                 const nlohmann::json &value) {
  metadata_[key] = value;
}

} // namespace memory_graph
