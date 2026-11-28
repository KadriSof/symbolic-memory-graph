#include "memory_graph/memory_graph.hpp"
#include "memory_graph/edge.hpp"
#include "memory_graph/exceptions.hpp"
#include "memory_graph/hebbian.hpp"
#include "memory_graph/node.hpp"
#include "memory_graph/utils/traversal.hpp"
#include "nlohmann/json.hpp"
#include <cstddef>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace memory_graph {

MemoryGraph::MemoryGraph(const nlohmann::json &metadata)
    : metadata_(metadata.is_object() ? metadata : nlohmann::json::object()),
      mode_(GraphMode::SYMBOLIC), // Default to SYMBOLIC for agentic use
      adjacencyDirty_(true) {}

GraphMode MemoryGraph::getMode() const { return mode_; }
void MemoryGraph::setMode(GraphMode mode) { mode_ = mode; }

void MemoryGraph::addNode(const Node &node) {
  if (nodes_.find(node.getId()) != nodes_.end()) {
    throw DuplicateIdError("Node with ID '" + node.getId() +
                           "' already exists.");
  }
  nodes_.emplace(node.getId(), node);
  invalidateCache();
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
    edges_.erase(edgeId);
  }

  nodes_.erase(nodeId);
  invalidateCache();
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

const std::unordered_map<std::string, Node> &MemoryGraph::getNodesRef() const {
  return nodes_;
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

    // Handle asymetric connections
  } else {
    const auto &conn_pair =
        std::get<AsymmetricConnections>(edge.getConnections());

    // Validate: both nodes must exist
    if (!hasNode(conn_pair.first) || !hasNode(conn_pair.second)) {
      throw InvalidConnectionError(
          !hasNode(conn_pair.first) ? conn_pair.first : conn_pair.second);
    }
  }

  edges_.emplace(edge.getId(), edge);
  invalidateCache();
}

void MemoryGraph::addGroupEdge(const std::string &id, const std::string &label,
                               const std::unordered_set<std::string> &nodeIds,
                               float weight, const nlohmann::json &metadata) {
  // 1. Validate edge ID
  if (edges_.find(id) != edges_.end()) {
    throw DuplicateIdError("[MemoryGraph:addGroupEdge] Edge with ID '" + id +
                           "' already exists");
  }

  // 2. Validate group size
  if (nodeIds.size() < 2) {
    throw InvalidConnectionError(InvalidConnectionError::RawMessageTag{},
                                 "[MemoryGraph:addGroupEdge] Group edge "
                                 "requires at least 2 nodes, got " +
                                     std::to_string(nodeIds.size()));
  }

  // 3. Validate all nodes exists
  for (const auto &nodeId : nodeIds) {
    if (!hasNode(nodeId)) {
      throw InvalidConnectionError(nodeId);
    }
  }

  // 4. Check for duplicate group
  for (const auto &[existingId, existingEdge] : edges_) {
    if (existingEdge.getType() == EdgeType::SYMMETRIC) {
      const auto &existingSet =
          std::get<SymmetricConnections>(existingEdge.getConnections());
      if (existingSet == nodeIds && existingEdge.getLabel() == label) {
        throw DuplicateIdError("[MemoryGraph:addGroupEdge] Group with same "
                               "nodes and label already exist: '" +
                               label + "'");
      }
    }
  }

  // 5. Validate weight
  if (weight < 0.0f || weight > 1.0f) {
    throw std::invalid_argument("[MemoryGraph:addGroupEdge] Group edge weight "
                                "must be between 0.0 and 1.0");
  }

  // 5. Create the group edge (symmetric by definition)
  SymmetricConnections conn(nodeIds);
  Edge groupEdge(id, label, EdgeType::SYMMETRIC, conn, weight, metadata, true);

  // 6. Store the edge
  edges_.emplace(id, groupEdge);
  invalidateCache();
}

bool MemoryGraph::hasEdge(const std::string &edgeId) const {
  return edges_.find(edgeId) != edges_.end();
}

void MemoryGraph::removeEdge(const std::string &edgeId) {
  if (!hasEdge(edgeId)) {
    throw EdgeNotFoundError(edgeId);
  }

  edges_.erase(edgeId);
  invalidateCache();
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

std::optional<std::string>
MemoryGraph::findEdgeBetween(const std::string &a, const std::string &b) const {
  const auto &adjList = getAdjacencyList(); // builds cache for us if dirty
  auto it = adjList.find(a);
  if (it == adjList.end()) {
    return std::nullopt;
  }

  for (const auto &entry : it->second) {
    if (entry.neighbor_id == b) {
      return entry.edge_id;
    }
  }

  return std::nullopt;
}

std::vector<Edge> MemoryGraph::getGroupEdges() const {
  std::vector<Edge> result;
  for (const auto &[id, edge] : edges_) {
    if (edge.isGroupEdge()) {
      result.push_back(edge);
    }
  }

  return result;
}

const std::unordered_map<std::string, Edge> &MemoryGraph::getEdgesRef() const {
  return edges_;
}

std::vector<Node> MemoryGraph::getNeighbors(const std::string &nodeId) const {
  if (!hasNode(nodeId)) {
    throw NodeNotFoundError(nodeId);
  }

  const auto &adjList = getAdjacencyList();
  std::vector<Node> neighbors;
  std::unordered_set<std::string> seen;

  auto it = adjList.find(nodeId);
  if (it != adjList.end()) {
    for (const auto &entry : it->second) {
      if (seen.find(entry.neighbor_id) == seen.end()) {
        seen.insert(entry.neighbor_id);
        neighbors.push_back(nodes_.at(entry.neighbor_id));
      }
    }
  }

  return neighbors;
}

std::unordered_set<std::string>
MemoryGraph::getGroupMembers(const std::string &groupId) const {
  if (!hasEdge(groupId)) {
    throw EdgeNotFoundError("[MemoryGraph:getGroupMembers] Edge with ID '" +
                            groupId + "' not found");
  }

  const Edge &edge = getEdge(groupId);

  if (!edge.isGroupEdge()) {
    throw std::invalid_argument("[MemoryGraph:getGroupMembers] Edge '" +
                                groupId + "' is not a group edge");
  }

  const auto &conn_set = std::get<SymmetricConnections>(edge.getConnections());

  return conn_set;
}

nlohmann::json MemoryGraph::query(const std::string &nodeId, int maxDepth,
                                  float minWeight) const {
  if (!hasNode(nodeId)) {
    throw NodeNotFoundError(nodeId);
  }

  const auto &adjList = getAdjacencyList();
  nlohmann::json result;
  result["nodes"] = nlohmann::json::array();
  result["edges"] = nlohmann::json::array();

  std::unordered_set<std::string> visitedNodes;
  std::unordered_set<std::string> visitedEdges;

  // Mark as visited when adding to queue
  std::queue<std::pair<std::string, int>> queue;
  queue.push({nodeId, 0});
  visitedNodes.insert(nodeId);

  // Add start node with depth 0
  const Node &startNode = nodes_.at(nodeId);
  result["nodes"].push_back({{"id", startNode.getId()},
                             {"label", startNode.getLabel()},
                             {"metadata", startNode.getMetadata()},
                             {"depth", 0}});

  while (!queue.empty()) {
    auto [currentId, depth] = queue.front();
    queue.pop();

    if (depth >= maxDepth) {
      continue;
    }

    auto it = adjList.find(currentId);
    if (it == adjList.end()) {
      continue;
    }

    // Explore neighbors (edge_id from adjacency entry for O(1) weight lookup)
    for (const auto &entry : it->second) {
      const std::string &neighborId = entry.neighbor_id;
      const std::string &edgeId = entry.edge_id;

      // Skipp if edge already visited
      if (visitedEdges.find(edgeId) != visitedEdges.end()) {
        continue;
      }

      const Edge &edge = edges_.at(edgeId);
      float edgeWeight = edge.getWeight();

      if (edgeWeight >= minWeight) {
        visitedEdges.insert(edgeId);

        // Add edge to result
        nlohmann::json edgeJson = {{"id", edge.getId()},
                                   {"label", edge.getLabel()},
                                   {"weight", edge.getWeight()},
                                   {"source", currentId},
                                   {"target", neighborId}};
        result["edges"].push_back(edgeJson);

        // Mark neighbor as visited before adding to queue
        if (visitedNodes.find(neighborId) == visitedNodes.end()) {
          visitedNodes.insert(neighborId);

          const Node &neighborNode = nodes_.at(neighborId);
          result["nodes"].push_back({{"id", neighborNode.getId()},
                                     {"label", neighborNode.getLabel()},
                                     {"metadata", neighborNode.getMetadata()},
                                     {"depth", depth + 1}});

          queue.push({neighborId, depth + 1});
        }
      }
    }
  }

  return result;
}

void MemoryGraph::buildAdjacencyCache() const {
  // Clear existing cache
  adjacencyCache_.clear();

  // Build adjacency list from all edges
  for (const auto &[edgeId, edge] : edges_) {
    if (edge.getType() == EdgeType::SYMMETRIC) {
      // Bidirectional connections between all pairs
      const auto &conn_set =
          std::get<SymmetricConnections>(edge.getConnections());
      // Convert set to vector for ordered pair generation
      std::vector<std::string> nodes(conn_set.begin(), conn_set.end());

      // Create bidirectional connections between every pair in the set
      for (size_t i = 0; i < nodes.size(); ++i) {
        for (size_t j = i + 1; j < nodes.size(); ++j) {
          // Add both directions with isSymmetric = true
          adjacencyCache_[nodes[i]].push_back({nodes[j], true, edgeId});
          adjacencyCache_[nodes[j]].push_back({nodes[i], true, edgeId});
        }
      }
    } else {
      // Directed connection
      const auto &conn_pair =
          std::get<AsymmetricConnections>(edge.getConnections());
      // Add one direction with isSymmetric = false
      adjacencyCache_[conn_pair.first].push_back(
          {conn_pair.second, false, edgeId});
      adjacencyCache_[conn_pair.second].push_back(
          {conn_pair.first, false, edgeId});
    }
  }

  // Ensure all nodes exist in adjacency list (even isolated ones)
  for (const auto &[nodeId, node] : nodes_) {
    if (adjacencyCache_.find(nodeId) == adjacencyCache_.end()) {
      adjacencyCache_[nodeId] = {}; // Empty adjacency list
    }
  }

  // Mark cache as clear
  adjacencyDirty_ = false;
}

const std::unordered_map<std::string, std::vector<AdjacencyEntry>> &
MemoryGraph::getAdjacencyList() const {
  if (adjacencyDirty_) {
    buildAdjacencyCache();
  }

  return adjacencyCache_;
}

void MemoryGraph::invalidateCache() { adjacencyDirty_ = true; }

nlohmann::json MemoryGraph::toJson() const {
  nlohmann::json graphJson;
  graphJson["metadata"] = metadata_;
  graphJson["mode"] =
      (mode_ == GraphMode::KNOWLEDGE) ? "KNOWLEDGE" : "SYMBOLIC";

  nlohmann::json nodesJson;
  for (const auto &[nodeId, node] : nodes_) {
    nodesJson[nodeId] = node.toJson();
  }
  graphJson["nodes"] = nodesJson;

  nlohmann::json edgesJson;
  for (const auto &[edgeId, edge] : edges_) {
    edgesJson[edgeId] = edge.toJson();
  }
  graphJson["edges"] = edgesJson;

  return graphJson;
}

MemoryGraph MemoryGraph::fromJson(const nlohmann::json &graphJson) {
  MemoryGraph graph(graphJson.value("metadata", nlohmann::json::object()));

  std::string modeStr = graphJson.value("mode", "SYMBOLIC");
  graph.mode_ =
      (modeStr == "KNOWLEDGE") ? GraphMode::KNOWLEDGE : GraphMode::SYMBOLIC;

  if (graphJson.contains("nodes") && graphJson["nodes"].is_object()) {
    for (const auto &[nodeId, nodesJson] : graphJson["nodes"].items()) {
      try {
        graph.addNode(Node::fromJson(nodesJson));
      } catch (const std::exception &e) {
        std::cerr << "Failed to deserialize node '" << nodeId
                  << "': " << e.what() << std::endl;
      }
    }
  }

  if (graphJson.contains("edges") && graphJson["edges"].is_object()) {
    for (const auto &[edgeId, edgeJson] : graphJson["edges"].items()) {
      try {
        Edge edge = Edge::fromJson(edgeJson);
        if (edge.isGroupEdge()) {
          const auto &conn =
              std::get<SymmetricConnections>(edge.getConnections());
          std::unordered_set<std::string> nodeIds(conn.begin(), conn.end());
          graph.addGroupEdge(edge.getId(), edge.getLabel(), nodeIds,
                             edge.getWeight(), edge.getMetadata());
        } else {
          graph.addEdge(edge);
        }
      } catch (const std::exception &e) {
        std::cerr << "Failed to deserialize edge '" << edgeId
                  << "': " << e.what() << std::endl;
      }
    }
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

void MemoryGraph::attachHebbian(const HebbianConfig &config) {
  if (mode_ == GraphMode::KNOWLEDGE) {
    throw ModeError("attachHebbian", "KNOWLEDGE");
  }
  hebbian_ = std::make_unique<HebbianEngine>(*this, config);
}

void MemoryGraph::detachHebbian() { hebbian_.reset(); }

bool MemoryGraph::hasHebbian() const { return hebbian_ != nullptr; }

HebbianEngine &MemoryGraph::hebbian() {
  if (!hebbian_) {
    throw std::runtime_error(
        "HebbianEngine not attached. Call attachHebbian() first.");
  }

  return *hebbian_;
}

void MemoryGraph::learn(const std::vector<std::string> &firedNodes) {
  if (!hebbian_) {
    throw std::runtime_error(
        "HebbianEngine not attached. Call attachHebbian() first.");
  }
  hebbian_->fireAll(firedNodes);
  hebbian_->potentiate();
}

ActivationResult MemoryGraph::activate(const std::string &seed, size_t k,
                                       int maxDepth) {
  if (hebbian_) {
    return hebbian_->activate({seed}, k, maxDepth);
  }

  // Fallback to BFS for KNOWLEDGE MODE
  ActivationResult result;
  result.tick = 0;
  auto nodeIds = utils::bfs(*this, seed, maxDepth);
  for (const auto &id : nodeIds) {
    if (id != seed) {
      result.frontier.emplace_back(id, 1.0f);
    }
  }

  return result;
}

bool MemoryGraph::isAdjacencyCacheValid() const { return !adjacencyDirty_; }

} // namespace memory_graph
