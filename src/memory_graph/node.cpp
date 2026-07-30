#include "memory_graph/node.hpp"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <string>
#include <vector>

namespace memory_graph {

Node::Node(const std::string &id, const std::string &label,
           const nlohmann::json &metadata)
    : id_(id), label_(label), metadata_(metadata) {}

const std::string &Node::getId() const { return id_; }
const std::string &Node::getLabel() const { return label_; }
const std::vector<std::string> &Node::getConnections() const {
  return connections_;
}
const nlohmann::json &Node::getMetadata() const { return metadata_; }

void Node::setLabel(const std::string &label) { label_ = label; }
void Node::setMetadata(const nlohmann::json &metadata) { metadata_ = metadata; }
void Node::updateMetadata(const std::string &key, const nlohmann::json &value) {
  metadata_[key] = value;
}

void Node::addConnection(const std::string &nodeId) {
  // DEPRECATED: USED ONLY FOR DESERIALIZATION!
  if (std::find(connections_.begin(), connections_.end(), nodeId) ==
      connections_.end()) {
    connections_.push_back(nodeId);
  }
}

void Node::removeConnection(const std::string &nodeId) {
  // DEPRECATED: USED ONLY FOR DESERIALIZATION!
  connections_.erase(
      std::remove(connections_.begin(), connections_.end(), nodeId),
      connections_.end());
}

nlohmann::json Node::toJson() const {
  nlohmann::json nodeJson;
  nodeJson["id"] = id_;
  nodeJson["label"] = label_;
  nodeJson["connections"] = connections_;
  nodeJson["metadata"] = metadata_;
  return nodeJson;
}

Node Node::fromJson(const nlohmann::json &nodeJson) {
  std::string id = nodeJson.at("id").get<std::string>();
  std::string label = nodeJson.at("label").get<std::string>();

  nlohmann::json metadata;
  if (nodeJson.contains("metadata") && !nodeJson["metadata"].is_null()) {
    metadata = nodeJson["metadata"];
  } else {
    metadata = nlohmann::json::object();
  }

  Node node(id, label, metadata);

  // Parse connections...
  if (nodeJson.contains("connections") && nodeJson["connections"].is_array()) {
    for (const auto &connectionId : nodeJson["connections"]) {
      node.addConnection(connectionId.get<std::string>());
    }
  }

  return node;
}
} // namespace memory_graph
