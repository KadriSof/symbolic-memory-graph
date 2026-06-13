#include "memory_graph/edge.hpp"
#include "nlohmann/json.hpp"
#include <vector>

namespace memory_graph {

Edge::Edge(const std::string &id, const std::string &label, EdgeType type,
           const Connections &connections, float weight,
           const nlohmann::json &metadata)
    : id_(id), label_(label), type_(type), connections_(connections),
      weight_(weight), metadata_(metadata) {
  validateWeight(weight);
}

const std::string &Edge::getId() const { return id_; }
const std::string &Edge::getLabel() const { return label_; }
EdgeType Edge::getType() const { return type_; }
const Connections &Edge::getConnections() const { return connections_; }
float Edge::getWeight() const { return weight_; }
const nlohmann::json &Edge::getMetadata() const { return metadata_; }

void Edge::setLabel(const std::string &label) { label_ = label; }

void Edge::setWeight(float weight) {
  validateWeight(weight);
  weight_ = weight;
}

void Edge::setMetadata(const nlohmann::json &metadata) { metadata_ = metadata; }

void Edge::updateMetadata(const std::string &key, const nlohmann::json &value) {
  metadata_[key] = value;
}

void Edge::validateWeight(float weight) {
  if (weight < 0.0f || weight > 1.0f) {
    throw std::invalid_argument("Edge weight must be between 0.0 and 1.0.");
  }
}

nlohmann::json Edge::toJson() const {
  nlohmann::json edgeJson;
  edgeJson["id"] = id_;
  edgeJson["label"] = label_;
  edgeJson["type"] =
      (type_ == EdgeType::SYMMETRIC) ? "SYMMETRIC" : "ASYMMETRIC";
  edgeJson["weight"] = weight_;
  edgeJson["metadata"] = metadata_;

  // Serialize connections
  if (std::holds_alternative<SymmetricConnections>(connections_)) {
    const auto &conn_set = std::get<SymmetricConnections>(connections_);
    edgeJson["connections"] = nlohmann::json{
        {"type", "symmetric"}, {"nodes", nlohmann::json::array()}};
    for (const auto &nodeId : conn_set) {
      edgeJson["connections"]["nodes"].push_back(nodeId);
    }
  } else {
    const auto &conn_pair = std::get<AsymmetricConnections>(connections_);
    edgeJson["connections"] = nlohmann::json{{"type", "asymmetric"},
                                             {"source", conn_pair.first},
                                             {"target", conn_pair.second}};
  }

  return edgeJson;
}

Edge Edge::fromJson(const nlohmann::json &edgeJson) {
  EdgeType type = (edgeJson["type"] == "SYMMETRIC") ? EdgeType::SYMMETRIC
                                                    : EdgeType::ASYMMETRIC;
  float weight = edgeJson.value("weight", 1.0f);
  nlohmann::json metadata =
      edgeJson.value("metadata", nlohmann::json::object());

  Connections connections;
  if (edgeJson["connections"]["type"] == "symmetric") {
    auto nodes =
        edgeJson["connections"]["nodes"].get<std::vector<std::string>>();
    SymmetricConnections conn_set(nodes.begin(), nodes.end());
    connections = conn_set;
  } else {
    std::string source = edgeJson["connections"]["source"];
    std::string target = edgeJson["connections"]["target"];
    connections = AsymmetricConnections(source, target);
  }

  return Edge(edgeJson.at("id").get<std::string>(),
              edgeJson.at("label").get<std::string>(), type, connections,
              weight, metadata);
}
} // namespace memory_graph
