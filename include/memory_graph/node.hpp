#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace memory_graph {

/**
 * @class Node
 * @brief Represents an entity in the memory graph (e.g., a concept, person, or
 * event).
 */

class Node {
public:
  /**
   * @brief Construct a new Node.
   * @param id Unique identifier for the node.
   * @param label Human-readable name for the node.
   * @param metadata Additional context (e.g., type, timestamp).
   */
  Node(const std::string &id, const std::string &label,
       const nlohmann::json &metadata = {});

  // Getters
  const std::string &getId() const;
  const std::string &getLabel() const;
  const std::vector<std::string> &getConnections() const;
  const nlohmann::json &getMetadata() const;

  // Setters
  void setLabel(const std::string &label);
  void addConnection(const std::string &nodeId);
  void removeConnection(const std::string &nodeId);
  void setMetadata(const nlohmann::json &metadata);
  void updateMetadata(const std::string &key, const nlohmann::json &value);

  // Serialization
  nlohmann::json toJson() const;
  static Node fromJson(const nlohmann::json &nodeJson);

private:
  std::string id_;
  std::string label_;
  std::vector<std::string> connections_;
  nlohmann::json metadata_;
};

} // namespace memory_graph
