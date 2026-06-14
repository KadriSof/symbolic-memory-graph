#pragma once

#include "memory_graph/exceptions.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_set>
#include <utility>
#include <variant>

namespace memory_graph {
/**
 * @enum EdgeType
 * @brief Directionality of the edge (symmetric or symmetric).
 */
enum class EdgeType { SYMMETRIC, ASYMMETRIC };

// Type aliases for connections
using SymmetricConnections = std::unordered_set<std::string>;
using AsymmetricConnections = std::pair<std::string, std::string>;
using Connections = std::variant<SymmetricConnections, AsymmetricConnections>;

/**
 * @class Edge
 * @brief Represents a relationship between nodes in the memory graph.
 */
class Edge {
public:
  /**
   * @brief Construct a new Edge.
   * @param id Unique identifier for the edge.
   * @param label Human-readable name for the edge (e.g., "inspired_by")
   * @param type Directionality (SYMMETRIC or ASYMMETRIC).
   * @param connections Nodes connected by this edge (pair of asymmetric, set
   * for symmetric).
   * @param weight Strength of the relationship (0.0 to 1.0).
   * @param metadata Additional context (e.g., timestamp, location).
   */
  Edge(const std::string &id, const std::string &label, EdgeType type,
       const Connections &connections, float weight = 1.0f,
       const nlohmann::json &metadata = {});

  // Getters
  const std::string &getId() const;
  const std::string &getLabel() const;
  EdgeType getType() const;
  const Connections &getConnections() const;
  float getWeight() const;
  const nlohmann::json &getMetadata() const;

  // Setters
  void setLabel(const std::string &label);
  void setWeight(float weight);
  void setMetadata(const nlohmann::json &metadata);
  void updateMetadata(const std::string &key, const nlohmann::json &value);

  // Serialization
  nlohmann::json toJson() const;
  static Edge fromJson(const nlohmann::json &edgeJson);

private:
  std::string id_;
  std::string label_;
  EdgeType type_;
  Connections connections_;
  float weight_;
  nlohmann::json metadata_;

  // Helper to validate weight
  void validateWeight(float weight);
};

} // namespace memory_graph
