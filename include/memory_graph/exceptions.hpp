#pragma once

#include <stdexcept>
#include <string>

namespace memory_graph {
/**
 * @class DuplicateIdError
 * @brief Thrown when a duplicate node/edge ID is detected.
 */
class DuplicateIdError : public std::runtime_error {
public:
  explicit DuplicateIdError(const std::string &message)
      : std::runtime_error(message) {}
};

/**
 * @class NodeNotFoundError
 * @brief Thrown when a node is not found in the graph.
 */
class NodeNotFoundError : public std::runtime_error {
public:
  explicit NodeNotFoundError(const std::string &nodeId)
      : std::runtime_error("Node with ID '" + nodeId + "'not found.") {}
};

/**
 * @class NodeNotFoundError
 * @brief Throw when an edge is not found in the graph.
 */
class EdgeNotFoundError : public std::runtime_error {
public:
  explicit EdgeNotFoundError(const std::string &edgeId)
      : std::runtime_error("Edge with ID '" + edgeId + "' not found.") {}
};

/**
 * @class InvalidConnectionError
 * @brief Thrown when an edge references a non-existent node.
 */
class InvalidConnectionError : public std::runtime_error {
public:
  explicit InvalidConnectionError(const std::string &nodeId)
      : std::runtime_error("Node with ID '" + nodeId + "' does not exist.") {}
};

} // namespace memory_graph
