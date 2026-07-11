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
      : std::runtime_error("Node with ID '" + nodeId + "' not found.") {}
};

/**
 * @class EdgeNotFoundError
 * @brief Throw when an edge is not found in the graph.
 */
class EdgeNotFoundError : public std::runtime_error {
public:
  explicit EdgeNotFoundError(const std::string &edgeId)
      : std::runtime_error("Edge with ID '" + edgeId + "' not found.") {}
};

/**
 * @class InvalidConnectionError
 * @brief Thrown when a connection/edge definition is invalid.
 *
 * This exception has two constructors:
 * - For missing node IDs: InvalidConnectionError(nodeId)
 * - For custom validation messages: InvalidConnectionError(RawMessageTag,
 * message)
 */
class InvalidConnectionError : public std::runtime_error {
public:
  /**
   * @brief Constructor for missing node ID
   * @param nodeId The ID of the node that does not exist
   */
  explicit InvalidConnectionError(const std::string &nodeId)
      : std::runtime_error("Node with ID '" + nodeId + "' does not exist.") {}

  /**
   * @brief Tag type to distinguish the raw message constructor
   */
  struct RawMessageTag {};

  /**
   * @brief Constructor for raw error messages
   * @param tag Tag to distinguish from the node ID constructor
   * @param message The raw error message
   */
  InvalidConnectionError(RawMessageTag, const std::string &message)
      : std::runtime_error(message) {}
};

} // namespace memory_graph
