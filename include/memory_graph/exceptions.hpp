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
      : std::runtime_error("Node with ID '" + nodeId + "' not found"),
        nodeId_(nodeId) {}
  const std::string &nodeId() const { return nodeId_; }

private:
  std::string nodeId_;
};

/**
 * @class EdgeNotFoundError
 * @brief Throw when an edge is not found in the graph.
 */
class EdgeNotFoundError : public std::runtime_error {
public:
  explicit EdgeNotFoundError(const std::string &edgeId)
      : std::runtime_error("Edge with ID '" + edgeId + "' not found"),
        edgeId_(edgeId) {}
  const std::string &edgeId() const { return edgeId_; }

private:
  std::string edgeId_;
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
  struct RawMessageTag {};
  explicit InvalidConnectionError(const std::string &nodeId)
      : std::runtime_error("Node '" + nodeId + "' does not exist in the graph"),
        nodeId_(nodeId) {}
  InvalidConnectionError(RawMessageTag, const std::string &msg)
      : std::runtime_error(msg) {}
  const std::string &nodeId() const { return nodeId_; }

private:
  std::string nodeId_;
};

class ModeError : public std::runtime_error {
public:
  explicit ModeError(const std::string &operation,
                     const std::string &currentMode)
      : std::runtime_error("Operation '" + operation + "' is not allowed in " +
                           currentMode + " mode"),
        operation_(operation), currentMode_(currentMode) {}
  const std::string &operation() const { return operation_; }
  const std::string &currentMode() const { return currentMode_; }

private:
  std::string operation_;
  std::string currentMode_;
};

} // namespace memory_graph
