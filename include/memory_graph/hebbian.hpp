#pragma once

#include "memory_graph/node.hpp"
#include <cstddef>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace memory_graph {

class MemoryGraph;

// Configuration for Hebbian learning
struct HebbianConfig {
  // Learning rate for potentiation
  float learning_rate = 0.1f;
  // Multiplicative decay per tick (lambda)
  float decay_rate = 0.01f;
  // Activation decay time constant
  float activation_tau = 3.0f;
  // Co-activation count threshold before wiring a new edge
  uint32_t wiring_threshold = 2;
  // Minimum weight below which an edge is pruned
  float prune_threshold = 0.05f;
  // Maximum weight (saturation)
  float max_weight = 1.0f;
  // Spreading activation falloff per hop
  float spreading_gamma = 0.5f;
  // Whether to auto-prune weak edges on decay
  bool auto_prune = true;
  // Whether to auto-create assemblies (group edges) from co-firing
  bool auto_assembly = true;
  // Minimum co-firing count to form an assembly
  uint32_t assembly_threshold = 3;
  // Maximum degree before triggering normalization
  uint32_t degree_cap = 50;
  // Initial weight for newly created edges
  float initial_edge_weight = 0.1f;
};

// A transient activation snapshot (debugging/inspection)
struct ActivationSnapshot {
  std::unordered_map<std::string, float> activations;
  uint32_t tick;
};

// Result of a spreading-activation query
struct ActivationResult {
  std::vector<std::pair<std::string, float>> frontier; // (node_id, activation)
  std::vector<std::string> fired_edges; // edges that were potentiated
  uint32_t tick;
};

class HebbianEngine {
public:
  explicit HebbianEngine(MemoryGraph &graph,
                         HebbianConfig config = HebbianConfig{});

  // Configuration
  const HebbianConfig &config() const;
  void setConfig(const HebbianConfig &config);

  // Core Hebbian Operations

  /*
   * LAW 1 - FIRE
   * Activate a node at the given level (default 1.0 = full fire).
   * Idempotent within the same tick: max(current, level).
   * Complexity: O(1)
   */
  void fire(const std::string &nodeId, float level = 1.0f);

  /*
   * Fire a set of nodes at once (batch).
   * Complexity: O(|seeds|)
   */
  void fireAll(const std::vector<std::string> &nodeIds, float level = 1.0f);

  /*
   * LAW 2 - WIRE (potentiation)
   * Apply Hebbian potentiation to all co-activated pairs.
   * Must be called AFTER fire() for the current event.
   * Complexity: O(|F|^2 . d) where |F| = |firing_set|, d = avg degree
   */
  void potentiate();

  /*
   * LAW 3 - FORGET (decay)
   * Advance the tick counter and decay:
   *  - activations: a <- a . exp(-1/t_a)
   *  - weights: w <- w . (1 - lambda) [lazy, on read]
   * If auto_prune is enable, removes edges with w < prune_threshold/
   * Complexity: O(|F|) for activation decay; pruning is O(E_pruned)
   */
  void decay(uint32_t ticks = 1);

  // Spreading Activation (retrieval)

  /*
   * Activate a frontier starting from seed nodes.
   * Returns the top-k activated nodes (excluding seeds).
   * Complexity: O(k . d_max) where d_max = max degree in frontier.
   */
  ActivationResult activate(const std::vector<std::string> &seeds,
                            size_t k = 10, int maxDepth = 3);

  // Structural Plasticity

  /*
   * Extract cell assemblies: groups of nodes that have co-fired.
   * >= assembly_threshold times. Creates group edges if auto_assembly.
   * Complexity: O(E . log E) - runs during consolidation, not per-turn.
   */
  std::vector<std::string> consolidate();

  // Inspection
  float getActivation(const std::string &nodeId) const;
  const std::unordered_map<std::string, float> &getActivations() const;
  uint32_t getCoactivationCount(const std::string &a,
                                const std::string &b) const;
  uint32_t getTick() const;
  ActivationSnapshot snapshot() const;
  float getEffectiveWeight(const std::string &edgeId) const;

  // State management
  /*
   * Reset transient state (activations, co-activation counters)
   * but keeps learning weights.
   */
  void resetTransient();

  // Full reset including weights (restores graph to initial state)
  void resetAll();

  // Serialization of Hebbian state (for checkpointing)
  nlohmann::json toHebbianJson() const;
  void fromHebbianJson(const nlohmann::json &state);

private:
  MemoryGraph &graph_;
  HebbianConfig config_;

  // Transient state (Not persisted in graph topology)
  std::unordered_map<std::string, float> activations_;
  std::unordered_map<std::string, uint32_t> last_fired_tick_;
  std::unordered_map<std::string, uint32_t> coactivation_count_;
  uint32_t tick_ = 0;

  // Internal helpers
  std::string makePairKey(const std::string &a, const std::string &b) const;
  void potentiatePair(const std::string &a, const std::string &b, float a_act,
                      float b_act, float a_conf, float b_conf);
  void ensureOrCreateEdge(const std::string &a, const std::string &b);
  void pruneWeakEdges();
  void normalizeHub(const std::string &nodeId);
  float readConfidence(const Node &node) const;
  void ensureModeAllowed(const std::string &operation) const;
};

} // namespace memory_graph
