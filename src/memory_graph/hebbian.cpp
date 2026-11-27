#include "memory_graph/hebbian.hpp"
#include "memory_graph/edge.hpp"
#include "memory_graph/exceptions.hpp"
#include "memory_graph/memory_graph.hpp"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace memory_graph {

HebbianEngine::HebbianEngine(MemoryGraph &graph, HebbianConfig config)
    : graph_(graph), config_(config) {
  ensureModeAllowed("construct HebbianEngine");
}

const HebbianConfig &HebbianEngine::config() const { return config_; }
void HebbianEngine::setConfig(const HebbianConfig &config) { config_ = config; }

// LAW 1 - FIRE
void HebbianEngine::fire(const std::string &nodeId, float level) {
  ensureModeAllowed("fire");

  if (!graph_.hasNode(nodeId)) {
    throw NodeNotFoundError(nodeId);
  }

  if (level < 0.0f)
    level = 0.0f;
  if (level > 1.0f)
    level = 1.0f;

  // Idempotent within same tick: take max
  auto it = activations_.find(nodeId);
  if (it == activations_.end() || it->second < level) {
    activations_[nodeId] = level;
  }
  last_fired_tick_[nodeId] = tick_;
}

void HebbianEngine::fireAll(const std::vector<std::string> &nodeIds,
                            float level) {
  ensureModeAllowed("fireAll");

  for (const auto &id : nodeIds) {
    fire(id, level);
  }
}

// LAW 2 - WIRE (potentiation)
void HebbianEngine::ensureOrCreateEdge(const std::string &a,
                                       const std::string &b) {
  std::string edgeId = "hebb_" + a + "_" + b;
  AsymmetricConnections conn(a, b);
  nlohmann::json meta = {
      {"relation", "hebbian", {"last_potentiated_tick", "tick"}}};
  Edge edge(edgeId, "HEBBIAN", EdgeType::ASYMMETRIC, conn,
            config_.initial_edge_weight, meta);
  graph_.addEdge(edge);
}

void HebbianEngine::potentiate() {
  ensureModeAllowed("potentiation");

  // Collect firing set (nodes with activations > 0)
  std::vector<std::pair<std::string, float>> firing_set;
  firing_set.reserve(activations_.size());
  for (const auto &[nodeId, act] : activations_) {
    if (act > 0.0f) {
      firing_set.emplace_back(nodeId, act);
    }
  }

  if (firing_set.size() < 2) {
    return; // we need at least 2 co-activated nodes
  }

  // For each piar in the firing set
  for (size_t i = 0; i < firing_set.size(); i++) {
    for (size_t j = i + 1; j < firing_set.size(); ++j) {
      const auto &[a, a_act] = firing_set[i];
      const auto &[b, b_act] = firing_set[j];

      // Increment co-activation counter
      std::string key = makePairKey(a, b);
      uint32_t &count = coactivation_count_[key];
      count++;

      // Read confidences
      float a_conf = readConfidence(graph_.getNode(a));
      float b_conf = readConfidence(graph_.getNode(b));

      // Check if edge exists
      auto edgeId = graph_.findEdgeBetween(a, b);
      if (edgeId) {
        potentiatePair(a, b, a_act, b_act, a_conf, b_conf);
      } else if (count >= config_.wiring_threshold) {
        ensureOrCreateEdge(a, b);
        potentiatePair(a, b, a_act, b_act, a_conf, b_conf);
      }
    }
  }
}

void HebbianEngine::potentiatePair(const std::string &a, const std::string &b,
                                   float a_act, float b_act, float a_conf,
                                   float b_conf) {
  auto edgeIdOpt = graph_.findEdgeBetween(a, b);
  if (!edgeIdOpt) {
    return;
  }
  Edge &edge = const_cast<Edge &>(graph_.getEdge(*edgeIdOpt));

  // Read current weight (apply lazy decay first)
  float current_w = getEffectiveWeight(*edgeIdOpt);

  // Hebbian potentiation
  float delta = config_.learning_rate * a_act * b_act * a_conf * b_conf *
                (config_.max_weight - current_w);

  float new_w = current_w + delta;
  if (new_w > config_.max_weight) {
    new_w = config_.max_weight;
  }
  if (new_w < 0.0f) {
    new_w = 0.0f;
  }

  edge.setWeight(new_w);

  // Update last_potentiated_tick in metadata
  nlohmann::json meta = edge.getMetadata();
  meta["last_potentiated_tick"] = tick_;
  edge.setMetadata(meta);

  // Hub normalization (if degree too high, scale down weakest edges)
  if (graph_.getNeighbors(a).size() > config_.degree_cap) {
    normalizeHub(a);
  }
  if (graph_.getNeighbors(b).size() > config_.degree_cap) {
    normalizeHub(b);
  }
}

// LAW 3 - FORGET (decay)

void HebbianEngine::pruneWeakEdges() {
  std::vector<std::string> to_prune;
  for (const auto &[edgeId, edge] : graph_.getEdgesRef()) {
    float effective_w = getEffectiveWeight(edgeId);
    if (effective_w < config_.prune_threshold) {
      to_prune.push_back(edgeId);
    }
  }

  for (const auto &edgeId : to_prune) {
    graph_.removeEdge(edgeId);
  }
}

void HebbianEngine::decay(uint32_t ticks) {
  ensureModeAllowed("decay");

  // Advance ticks
  tick_ += ticks;

  // Decay activations: a <- a . exp(-ticks/t_a)
  float decay_factor =
      std::exp(-static_cast<float>(ticks) / config_.activation_tau);
  std::vector<std::string> to_remove;
  for (auto &[nodeId, act] : activations_) {
    act *= decay_factor;
    if (act < 0.01f) {
      to_remove.push_back(nodeId);
    }
  }

  for (const auto &id : to_remove) {
    activations_.erase(id);
  }

  // Prune weak edges if enabled
  if (config_.auto_prune) {
    pruneWeakEdges();
  }
}

void HebbianEngine::normalizeHub(const std::string &nodeId) {
  // Get all outgoing edges from this node
  std::vector<std::pair<std::string, float>> edge_weights;
  const auto &adjList = graph_.getAdjacencyList();
  auto it = adjList.find(nodeId);
  if (it == adjList.end())
    return;

  for (const auto &entry : it->second) {
    float w = getEffectiveWeight(entry.edge_id);
    edge_weights.emplace_back(entry.edge_id, w);
  }

  // Sort by weight ascending
  std::sort(edge_weights.begin(), edge_weights.end(),
            [](const auto &a, const auto &b) { return a.second < b.second; });

  // Remove weakest edges until degree is within cap
  size_t to_remove = edge_weights.size() - config_.degree_cap;
  for (size_t i = 0; i < to_remove; ++i) {
    graph_.removeEdge(edge_weights[i].first);
  }
}
} // namespace memory_graph
