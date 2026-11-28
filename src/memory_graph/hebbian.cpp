#include "memory_graph/hebbian.hpp"
#include "memory_graph/edge.hpp"
#include "memory_graph/exceptions.hpp"
#include "memory_graph/memory_graph.hpp"
#include "memory_graph/node.hpp"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace memory_graph {

HebbianEngine::HebbianEngine(MemoryGraph &graph, HebbianConfig config)
    : graph_(graph), config_(config) {
  ensureModeAllowed("construct HebbianEngine");
}

const HebbianConfig &HebbianEngine::config() const { return config_; }
void HebbianEngine::setConfig(const HebbianConfig &config) { config_ = config; }

// Helpers
std::string HebbianEngine::makePairKey(const std::string &a,
                                       const std::string &b) const {
  if (a < b)
    return a + "|" + b;
  return b + "|" + a;
}

float HebbianEngine::readConfidence(const Node &node) const {
  const auto &meta = node.getMetadata();
  if (meta.contains("confidence")) {
    return meta["confidence"].get<float>();
  }
  return 0.5f;
}

void HebbianEngine::ensureModeAllowed(const std::string &operation) const {
  if (graph_.getMode() == GraphMode::KNOWLEDGE) {
    throw ModeError(operation, "KNOWLEDGE");
  }
}

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
  std::string key = makePairKey(a, b);
  std::string edgeId = "hebb_" + key;

  if (graph_.hasEdge(edgeId))
    return;

  AsymmetricConnections conn(a, b);
  nlohmann::json meta = {{"relation", "hebbian"},
                         {"last_potentiated_tick", tick_}};

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

// SPREADING ACTIVATION
ActivationResult HebbianEngine::activate(const std::vector<std::string> &seeds,
                                         size_t k, int maxDepth) {
  ensureModeAllowed("activate");

  ActivationResult result;
  result.tick = tick_;

  if (seeds.empty())
    return result;

  // Accumulated activation per node
  std::unordered_map<std::string, float> accumulated;
  std::unordered_set<std::string> seed_set(seeds.begin(), seeds.end());

  // Initialize seeds
  std::unordered_map<std::string, float> current_frontier;
  for (const auto &seed : seeds) {
    if (!graph_.hasNode(seed)) {
      throw NodeNotFoundError(seed);
    }
    current_frontier[seed] = 1.0f;
  }

  const auto &adjList = graph_.getAdjacencyList();

  // Spread activation across hops
  for (int hop = 1; hop <= maxDepth; ++hop) {
    std::unordered_map<std::string, float> next_frontier;
    float gamma_hop = std::pow(config_.spreading_gamma, hop);

    for (const auto &[nodeId, act] : current_frontier) {
      auto it = adjList.find(nodeId);
      if (it == adjList.end())
        continue;

      for (const auto &entry : it->second) {
        if (seed_set.count(entry.neighbor_id))
          continue;

        float edge_w = getEffectiveWeight(entry.edge_id);
        float contribution = act * edge_w * gamma_hop;
        next_frontier[entry.neighbor_id] += contribution;
        accumulated[entry.neighbor_id] += contribution;
      }
    }

    current_frontier = std::move(next_frontier);
  }

  // Sort by accumulated activation top-k
  std::vector<std::pair<std::string, float>> ranked(accumulated.begin(),
                                                    accumulated.end());
  std::sort(ranked.begin(), ranked.end(),
            [](const auto &a, const auto &b) { return a.second > b.second; });

  size_t count = std::min(k, ranked.size());
  result.frontier.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    result.frontier.push_back(ranked[i]);
  }

  return result;
}

// CONSOLIDATION
std::vector<std::string> HebbianEngine::consolidate() {
  ensureModeAllowed("consolidate");

  std::vector<std::string> new_assemblies;
  if (!config_.auto_assembly)
    return new_assemblies;

  // Find pairs with coactivation >= assembly_threshold
  std::unordered_map<std::string, std::unordered_set<std::string>> strong_pairs;
  for (const auto &[key, count] : coactivation_count_) {
    if (count >= config_.assembly_threshold) {
      // This weak ass separator logic will break the code at some point, but
      // well..
      auto sep = key.find('|');
      std::string a = key.substr(0, sep);
      std::string b = key.substr(sep + 1);
      strong_pairs[a].insert(b);
      strong_pairs[b].insert(a);
    }
  }

  // For each node with strong pairs, try to form an assembly
  std::unordered_set<std::string> processed;
  for (const auto &[nodeId, neighbors] : strong_pairs) {
    if (processed.count(nodeId))
      continue;

    // Find common neighbors (triangle closure)
    std::unordered_set<std::string> assembly;
    assembly.insert(nodeId);

    for (const auto &n1 : neighbors) {
      if (!strong_pairs.count(n1))
        continue;
      for (const auto &n2 : neighbors) {
        if (n1 == n2)
          continue;
        if (strong_pairs[n1].count(n2)) {
          assembly.insert(n1);
          assembly.insert(n2);
        }
      }
    }

    if (assembly.size() >= 3) {
      // Check if this assembly already exists
      bool exists = false;
      for (const auto &edge : graph_.getGroupEdges()) {
        const auto &conn =
            std::get<SymmetricConnections>(edge.getConnections());
        if (conn == assembly) {
          exists = true;
          break;
        }
      }

      if (!exists) {
        std::string groupId = "assembly_" + nodeId;
        try {
          graph_.addGroupEdge(
              groupId, "ASSEMBLY", assembly, 0.8f,
              {{"relation", "assembly"}, {"formed_tick", tick_}});
          new_assemblies.push_back(groupId);
        } catch (const DuplicateIdError &) {
          // Well.. XDDD
          groupId += "_" + std::to_string(tick_);
          graph_.addGroupEdge(
              groupId, "ASSEMBLY", assembly, 0.8f,
              {{"relation", "assembly"}, {"formed_tick", tick_}});
          new_assemblies.push_back(groupId);
        }
      }

      for (const auto &id : assembly) {
        processed.insert(id);
      }
    }
  }

  return new_assemblies;
}

// INSPECTION
float HebbianEngine::getActivation(const std::string &nodeId) const {
  auto it = activations_.find(nodeId);
  return (it != activations_.end()) ? it->second : 0.0f;
}

const std::unordered_map<std::string, float> &
HebbianEngine::getActivations() const {
  return activations_;
}

uint32_t HebbianEngine::getCoactivationCount(const std::string &a,
                                             const std::string &b) const {
  std::string key = makePairKey(a, b);
  auto it = coactivation_count_.find(key);
  return (it != coactivation_count_.end()) ? it->second : 0;
}

uint32_t HebbianEngine::getTick() const { return tick_; }

ActivationSnapshot HebbianEngine::snapshot() const {
  return ActivationSnapshot{activations_, tick_};
}

float HebbianEngine::getEffectiveWeight(const std::string &edgeId) const {
  const Edge &edge = graph_.getEdge(edgeId);
  float stored_w = edge.getWeight();

  // Apply lazy decay based on last_potentiated_tick
  const auto &meta = edge.getMetadata();
  uint32_t last_tick = tick_;
  if (meta.contains("last_potentiated_tick")) {
    last_tick = meta["last_potentiated_tick"].get<uint32_t>();
  }

  uint32_t elapsed = (tick_ > last_tick) ? (tick_ - last_tick) : 0;
  if (elapsed == 0)
    return stored_w;

  // w_effective = w_stored * (1 - λ)^elapsed
  float effective_w = stored_w * std::pow(1.0f - config_.decay_rate, elapsed);
  return effective_w;
}

// STATE MANAGEMENT
void HebbianEngine::resetTransient() {
  activations_.clear();
  last_fired_tick_.clear();
  coactivation_count_.clear();
  tick_ = 0;
}

void HebbianEngine::resetAll() {
  resetTransient();
  // Remove all Hebbian-created edges
  std::vector<std::string> to_remove;
  for (const auto &[edgeId, edge] : graph_.getEdgesRef()) {
    if (edge.getLabel() == "HEBBIAN" || edge.getLabel() == "ASSEMBLY") {
      to_remove.push_back(edgeId);
    }
  }
  for (const auto &edgeId : to_remove) {
    graph_.removeEdge(edgeId);
  }
}

// SERIALIZATION
nlohmann::json HebbianEngine::toHebbianJson() const {
  nlohmann::json state;
  state["tick"] = tick_;
  state["config"] = {{"learning_rate", config_.learning_rate},
                     {"decay_rate", config_.decay_rate},
                     {"activation_tau", config_.activation_tau},
                     {"wiring_threshold", config_.wiring_threshold},
                     {"prune_threshold", config_.prune_threshold},
                     {"max_weight", config_.max_weight},
                     {"spreading_gamma", config_.spreading_gamma},
                     {"auto_prune", config_.auto_prune},
                     {"auto_assembly", config_.auto_assembly},
                     {"assembly_threshold", config_.assembly_threshold},
                     {"initial_edge_weight", config_.initial_edge_weight},
                     {"degree_cap", config_.degree_cap}};

  // Activations
  nlohmann::json acts = nlohmann::json::object();
  for (const auto &[id, act] : activations_) {
    acts[id] = act;
  }
  state["activations"] = acts;

  // Coactivation counts
  nlohmann::json counts = nlohmann::json::object();
  for (const auto &[key, count] : coactivation_count_) {
    counts[key] = count;
  }
  state["coactivation_counts"] = counts;

  return state;
}

void HebbianEngine::fromHebbianJson(const nlohmann::json &state) {
  tick_ = state.value("tick", 0u);

  if (state.contains("config")) {
    const auto &cfg = state["config"];
    config_.learning_rate = cfg.value("learning_rate", 0.1f);
    config_.decay_rate = cfg.value("decay_rate", 0.01f);
    config_.activation_tau = cfg.value("activation_tau", 3.0f);
    config_.wiring_threshold = cfg.value("wiring_threshold", 2u);
    config_.prune_threshold = cfg.value("prune_threshold", 0.05f);
    config_.max_weight = cfg.value("max_weight", 1.0f);
    config_.spreading_gamma = cfg.value("spreading_gamma", 0.5f);
    config_.auto_prune = cfg.value("auto_prune", true);
    config_.auto_assembly = cfg.value("auto_assembly", true);
    config_.assembly_threshold = cfg.value("assembly_threshold", 3u);
    config_.initial_edge_weight = cfg.value("initial_edge_weight", 0.1f);
    config_.degree_cap = cfg.value("degree_cap", 50u);
  }

  activations_.clear();
  if (state.contains("activations")) {
    for (auto it = state["activations"].begin();
         it != state["activations"].end(); ++it) {
      activations_[it.key()] = it.value().get<float>();
    }
  }

  coactivation_count_.clear();
  if (state.contains("coactivation_counts")) {
    for (auto it = state["coactivation_counts"].begin();
         it != state["coactivation_counts"].end(); ++it) {
      coactivation_count_[it.key()] = it.value().get<uint32_t>();
    }
  }
}

} // namespace memory_graph
