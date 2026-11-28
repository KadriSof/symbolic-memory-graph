#include "memory_graph/edge.hpp"
#include "memory_graph/exceptions.hpp"
#include "memory_graph/hebbian.hpp"
#include "memory_graph/memory_graph.hpp"
#include "memory_graph/node.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace memory_graph;

void test_fire_sets_activation() {
  MemoryGraph g;
  g.setMode(GraphMode::SYMBOLIC);
  g.addNode(Node("n1", "Node1", {{"confidence", 0.9}}));
  g.attachHebbian();

  g.hebbian().fire("n1", 0.8f);
  assert(std::abs(g.hebbian().getActivation("n1") - 0.8f) < 1e-6f);

  // Idempotent: lower level doesn't reduce
  g.hebbian().fire("n1", 0.5f);
  assert(std::abs(g.hebbian().getActivation("n1") - 0.8f) < 1e-6f);

  // Higher level does increase
  g.hebbian().fire("n1", 1.0f);
  assert(std::abs(g.hebbian().getActivation("n1") - 1.0f) < 1e-6f);

  std::cout << "Fire sets activation\n";
}

void test_potentiate_increases_weight() {
  MemoryGraph g;
  g.setMode(GraphMode::SYMBOLIC);
  g.addNode(Node("n1", "Node1", {{"confidence", 1.0}}));
  g.addNode(Node("n2", "Node2", {{"confidence", 1.0}}));
  g.addEdge(Edge("e1", "relates", EdgeType::ASYMMETRIC,
                 AsymmetricConnections("n1", "n2"), 0.5f));
  g.attachHebbian(HebbianConfig{.learning_rate = 0.2f});

  g.hebbian().fire("n1", 1.0f);
  g.hebbian().fire("n2", 1.0f);
  g.hebbian().potentiate();

  float new_w = g.hebbian().getEffectiveWeight("e1");
  // Δw = 0.2 * 1.0 * 1.0 * 1.0 * 1.0 * (1.0 - 0.5) = 0.1
  assert(std::abs(new_w - 0.6f) < 1e-5f);

  std::cout << "Potentiate increases weight\n";
}

void test_coactivation_creates_edge() {
  MemoryGraph g;
  g.setMode(GraphMode::SYMBOLIC);
  g.addNode(Node("n1", "Node1", {{"confidence", 1.0}}));
  g.addNode(Node("n2", "Node2", {{"confidence", 1.0}}));
  g.attachHebbian(
      HebbianConfig{.wiring_threshold = 2, .initial_edge_weight = 0.1f});

  // First co-firing: no edge yet
  g.learn({"n1", "n2"});
  assert(!g.hasEdge("hebb_n1|n2"));

  // Second co-firing: edge created
  g.learn({"n1", "n2"});
  assert(g.hasEdge("hebb_n1|n2"));
  assert(g.getEdge("hebb_n1|n2").getWeight() > 0.0f);

  std::cout << "Coactivation creates edge after threshold\n";
}

void test_decay_reduces_activation() {
  MemoryGraph g;
  g.setMode(GraphMode::SYMBOLIC);
  g.addNode(Node("n1", "Node1"));
  g.attachHebbian(HebbianConfig{.activation_tau = 2.0f});

  g.hebbian().fire("n1", 1.0f);
  g.hebbian().decay(2); // 2 ticks

  float act = g.hebbian().getActivation("n1");
  // a = 1.0 * exp(-2/2) = exp(-1) ≈ 0.368
  assert(std::abs(act - std::exp(-1.0f)) < 1e-5f);

  std::cout << "Decay reduces activation\n";
}

void test_spreading_activation() {
  MemoryGraph g;
  g.setMode(GraphMode::SYMBOLIC);
  g.addNode(Node("n1", "Node1"));
  g.addNode(Node("n2", "Node2"));
  g.addNode(Node("n3", "Node3"));
  g.addEdge(Edge("e1", "relates", EdgeType::ASYMMETRIC,
                 AsymmetricConnections("n1", "n2"), 0.9f));
  g.addEdge(Edge("e2", "relates", EdgeType::ASYMMETRIC,
                 AsymmetricConnections("n2", "n3"), 0.5f));
  g.attachHebbian(HebbianConfig{.spreading_gamma = 0.5f});

  auto result = g.hebbian().activate({"n1"}, 2, 2);

  assert(result.frontier.size() == 2);
  assert(result.frontier[0].first == "n2"); // Strongest
  assert(result.frontier[0].second > result.frontier[1].second);

  std::cout << "Spreading activation returns ranked results\n";
}

void test_mode_enforcement() {
  MemoryGraph g;
  g.setMode(GraphMode::KNOWLEDGE);
  g.addNode(Node("n1", "Node1"));

  bool threw = false;
  try {
    g.attachHebbian();
  } catch (const ModeError &) {
    threw = true;
  }
  assert(threw);

  std::cout << "Mode enforcement blocks Hebbian in KNOWLEDGE mode\n";
}

void test_serialization() {
  MemoryGraph g;
  g.setMode(GraphMode::SYMBOLIC);
  g.addNode(Node("n1", "Node1"));
  g.addNode(Node("n2", "Node2"));
  g.attachHebbian(HebbianConfig{.learning_rate = 0.15f});

  g.learn({"n1", "n2"});
  g.learn({"n1", "n2"});

  auto state = g.hebbian().toHebbianJson();
  assert(state["tick"] == 2);
  assert(state.contains("coactivation_counts"));

  // Restore
  MemoryGraph g2;
  g2.setMode(GraphMode::SYMBOLIC);
  g2.addNode(Node("n1", "Node1"));
  g2.addNode(Node("n2", "Node2"));
  g2.attachHebbian();
  g2.hebbian().fromHebbianJson(state);

  assert(g2.hebbian().getTick() == 2);
  assert(g2.hebbian().getCoactivationCount("n1", "n2") == 2);

  std::cout << "Hebbian state serialization\n";
}

int main() {
  std::cout << "Running Hebbian tests...\n\n";

  test_fire_sets_activation();
  test_potentiate_increases_weight();
  test_coactivation_creates_edge();
  test_decay_reduces_activation();
  test_spreading_activation();
  test_mode_enforcement();
  test_serialization();

  std::cout << "\nAll Hebbian tests passed!\n";
  return 0;
}
