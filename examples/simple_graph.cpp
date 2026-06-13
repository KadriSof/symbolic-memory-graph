#include "memory_graph/edge.hpp"
#include "memory_graph/memory_graph.hpp"
#include "memory_graph/node.hpp"
#include <iostream>

using namespace memory_graph;
using json = nlohmann::json;

int main() {
  // Create a memory graph
  MemoryGraph graph(
      json{{"name", "One Piece Memory Graph"}, {"version", "1.0"}});

  // Add nodes
  Node luffy("luffy", "Monkey D. Luffy",
             json{{"type", "Character"}, {"bounty", 3000000000}});
  Node shanks("shanks", "Shanks",
              json{{"type", "Character"}, {"bounty", 4048900000}});
  Node strawHats("straw_hats", "Straw Hat Pirates", json{{"type", "Crew"}});

  graph.addNode(luffy);
  graph.addNode(shanks);
  graph.addNode(strawHats);

  // Add edges
  AsymmetricConnections luffyShanksConn = {"luffy", "shanks"};
  Edge inspirationEdge("luffy_shanks_inspiration", "inspired by",
                       EdgeType::ASYMMETRIC, luffyShanksConn, 0.95f,
                       json{{"since", "Age 7"}, {"location", "East Blue"}});
  graph.addEdge(inspirationEdge);

  AsymmetricConnections luffyStrawHatsConn = {"luffy", "straw_hats"};
  Edge captainOf("luffy_captai_straw_hats", "is captain of",
                 EdgeType::ASYMMETRIC, luffyStrawHatsConn, 1.0f,
                 json{{"since", "151st year of the Greate Pirate Era"}});
  graph.addEdge(captainOf);

  // Query the graph
  std::cout << "=== luffy's Neighbors ===" << std::endl;
  for (const auto &neighbor : graph.getNeighbors("luffy")) {
    std::cout << "- " << neighbor.getLabel() << std::endl;
  }

  // Test query with depth
  std::cout << "\n === Query Result (Depth 1) ===" << std::endl;
  auto queryResult = graph.query("luffy", 1);
  std::cout << queryResult.dump(2) << std::endl;

  // Serialize to JSON
  std::cout << "\n Graph JSON ===" << std::endl;
  json graphJson = graph.toJson();
  std::cout << graphJson.dump(2) << std::endl;

  // Deserialize from JSON
  MemoryGraph newGraph = MemoryGraph::fromJson(graphJson);
  std::cout << "\n === Deserailization Graph Node ===" << std::endl;
  for (const auto &node : newGraph.getNodes()) {
    std::cout << "- " << node.getId() << ": " << node.getLabel() << std::endl;
  }

  return 0;
}
