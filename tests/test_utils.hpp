#pragma once

#include "memory_graph/edge.hpp"
#include "memory_graph/memory_graph.hpp"
#include "memory_graph/node.hpp"
#include <nlohmann/json.hpp>

using namespace memory_graph;
using json = nlohmann::json;

namespace test_utils {

/**
 * @brief Create a test graph with various node and edge types
 *
 * Graph structure:
 * - Nodes: luffy, zoro, nami, shanks, straw_hats
 * - Symmetric edges: individual pairwise crew relationships (each with exactly
 * 2 nodes)
 * - Asymmetric edges: luffy → shanks (inspired_by), luffy → straw_hats
 * (is_captain_of)
 */
inline MemoryGraph createTestGraph() {
  MemoryGraph graph(json{{"name", "Test Graph"}, {"version", "1.0"}});

  // Add nodes
  Node luffy("luffy", "Monkey D. Luffy",
             json{{"type", "character"}, {"bounty", 3000000000}});
  Node zoro("zoro", "Roronoa Zoro",
            json{{"type", "character"}, {"bounty", 1111000000}});
  Node nami("nami", "Nami", json{{"type", "character"}, {"bounty", 366000000}});
  Node shanks("shanks", "Shanks",
              json{{"type", "character"}, {"bounty", 4048900000}});
  Node strawHats("straw_hats", "Straw Hat Pirates", json{{"type", "crew"}});

  graph.addNode(luffy);
  graph.addNode(zoro);
  graph.addNode(nami);
  graph.addNode(shanks);
  graph.addNode(strawHats);

  // ========================================================================
  // SYMMETRIC: Individual pairwise crew relationships (exactly 2 nodes each)
  // ========================================================================
  // Each edge creates a bidirectional connection between exactly 2 nodes:
  // luffy ↔ zoro, luffy ↔ nami, zoro ↔ nami
  // (Matches core logic requirement: symmetric = exactly 2 nodes)

  SymmetricConnections crew1{"luffy", "zoro"};
  Edge edge1("luffy_zoro", "crew_member", EdgeType::SYMMETRIC, crew1, 1.0f);
  graph.addEdge(edge1);

  SymmetricConnections crew2{"luffy", "nami"};
  Edge edge2("luffy_nami", "crew_member", EdgeType::SYMMETRIC, crew2, 1.0f);
  graph.addEdge(edge2);

  SymmetricConnections crew3{"zoro", "nami"};
  Edge edge3("zoro_nami", "crew_member", EdgeType::SYMMETRIC, crew3, 1.0f);
  graph.addEdge(edge3);

  // ========================================================================
  // ASYMMETRIC: Directed relationships
  // ========================================================================

  // Luffy was inspired by Shanks (directed: luffy → shanks)
  AsymmetricConnections inspirationConn{"luffy", "shanks"};
  Edge inspiration("luffy_shanks", "inspired_by", EdgeType::ASYMMETRIC,
                   inspirationConn, 0.95f,
                   json{{"since", "Age 7"}, {"location", "East Blue"}});
  graph.addEdge(inspiration);

  // Luffy is captain of the Straw Hat Pirates (directed: luffy → straw_hats)
  AsymmetricConnections captainConn{"luffy", "straw_hats"};
  Edge captain("luffy_captain", "is_captain_of", EdgeType::ASYMMETRIC,
               captainConn, 1.0f);
  graph.addEdge(captain);

  return graph;
}

/**
 * @brief Create a modified test graph for delta testing
 *
 * This graph represents changes from the original:
 * - Removed: nami, shanks
 * - Added: usopp, sanji
 * - Modified: crew edges now use different node combinations
 * - Modified: metadata version updated
 *
 * Graph structure:
 * - Nodes: luffy, zoro, usopp, sanji, straw_hats
 * - Symmetric edges: pairwise crew relationships (luffy↔zoro, luffy↔usopp,
 *   luffy↔sanji, zoro↔usopp, zoro↔sanji, usopp↔sanji)
 * - Asymmetric edges: luffy → straw_hats (is_captain_of), zoro → luffy
 * (follows)
 */
inline MemoryGraph createModifiedGraph() {
  MemoryGraph graph(json{{"name", "Modified Test"}, {"version", "2.0"}});

  // Add nodes (removed nami, shanks; added usopp, sanji)
  Node luffy("luffy", "Monkey D. Luffy",
             json{{"type", "character"}, {"bounty", 3000000000}});
  Node zoro("zoro", "Roronoa Zoro",
            json{{"type", "character"}, {"bounty", 1111000000}});
  Node usopp("usopp", "Usopp",
             json{{"type", "character"}, {"bounty", 500000000}});
  Node sanji("sanji", "Sanji",
             json{{"type", "character"}, {"bounty", 1032000000}});
  Node strawHats("straw_hats", "Straw Hat Pirates", json{{"type", "crew"}});

  graph.addNode(luffy);
  graph.addNode(zoro);
  graph.addNode(usopp);
  graph.addNode(sanji);
  graph.addNode(strawHats);

  // ========================================================================
  // SYMMETRIC: Individual pairwise crew relationships (exactly 2 nodes each)
  // ========================================================================
  // This creates bidirectional connections between ALL crew member pairs:
  // luffy ↔ zoro, luffy ↔ usopp, luffy ↔ sanji,
  // zoro ↔ usopp, zoro ↔ sanji, usopp ↔ sanji
  // (6 edges for 4 nodes - matches core logic requirement)

  // Luffy with others
  SymmetricConnections crew1{"luffy", "zoro"};
  Edge edge1("luffy_zoro", "crew_member", EdgeType::SYMMETRIC, crew1, 1.0f);
  graph.addEdge(edge1);

  SymmetricConnections crew2{"luffy", "usopp"};
  Edge edge2("luffy_usopp", "crew_member", EdgeType::SYMMETRIC, crew2, 1.0f);
  graph.addEdge(edge2);

  SymmetricConnections crew3{"luffy", "sanji"};
  Edge edge3("luffy_sanji", "crew_member", EdgeType::SYMMETRIC, crew3, 1.0f);
  graph.addEdge(edge3);

  // Zoro with others (excluding luffy which is already defined)
  SymmetricConnections crew4{"zoro", "usopp"};
  Edge edge4("zoro_usopp", "crew_member", EdgeType::SYMMETRIC, crew4, 1.0f);
  graph.addEdge(edge4);

  SymmetricConnections crew5{"zoro", "sanji"};
  Edge edge5("zoro_sanji", "crew_member", EdgeType::SYMMETRIC, crew5, 1.0f);
  graph.addEdge(edge5);

  // Usopp with Sanji (only remaining pair)
  SymmetricConnections crew6{"usopp", "sanji"};
  Edge edge6("usopp_sanji", "crew_member", EdgeType::SYMMETRIC, crew6, 1.0f);
  graph.addEdge(edge6);

  // ========================================================================
  // ASYMMETRIC: Directed relationships
  // ========================================================================

  // Luffy is captain of the crew (directed: luffy → straw_hats)
  AsymmetricConnections captainConn{"luffy", "straw_hats"};
  Edge captain("luffy_captain", "is_captain_of", EdgeType::ASYMMETRIC,
               captainConn, 1.0f);
  graph.addEdge(captain);

  // Zoro follows Luffy (directed: zoro → luffy)
  AsymmetricConnections followsConn{"zoro", "luffy"};
  Edge follows("zoro_follows", "follows", EdgeType::ASYMMETRIC, followsConn,
               1.0f);
  graph.addEdge(follows);

  return graph;
}

} // namespace test_utils
