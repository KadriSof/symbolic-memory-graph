#include "memory_graph/edge.hpp"
#include "memory_graph/memory_graph.hpp"
#include "memory_graph/node.hpp"
#include "memory_graph/utils/traversal.hpp"
#include <cstddef>
#include <exception>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_set>
#include <vector>

using namespace memory_graph;
using namespace memory_graph::utils;
using json = nlohmann::json;

/**
 * @brief Create a Witcher-themed graph with various node and edge types
 *
 * Graph structure:
 * - Characters: Geralt, Yennefer, Triss, Ciri, Vesemir, Lambert, Eskel,
 * Dandelion
 * - Locations: Kaer Morhen, Novigrad, Skellige, Vizima
 * - Factions: Witchers, Nilfgaard, Temeria
 *
 * Relationships (Symmetric - bidirectional):
 * - Siblings: Ciri ↔ Geralt (adoptive father-daughter)
 * - Friends: Geralt ↔ Dandelion, Geralt ↔ Vesemir
 * - Witcher brothers: Geralt, Lambert, Eskel, Vesemir (all connected)
 *
 * Relationships (Asymmetric - directed):
 * - Trained by: Geralt → Vesemir, Ciri → Geralt, Lambert → Vesemir
 * - Loves: Geralt → Yennefer, Triss → Geralt (one-sided)
 * - Belongs to: Geralt → Witchers, Ciri → Witchers, Vesemir → Witchers
 * - Located in: Kaer Morhen → Kaer Morhen (self-loop for location)
 */
MemoryGraph createWitcherGraph() {
  MemoryGraph graph(json{{"name", "Witcher Universe Graph"},
                         {"version", "1.0"},
                         {"theme", "The Witcher"}});

  // CHARACTERS
  Node geralt("geralt", "Geralt of Rivia",
              json{{"type", "witcher"}, {"school", "Wolf"}, {"age", 98}});
  Node yen("yennefer", "Yennefer of Vengerberg",
           json{{"type", "sorceress"}, {"age", 94}});
  Node triss("triss", "Triss Merigold",
             json{{"type", "sorceress"}, {"age", 35}});
  Node ciri("ciri", "Cirilla Fiona Elen Riannon",
            json{{"type", "witcher"}, {"age", 21}, {"elder_blood", true}});
  Node vesemir("vesemir", "Vesemir",
               json{{"type", "witcher"}, {"school", "Wolf"}, {"age", 350}});
  Node lambert("lambert", "Lambert",
               json{{"type", "witcher"}, {"school", "Wolf"}, {"age", 70}});
  Node eskel("eskel", "Eskel",
             json{{"type", "witcher"}, {"school", "Wolf"}, {"age", 75}});
  Node dandelion("dandelion", "Dandelion", json{{"type", "bard"}, {"age", 42}});
  Node emhyr("emhyr", "Emhyr var Emreis",
             json{{"type", "emperor"}, {"title", "Emperor of Nilfgaard"}});
  Node calanthe("calanthe", "Calanthe",
                json{{"type", "queen"}, {"title", "Queen of Cintra"}});

  // LOCATIONS
  Node kaerMorhen("kaer_morhen", "Kaer Morhen",
                  json{{"type", "location"}, {"region", "Kaedwen"}});
  Node novigrad("novigrad", "Novigrad",
                json{{"type", "location"}, {"region", "Redania"}});
  Node skellige("skellige", "Skellige",
                json{{"type", "location"}, {"region", "Skellige Islands"}});
  Node vizima("vizima", "Vizima",
              json{{"type", "location"}, {"region", "Temeria"}});

  // FACTIONS
  Node witchers("witchers", "School of the Wolf",
                json{{"type", "faction"}, {"motto", "Killing monsters"}});
  Node nilfgaard("nilfgaard", "Nilfgaardian Empire",
                 json{{"type", "faction"}, {"motto", "In the Empire's name"}});
  Node temeria("temeria", "Kingdom of Temeria",
               json{{"type", "faction"}, {"motto", "For Temeria"}});

  // Add all nodes
  graph.addNode(geralt);
  graph.addNode(yen);
  graph.addNode(triss);
  graph.addNode(ciri);
  graph.addNode(vesemir);
  graph.addNode(lambert);
  graph.addNode(eskel);
  graph.addNode(dandelion);
  graph.addNode(emhyr);
  graph.addNode(calanthe);
  graph.addNode(kaerMorhen);
  graph.addNode(novigrad);
  graph.addNode(skellige);
  graph.addNode(vizima);
  graph.addNode(witchers);
  graph.addNode(nilfgaard);
  graph.addNode(temeria);

  // SYMMETRIC EDGES (Bidirectional / Mutual Relationships)

  // 1. Witcher Brotherhood - all Wolf School witchers are connected
  // This creates a complete graph (clique) among witchers
  // All witchers: geralt, vesemir, lambert, eskel, ciri
  std::unordered_set<std::string> witcherBrothers = {
      "geralt", "vesemir", "lambert", "eskel", "ciri"};
  graph.addGroupEdge("wolf_brotherhood", "witcher_brother", witcherBrothers,
                     1.0f, json{{"school", "wolf"}});

  // 2. Close Friends
  SymmetricConnections friendsConn{"geralt", "dandelion"};
  Edge friends("geralt_dandelion", "close_friend", EdgeType::SYMMETRIC,
               friendsConn, 0.9f);
  graph.addEdge(friends);

  SymmetricConnections friendsConn2{"geralt", "vesemir"};
  Edge friends2("geralt_vesemir", "mentor_mentee", EdgeType::SYMMETRIC,
                friendsConn2, 0.95f);
  graph.addEdge(friends2);

  // 3. Family - Ciri and Geralt
  SymmetricConnections familyConn{"geralt", "ciri"};
  Edge family("geralt_ciri", "adoptive_father_daughter", EdgeType::SYMMETRIC,
              familyConn, 1.0f);
  graph.addEdge(family);

  SymmetricConnections familyConn2{"ciri", "calanthe"};
  Edge family2("ciri_calanthe", "grandmother_granddaughter",
               EdgeType::SYMMETRIC, familyConn2, 0.8f);
  graph.addEdge(family2);

  // ASYMMETRIC EDGES (Directed Relationships)

  // 4. Training relationships (mentor → student)
  AsymmetricConnections geraltVesemir{"geralt", "vesemir"};
  Edge trainedBy1("geralt_trained_by", "trained_by", EdgeType::ASYMMETRIC,
                  geraltVesemir, 1.0f);
  graph.addEdge(trainedBy1);

  AsymmetricConnections ciriGeralt{"ciri", "geralt"};
  Edge trainedBy2("ciri_trained_by", "trained_by", EdgeType::ASYMMETRIC,
                  ciriGeralt, 1.0f);
  graph.addEdge(trainedBy2);

  AsymmetricConnections lambertVesemir{"lambert", "vesemir"};
  Edge trainedBy3("lambert_trained_by", "trained_by", EdgeType::ASYMMETRIC,
                  lambertVesemir, 1.0f);
  graph.addEdge(trainedBy3);

  AsymmetricConnections eskelVesemir{"eskel", "vesemir"};
  Edge trainedBy4("eskel_trained_by", "trained_by", EdgeType::ASYMMETRIC,
                  eskelVesemir, 1.0f);
  graph.addEdge(trainedBy4);

  // 5. Love relationships (one-sided or mutual)
  AsymmetricConnections geraltYen{"geralt", "yennefer"};
  Edge loves1("geralt_loves", "loves", EdgeType::ASYMMETRIC, geraltYen, 1.0f);
  graph.addEdge(loves1);

  AsymmetricConnections yenGeralt{"yennefer", "geralt"};
  Edge loves2("yennefer_loves", "loves", EdgeType::ASYMMETRIC, yenGeralt, 1.0f);
  graph.addEdge(loves2);

  // Triss loves Geralt (one-sided)
  AsymmetricConnections trissGeralt{"triss", "geralt"};
  Edge loves3("triss_loves", "loves", EdgeType::ASYMMETRIC, trissGeralt, 0.85f);
  graph.addEdge(loves3);

  // 6. Faction membership
  AsymmetricConnections geraltWitchers{"geralt", "witchers"};
  Edge belongs1("geralt_belongs", "belongs_to", EdgeType::ASYMMETRIC,
                geraltWitchers, 1.0f);
  graph.addEdge(belongs1);

  AsymmetricConnections ciriWitchers{"ciri", "witchers"};
  Edge belongs2("ciri_belongs", "belongs_to", EdgeType::ASYMMETRIC,
                ciriWitchers, 1.0f);
  graph.addEdge(belongs2);

  AsymmetricConnections vesemirWitchers{"vesemir", "witchers"};
  Edge belongs3("vesemir_belongs", "belongs_to", EdgeType::ASYMMETRIC,
                vesemirWitchers, 1.0f);
  graph.addEdge(belongs3);

  // 7. Location relationships (character → location)
  AsymmetricConnections geraltKaer{"geralt", "kaer_morhen"};
  Edge located1("geralt_located", "located_in", EdgeType::ASYMMETRIC,
                geraltKaer, 0.7f);
  graph.addEdge(located1);

  AsymmetricConnections ciriKaer{"ciri", "kaer_morhen"};
  Edge located2("ciri_located", "located_in", EdgeType::ASYMMETRIC, ciriKaer,
                0.7f);
  graph.addEdge(located2);

  AsymmetricConnections vesemirKaer{"vesemir", "kaer_morhen"};
  Edge located3("vesemir_located", "located_in", EdgeType::ASYMMETRIC,
                vesemirKaer, 0.8f);
  graph.addEdge(located3);

  AsymmetricConnections dandelionNovigrad{"dandelion", "novigrad"};
  Edge located4("dandelion_located", "located_in", EdgeType::ASYMMETRIC,
                dandelionNovigrad, 0.6f);
  graph.addEdge(located4);

  // 8. Political relationships
  AsymmetricConnections calantheCiri{"calanthe", "ciri"};
  Edge grandmother("calanthe_relationship", "grandmother_of",
                   EdgeType::ASYMMETRIC, calantheCiri, 0.9f);
  graph.addEdge(grandmother);

  AsymmetricConnections emhyrCiri{"emhyr", "ciri"};
  Edge father("emhyr_relationship", "father_of", EdgeType::ASYMMETRIC,
              emhyrCiri, 0.6f);
  graph.addEdge(father);

  // 9. Location hierarchy (nested locations)
  AsymmetricConnections kaerKaedwen{"kaer_morhen", "temeria"};
  Edge location1("kaer_region", "located_in_region", EdgeType::ASYMMETRIC,
                 kaerKaedwen, 1.0f);
  graph.addEdge(location1);

  AsymmetricConnections novigradRedania{"novigrad", "temeria"};
  Edge location2("novigrad_region", "located_in_region", EdgeType::ASYMMETRIC,
                 novigradRedania, 1.0f);
  graph.addEdge(location2);

  return graph;
}

/**
 * @brief Print a section header
 */
void printSection(const std::string &title) {
  std::cout << "\n" << std::string(80, '=') << std::endl;
  std::cout << "  " << title << std::endl;
  std::cout << std::string(80, '=') << std::endl;
}

/**
 * @brief Print a vector of node IDs
 */
void printNodes(const std::vector<std::string> &nodes,
                const std::string &label = "") {
  if (!label.empty()) {
    std::cout << label << ": ";
  }
  for (size_t i = 0; i < nodes.size(); ++i) {
    std::cout << nodes[i];
    if (i < nodes.size() - 1)
      std::cout << " → ";
  }
  std::cout << std::endl;
}

int main() {
  std::cout
      << "╔═══════════════════════════════════════════════════════════════╗"
      << std::endl;
  std::cout
      << "║         Witcher Universe Graph - Traversal Examples           ║"
      << std::endl;
  std::cout
      << "╚═══════════════════════════════════════════════════════════════╝"
      << std::endl;

  // 1. Create the Witcher Graph
  auto graph = createWitcherGraph();
  std::cout << "\n[X] Created Witcher graph with " << graph.getNodes().size()
            << " nodes and " << graph.getEdges().size() << " edges"
            << std::endl;
  std::cout
      << "   Nodes include: geralt, yennefer, ciri, vesemir, dandelion, ..."
      << std::endl;
  std::cout << "   Edges include: witcher_brotherhood (symmetric), trained_by "
               "(asymmetric), ..."
            << std::endl;

  // 2. Basic Traversals - BFS and DFS
  printSection("2. Basic Traversals");

  std::cout << "\n[2.1] BFS from 'geralt' (depth 1):" << std::endl;
  auto bfsResult = bfs(graph, "geralt", 1);
  std::cout << "   Found " << bfsResult.size()
            << " nodes within depth 1:" << std::endl;
  std::cout << "   ";
  for (const auto &id : bfsResult) {
    std::cout << id << " ";
  }
  std::cout << std::endl;

  std::cout << "\n[2.2] BFS from 'ciri' (depth 2):" << std::endl;
  auto bfsCiri = bfs(graph, "ciri", 2);
  std::cout << "   Found " << bfsCiri.size()
            << " nodes within depth 2:" << std::endl;
  std::cout << "   ";
  for (const auto &id : bfsCiri) {
    std::cout << id << " ";
  }
  std::cout << std::endl;

  std::cout << "\n[2.3] DFS from 'geralt' (depth 2):" << std::endl;
  auto dfsResult = dfs(graph, "geralt", 2);
  std::cout << "   Found " << dfsResult.size()
            << " nodes within depth 2:" << std::endl;
  std::cout << "   ";
  for (const auto &id : dfsResult) {
    std::cout << id << " ";
  }
  std::cout << std::endl;

  // 3. Path Finding
  printSection("3. Path Finding");

  try {
    std::cout << "\n[3.1] Shortest path from 'geralt' to 'ciri':" << std::endl;
    auto path = shortestPath(graph, "geralt", "ciri");
    printNodes(path, "   Path");
    std::cout << "   Path length: " << (path.size() - 1) << " edges"
              << std::endl;

    std::cout << "\n[3.2] Shortest path from 'geralt' to 'novigrad':"
              << std::endl;
    auto path2 = shortestPath(graph, "geralt", "novigrad");
    printNodes(path2, "   Path");
    std::cout << "   Path length: " << (path2.size() - 1) << " edges"
              << std::endl;

    std::cout << "\n[3.3] All paths from 'geralt' to 'witchers' (max depth 3):"
              << std::endl;
    auto allPaths = findAllPaths(graph, "geralt", "witchers", 3);
    std::cout << "   Found " << allPaths.size() << " paths:" << std::endl;
    for (size_t i = 0; i < allPaths.size(); ++i) {
      std::cout << "   Path " << (i + 1) << ": ";
      printNodes(allPaths[i]);
    }
  } catch (const std::exception &e) {
    std::cout << "   ❌ Error: " << e.what() << std::endl;
  }

  // 4. Graph Properties
  printSection("4. Graph Properties");

  std::cout << "\n[4.1] Connectivity check from 'geralt':" << std::endl;
  bool connected = isConnected(graph, "geralt");
  std::cout << "   Graph is " << (connected ? "connected" : "disconnected")
            << std::endl;

  std::cout << "\n[4.2] Cycle detection:" << std::endl;
  bool hasCycles = hasCycle(graph);
  std::cout << "   Graph has cycles: " << (hasCycles ? "✅ YES" : "❌ NO")
            << std::endl;
  if (hasCycles) {
    std::cout << "   (Cycles exist due to symmetric edges like the witcher "
                 "brotherhood clique)"
              << std::endl;
  }

  try {
    std::cout << "\n[4.3] Topological sort:" << std::endl;
    auto sorted = topologicalSort(graph);
    std::cout << "   Found " << sorted.size()
              << " nodes in topological order:" << std::endl;
    std::cout << "   ";
    for (const auto &id : sorted) {
      std::cout << id << " ";
    }
    std::cout << std::endl;
    std::cout << "   (Topological sort only works for DAGs; it skipped cycles)"
              << std::endl;
  } catch (const std::exception &e) {
    std::cout << "   ❌ Error: " << e.what() << std::endl;
    std::cout << "   (Graph has cycles, so topological sort is impossible)"
              << std::endl;
  }

  // 5. Node Queries
  printSection("5. Node Queries");

  std::cout << "\n[5.1] Find nodes with label containing 'Geralt':"
            << std::endl;
  auto labels = findNodesByLabel(graph, "Geralt of Rivia");
  std::cout << "   Found: ";
  for (const auto &id : labels) {
    std::cout << id << " ";
  }
  std::cout << std::endl;

  std::cout << "\n[5.2] Find nodes by metadata (type = 'witcher'):"
            << std::endl;
  auto witchers = findNodesByMetadata(graph, "type", "witcher");
  std::cout << "   Found " << witchers.size() << " witchers: ";
  for (const auto &id : witchers) {
    std::cout << id << " ";
  }
  std::cout << std::endl;

  std::cout << "\n[5.3] Find nodes by metadata (school = 'Wolf'):" << std::endl;
  auto wolfSchool = findNodesByMetadata(graph, "school", "Wolf");
  std::cout << "   Found " << wolfSchool.size() << " Wolf School witchers: ";
  for (const auto &id : wolfSchool) {
    std::cout << id << " ";
  }
  std::cout << std::endl;

  // 6. Subgraph Extraction
  printSection("6. Subgraph Extraction");

  std::cout << "\n[6.1] Extract subgraph centered on 'geralt' (radius 1):"
            << std::endl;
  auto sub1 = subgraph(graph, "geralt", 1);
  std::cout << "   Subgraph has " << sub1.getNodes().size() << " nodes and "
            << sub1.getEdges().size() << " edges" << std::endl;
  std::cout << "   Nodes: ";
  for (const auto &node : sub1.getNodes()) {
    std::cout << node.getId() << " ";
  }
  std::cout << std::endl;

  std::cout << "\n[6.2] Extract subgraph centered on 'ciri' (radius 2):"
            << std::endl;
  auto sub2 = subgraph(graph, "ciri", 2);
  std::cout << "   Subgraph has " << sub2.getNodes().size() << " nodes and "
            << sub2.getEdges().size() << " edges" << std::endl;

  std::cout << "\n[6.3] Extract subgraph by predicate (all sorceresses):"
            << std::endl;
  auto sub3 = subgraphByPredicate(
      graph,
      [](const Node &node) {
        const auto &meta = node.getMetadata();
        return meta.contains("type") && meta["type"] == "sorceress";
      },
      true // Include neighbors
  );
  std::cout << "   Subgraph has " << sub3.getNodes().size() << " nodes and "
            << sub3.getEdges().size() << " edges" << std::endl;
  std::cout << "   Nodes: ";
  for (const auto &node : sub3.getNodes()) {
    std::cout << node.getId() << " ";
  }
  std::cout << std::endl;

  std::cout
      << "\n[6.4] Extract subgraph by predicate (only Geralt) WITH neighbors:"
      << std::endl;
  auto sub4 = subgraphByPredicate(
      graph, [](const Node &node) { return node.getId() == "geralt"; },
      true // Include neighbors
  );
  std::cout << "   Subgraph has " << sub4.getNodes().size() << " nodes and "
            << sub4.getEdges().size() << " edges" << std::endl;
  std::cout << "   Nodes: ";
  for (const auto &node : sub4.getNodes()) {
    std::cout << node.getId() << " ";
  }
  std::cout << std::endl;

  // 7. LLM Context Window
  printSection("7. LLM Context Window (for Agents)");

  std::cout << "\n[7.1] Get context window for 'geralt' (maxTokens=100, "
               "minRelevance=0.5):"
            << std::endl;
  auto context = getContextWindow(graph, "geralt", 100, 0.5f);
  std::cout << "   Context window:" << std::endl;
  std::cout << "   - Center: " << context["center"].get<std::string>()
            << std::endl;
  std::cout << "   - Node count: " << context["node_count"].get<size_t>()
            << std::endl;
  std::cout << "   - Edge count: " << context["edge_count"].get<size_t>()
            << std::endl;
  std::cout << "   - Token count: " << context["token_count"].get<size_t>()
            << std::endl;

  std::cout << "\n   Nodes in context: ";
  for (const auto &node : context["nodes"]) {
    std::cout << node["id"].get<std::string>() << " ";
  }
  std::cout << std::endl;

  std::cout << "\n[7.2] Get context window for 'ciri' (maxTokens=200, "
               "minRelevance=0.8):"
            << std::endl;
  auto context2 = getContextWindow(graph, "ciri", 200, 0.8f);
  std::cout << "   - Center: " << context2["center"].get<std::string>()
            << std::endl;
  std::cout << "   - Node count: " << context2["node_count"].get<size_t>()
            << std::endl;
  std::cout << "   - Edge count: " << context2["edge_count"].get<size_t>()
            << std::endl;
  std::cout << "   - Token count: " << context2["token_count"].get<size_t>()
            << std::endl;

  // 8. Serialization
  printSection("8. Serialization");

  std::cout << "\n[8.1] Serialize graph to JSON:" << std::endl;
  json graphJson = graph.toJson();
  std::cout << "   Graph JSON size: " << graphJson.dump().size() << " bytes"
            << std::endl;
  std::cout << "   Nodes count in JSON: " << graphJson["nodes"].size()
            << std::endl;
  std::cout << "   Edges count in JSON: " << graphJson["edges"].size()
            << std::endl;

  std::cout << "\n[8.2] Deserialize from JSON:" << std::endl;
  MemoryGraph deserialized = MemoryGraph::fromJson(graphJson);
  std::cout << "   Deserialized graph has " << deserialized.getNodes().size()
            << " nodes and " << deserialized.getEdges().size() << " edges"
            << std::endl;

  // Summary
  printSection("Summary");

  std::cout << "\n✅ All traversal methods demonstrated successfully!"
            << std::endl;
  std::cout << "\nMethods used:" << std::endl;
  std::cout << "  - bfs()        : BFS traversal with depth limiting"
            << std::endl;
  std::cout << "  - dfs()        : DFS traversal with depth limiting"
            << std::endl;
  std::cout << "  - shortestPath(): Find shortest path between nodes"
            << std::endl;
  std::cout << "  - findAllPaths(): Find all paths between nodes" << std::endl;
  std::cout << "  - isConnected() : Check graph connectivity" << std::endl;
  std::cout << "  - hasCycle()    : Detect cycles in the graph" << std::endl;
  std::cout << "  - topologicalSort(): Order nodes by dependencies"
            << std::endl;
  std::cout << "  - findNodesByLabel(): Search nodes by label" << std::endl;
  std::cout << "  - findNodesByMetadata(): Search nodes by metadata"
            << std::endl;
  std::cout << "  - subgraph()    : Extract subgraph by radius" << std::endl;
  std::cout << "  - subgraphByPredicate(): Extract subgraph by condition"
            << std::endl;
  std::cout << "  - getContextWindow(): Get LLM context window" << std::endl;

  std::cout << "\n🎮 Graph represents The Witcher universe with:" << std::endl;
  std::cout << "   - " << graph.getNodes().size()
            << " nodes (characters, locations, factions)" << std::endl;
  std::cout << "   - " << graph.getEdges().size()
            << " edges (symmetrical & asymmetrical relationships)" << std::endl;
  std::cout << "   - Multiple relationship types: family, love, training, "
               "location, etc."
            << std::endl;

  return 0;
}
