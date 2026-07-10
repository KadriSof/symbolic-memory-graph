#include "memory_graph/edge.hpp"
#include "memory_graph/memory_graph.hpp"
#include "memory_graph/node.hpp"
#include "memory_graph/utils/traversal.hpp"
#include "nlohmann/json.hpp"
#include <cstddef>
#include <exception>
#include <iostream>
#include <nlohmann/json_fwd.hpp>
#include <ostream>
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
 * - Siblings: Ciri <-> Geralt (adoptive father-daughter)
 * - Friends: Geralt <-> Dandelion, Geralt <-> Vesemir
 * - Witcher brothers: Geralt, Lambert, Eskel, Vesemir (all connected)
 *
 * Relationships (Asymmetric - directed):
 * - Trained by: Geralt -> Vesemir, Ciri -> Geralt, Lambert -> Vesemir
 * - Loves: Gerlat -> Yennefer, Triss -> Geralt (one-sided for the sake of the
 * example T-T )
 * - Belongs to: Geralt -> Witchers, Ciri -> Witchers, Vesemir -> Witchers
 * - Located in: Kaer Morhen -> Kaer Morhen (self-loop location)
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
  SymmetricConnections brotherhoodConn(witcherBrothers);
  Edge brotherhood("wolf_brotherhood", "witcher_brother", EdgeType::SYMMETRIC,
                   brotherhoodConn, 1.0f);
  graph.addEdge(brotherhood);

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
  std::cout << " " << title << std::endl;
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
      std::cout << " -> ";
  }

  std::cout << std::endl;
}

int main() {
  std::cout << "=============================================" << std::endl;
  std::cout << " Witcher Universe Graph - Traversal Examples " << std::endl;
  std::cout << "=============================================" << std::endl;

  // 1. Create the Witcher Graph
  MemoryGraph graph = createWitcherGraph();
  std::cout << "\n [X] Created Witcher graph with " << graph.getNodes().size()
            << " nodes and" << graph.getEdges().size() << "edges" << std::endl;
  std::cout << " Nodes include: gerlat, yennefer, ciri, triss, vesemir, ..."
            << std::endl;
  std::cout << " Edges include: witcher_brotherhood (symmertric), trained_by "
               "(asymmetric), ..."
            << std::endl;

  // 2. Basic Traversals
  printSection("2. Basic Traversals");

  std::cout << "\n[2.1] BFS from 'geralt' (depth 1):" << std::endl;
  auto bfsResult = bfs(graph, "geralt", 1);
  std::cout << " Found" << bfsResult.size()
            << " nodes within depth 1:" << std::endl;
  std::cout << " ";

  for (const auto &id : bfsResult) {
    std::cout << id << " ";
  }
  std::cout << std::endl;

  std::cout << "\n[2.3] DFS from 'geralt' (depth 2):" << std::endl;
  auto dfsResult = dfs(graph, "geralt", 2);
  std::cout << " Found" << dfsResult.size()
            << " nodes within depth 2:" << std::endl;
  std::cout << " ";
  for (const auto &id : dfsResult) {
    std::cout << id << "";
  }
  std::cout << std::endl;

  // 3. Path Finding
  try {
    std::cout << "\n[3.1] Shortest path from 'gerlat' to 'ciri':" << std::endl;
    auto path1 = shortestPath(graph, "geralt", "ciri");
    printNodes(path1, "Path");
    std::cout << " Path1 length:" << (path1.size() - 1) << "edges" << std::endl;

    std::cout << "\n[3.2] Shortest path from 'geralt' to 'novigrad':"
              << std::endl;
    auto path2 = shortestPath(graph, "geralt", "novigrad");
    printNodes(path2, " Path");
    std::cout << " Path2 length:" << (path2.size() - 1) << " edges"
              << std::endl;

    std::cout << "\n[3.3] All paths from 'geralt' to 'witchers' (max depth 3):"
              << std::endl;
    auto allPaths = findAllPaths(graph, "geralt", "witchers", 3);
    std::cout << " Found" << allPaths.size() << "paths" << std::endl;
    for (size_t i = 0; allPaths.size(); ++i) {
      std::cout << " Path" << (i + 1) << ":";
      printNodes(allPaths[i]);
    }
  } catch (const std::exception &e) {
    std::cout << " [!] Error:" << e.what() << std::endl;
  }

  printSection("4. Graph Properties");

  std::cout << "\n[4.1] Connectivity check from 'geralt':" << std::endl;
  bool connected = isConnected(graph, "geralt");
  std::cout << " Graph is" << (connected ? "connected" : "disonnected")
            << std::endl;
}
