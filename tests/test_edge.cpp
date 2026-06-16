#include <gtest/gtest.h>
#include <memory_graph/edge.hpp>
#include <nlohmann/json.hpp>
#include <stdexcept>

using namespace memory_graph;
using json = nlohmann::json;

// Constructor Tests
TEST(EdgeTest, ConstructorAsymmetric) {
  AsymmetricConnections conn{"luffy", "shanks"};
  Edge edge("e1", "inspired_by", EdgeType::ASYMMETRIC, conn, 0.95f,
            json{{"since", "Age 7"}});

  EXPECT_EQ(edge.getId(), "e1");
  EXPECT_EQ(edge.getLabel(), "inspired_by");
  EXPECT_EQ(edge.getType(), EdgeType::ASYMMETRIC);
  EXPECT_EQ(edge.getWeight(), 0.95f);
  EXPECT_EQ(edge.getMetadata()["since"], "Age 7");

  // Verify connections
  const auto &conn_result =
      std::get<AsymmetricConnections>(edge.getConnections());
  EXPECT_EQ(conn_result.first, "luffy");
  EXPECT_EQ(conn_result.second, "shanks");
}

TEST(EdgeTest, ConstructorSymmetric) {
  SymmetricConnections conn{{"luffy", "zoro", "nami"}};
  Edge edge("e1", "crew_members", EdgeType::SYMMETRIC, conn, 1.0f);

  EXPECT_EQ(edge.getId(), "e1");
  EXPECT_EQ(edge.getLabel(), "crew_members");
  EXPECT_EQ(edge.getType(), EdgeType::SYMMETRIC);
  EXPECT_EQ(edge.getWeight(), 1.0f);

  // Verify connections
  const auto &conn_result =
      std::get<SymmetricConnections>(edge.getConnections());
  EXPECT_EQ(conn_result.size(), 3);
  EXPECT_TRUE(conn_result.find("luffy") != conn_result.end());
  EXPECT_TRUE(conn_result.find("zoro") != conn_result.end());
  EXPECT_TRUE(conn_result.find("nami") != conn_result.end());
}

TEST(EdgeTest, DefaultWeightAndMetadata) {
  AsymmetricConnections conn{"luffy", "shanks"};
  Edge edge("e1", "inspired_by", EdgeType::SYMMETRIC, conn);

  EXPECT_EQ(edge.getWeight(), 1.0f);         // Default weight
  EXPECT_TRUE(edge.getMetadata().is_null()); // Defualt metadata
}

TEST(EdgeTest, InvalidWeightThrows) {
  AsymmetricConnections conn{"luffy", "shanks"};

  EXPECT_THROW(Edge("e1", "bad", EdgeType::ASYMMETRIC, conn, -0.5f),
               std::invalid_argument);
  EXPECT_THROW(Edge("e1", "bad", EdgeType::ASYMMETRIC, conn, 1.5f),
               std::invalid_argument);
  EXPECT_NO_THROW(Edge("e1", "good", EdgeType::ASYMMETRIC, conn, 0.5f));
}

// Getter Tests
TEST(EdgeTest, Getters) {
  AsymmetricConnections conn{"luffy", "zoro"};
  json metadata = {{"type", "friendship"}};
  Edge edge("e1", "ally", EdgeType::ASYMMETRIC, conn, 0.8f, metadata);

  EXPECT_EQ(edge.getId(), "e1");
  EXPECT_EQ(edge.getLabel(), "ally");
  EXPECT_EQ(edge.getType(), EdgeType::ASYMMETRIC);
  EXPECT_EQ(edge.getWeight(), 0.8f);
  EXPECT_EQ(edge.getMetadata()["type"], "friendship");
}

TEST(EdgeTest, SetLabel) {
  AsymmetricConnections conn{"luffy", "shanks"};
  Edge edge("e1", "inspired_by", EdgeType::ASYMMETRIC, conn);

  edge.setLabel("mentor");
  EXPECT_EQ(edge.getLabel(), "mentor");
}

TEST(EdgeTest, SetValidWeight) {
  AsymmetricConnections conn{"luffy", "shanks"};
  Edge edge("e1", "relation", EdgeType::ASYMMETRIC, conn);

  edge.setWeight(0.75f);
  EXPECT_EQ(edge.getWeight(), 0.75f);

  edge.setWeight(0.0f);
  EXPECT_EQ(edge.getWeight(), 0.0f);

  edge.setWeight(1.0f);
  EXPECT_EQ(edge.getWeight(), 1.0f);
}

TEST(EdgeTest, SetInvalidWeightThrows) {
  AsymmetricConnections conn{"luffy", "shanks"};
  Edge edge("e1", "relation", EdgeType::ASYMMETRIC, conn);

  EXPECT_THROW(edge.setWeight(-0.1f), std::invalid_argument);
  EXPECT_THROW(edge.setWeight(1.1f), std::invalid_argument);
  EXPECT_EQ(edge.getWeight(), 1.0f); // Original weight unchanged
}

TEST(EdgeTest, SetMetadata) {
  AsymmetricConnections conn{"luffy", "shanks"};
  Edge edge("e1", "relation", EdgeType::ASYMMETRIC, conn);

  json newMetadata = {{"importance", "high"}};
  edge.setMetadata(newMetadata);
  EXPECT_EQ(edge.getMetadata()["importance"], "high");
}

TEST(EdgeTest, UpdateMetadata) {
  AsymmetricConnections conn{"luffy", "shanks"};
  Edge edge("e1", "relation", EdgeType::ASYMMETRIC, conn, 1.0f,
            json{{"since", "Age 7"}});

  edge.updateMetadata("location", "East Blue");
  EXPECT_EQ(edge.getMetadata()["since"], "Age 7"); // Existing unchanged
  EXPECT_EQ(edge.getMetadata()["location"], "East Blue");

  edge.updateMetadata("since", "Childhood");
  EXPECT_EQ(edge.getMetadata()["since"], "Childhood"); // Overwritten
}

// Type-Specific Tests
TEST(EdgeTest, AsymmetricConnectionsAccess) {
  AsymmetricConnections conn{"luffy", "shanks"};
  Edge edge("e1", "inspired_by", EdgeType::ASYMMETRIC, conn);

  const auto &result = std::get<AsymmetricConnections>(edge.getConnections());
  EXPECT_EQ(result.first, "luffy");
  EXPECT_EQ(result.second, "shanks");
}

TEST(EdgeTest, SymmetricConnectionsAccess) {
  SymmetricConnections conn{{"luffy", "zoro", "nami"}};
  Edge edge("e1", "crew", EdgeType::SYMMETRIC, conn);

  const auto &result = std::get<SymmetricConnections>(edge.getConnections());
  EXPECT_EQ(result.size(), 3);
  EXPECT_TRUE(result.find("luffy") != result.end());
  EXPECT_TRUE(result.find("zoro") != result.end());
}

TEST(EdgeTest, EdgeTypeEnumComparison) {
  AsymmetricConnections asym_conn{"luffy", "shanks"};
  Edge asym_edge("e1", "relation", EdgeType::ASYMMETRIC, asym_conn);

  SymmetricConnections sym_conn{{"luffy", "zoro"}};
  Edge sym_edge("e1", "relation", EdgeType::SYMMETRIC, sym_conn);

  EXPECT_NE(asym_edge.getType(), sym_edge.getType());
  EXPECT_EQ(asym_edge.getType(), EdgeType::ASYMMETRIC);
  EXPECT_EQ(sym_edge.getType(), EdgeType::SYMMETRIC);
}

// Serialization Tests
TEST(EdgeTest, ToJsonAsymmetric) {
  AsymmetricConnections conn{"luffy", "shanks"};
  Edge edge("e1", "inspired_by", EdgeType::ASYMMETRIC, conn, 0.95f,
            json{{"since", "Age 7"}, {"location", "East Blue"}});

  json edgeJson = edge.toJson();

  EXPECT_EQ(edgeJson["id"], "e1");
  EXPECT_EQ(edgeJson["label"], "inspired_by");
  EXPECT_EQ(edgeJson["type"], "ASYMMETRIC");
  EXPECT_EQ(edgeJson["weight"], 0.95f);
  EXPECT_EQ(edgeJson["connections"]["type"], "asymmetric");
  EXPECT_EQ(edgeJson["connections"]["source"], "luffy");
  EXPECT_EQ(edgeJson["connections"]["target"], "shanks");
  EXPECT_EQ(edgeJson["metadata"]["since"], "Age 7");
  EXPECT_EQ(edgeJson["metadata"]["location"], "East Blue");
}

TEST(EdgeTest, ToJsonSymmetric) {
  SymmetricConnections conn{{"luffy", "zoro", "nami"}};
  Edge edge("e1", "crew_members", EdgeType::SYMMETRIC, conn, 1.0f);

  json edgeJson = edge.toJson();

  EXPECT_EQ(edgeJson["id"], "e1");
  EXPECT_EQ(edgeJson["label"], "crew_members");
  EXPECT_EQ(edgeJson["type"], "SYMMETRIC");
  EXPECT_EQ(edgeJson["connections"]["type"], "symmetric");
  EXPECT_EQ(edgeJson["connections"]["nodes"].size(), 3);
  EXPECT_TRUE(edgeJson["connections"]["nodes"][0] == "luffy" ||
              edgeJson["connections"]["nodes"][1] == "luffy" ||
              edgeJson["connections"]["nodes"][2] == "luffy");
}

TEST(EdgeTest, FromJsonAsymmetric) {
  json edgeJson = {
      {"id", "e1"},
      {"label", "inspired_by"},
      {"type", "ASYMMETRIC"},
      {"weight", 0.95},
      {"connections",
       {{"type", "asymmetric"}, {"source", "luffy"}, {"target", "shanks"}}},
      {"metadata", {{"since", "Age 7"}}}};

  Edge edge = Edge::fromJson(edgeJson);

  EXPECT_EQ(edge.getId(), "e1");
  EXPECT_EQ(edge.getLabel(), "inspired_by");
  EXPECT_EQ(edge.getType(), EdgeType::ASYMMETRIC);
  EXPECT_FLOAT_EQ(edge.getWeight(), 0.95f);
  EXPECT_EQ(edge.getMetadata()["since"], "Age 7");

  const auto &conn = std::get<AsymmetricConnections>(edge.getConnections());
  EXPECT_EQ(conn.first, "luffy");
  EXPECT_EQ(conn.second, "shanks");
}

TEST(EdgeTest, FromJsonSymmetric) {
  json edgeJson = {
      {"id", "e1"},
      {"label", "crew_members"},
      {"type", "SYMMETRIC"},
      {"weight", 1.0},
      {"connections",
       {{"type", "symmetric"}, {"nodes", {"luffy", "zoro", "nami"}}}}};

  Edge edge = Edge::fromJson(edgeJson);

  EXPECT_EQ(edge.getId(), "e1");
  EXPECT_EQ(edge.getLabel(), "crew_members");
  EXPECT_EQ(edge.getType(), EdgeType::SYMMETRIC);
  EXPECT_FLOAT_EQ(edge.getWeight(), 1.0f);

  const auto &conn = std::get<SymmetricConnections>(edge.getConnections());
  EXPECT_EQ(conn.size(), 3);
  EXPECT_TRUE(conn.find("luffy") != conn.end());
  EXPECT_TRUE(conn.find("zoro") != conn.end());
  EXPECT_TRUE(conn.find("nami") != conn.end());
}

TEST(EdgeTest, FromJsonDefaultWeight) {
  json edgeJson = {
      {"id", "e1"},
      {"label", "relation"},
      {"type", "ASYMMETRIC"},
      {"connections",
       {{"type", "asymmetric"}, {"source", "luffy"}, {"target", "shanks"}}}
      // No weight field - should default to 1.0
  };

  Edge edge = Edge::fromJson(edgeJson);
  EXPECT_FLOAT_EQ(edge.getWeight(), 1.0f);
}

TEST(EdgeTest, SerializationRoundTripAsymmetric) {
  // Original edge
  AsymmetricConnections original_conn{"luffy", "shanks"};
  Edge original("e1", "inspired_by", EdgeType::ASYMMETRIC, original_conn, 0.95f,
                json{{"since", "Age 7"}});

  // Serialize and deserialize
  json serialized = original.toJson();
  Edge deserialized = Edge::fromJson(serialized);

  // Compare
  EXPECT_EQ(original.getId(), deserialized.getId());
  EXPECT_EQ(original.getLabel(), deserialized.getLabel());
  EXPECT_EQ(original.getType(), deserialized.getType());
  EXPECT_FLOAT_EQ(original.getWeight(), deserialized.getWeight());
  EXPECT_EQ(original.getMetadata()["since"],
            deserialized.getMetadata()["since"]);

  const auto &orig_conn =
      std::get<AsymmetricConnections>(original.getConnections());
  const auto &new_conn =
      std::get<AsymmetricConnections>(deserialized.getConnections());
  EXPECT_EQ(orig_conn.first, new_conn.first);
  EXPECT_EQ(orig_conn.second, new_conn.second);
}

TEST(EdgeTest, SerializationRoundTripSymmetric) {
  // Original edge
  SymmetricConnections original_conn({"luffy", "zoro", "nami"});
  Edge original("e1", "crew", EdgeType::SYMMETRIC, original_conn);

  // Serialize and deserialize
  json serialized = original.toJson();
  Edge deserialized = Edge::fromJson(serialized);

  // Compare
  EXPECT_EQ(original.getId(), deserialized.getId());
  EXPECT_EQ(original.getLabel(), deserialized.getLabel());
  EXPECT_EQ(original.getType(), deserialized.getType());
  EXPECT_FLOAT_EQ(original.getWeight(), deserialized.getWeight());

  const auto &orig_conn =
      std::get<SymmetricConnections>(original.getConnections());
  const auto &new_conn =
      std::get<SymmetricConnections>(deserialized.getConnections());
  EXPECT_EQ(orig_conn.size(), new_conn.size());
}

// Edge Case Tests
TEST(EdgeTest, EmptyStrings) {
  AsymmetricConnections conn{"", ""};
  Edge edge("", "", EdgeType::ASYMMETRIC, conn);

  EXPECT_EQ(edge.getId(), "");
  EXPECT_EQ(edge.getLabel(), "");

  const auto &result = std::get<AsymmetricConnections>(edge.getConnections());
  EXPECT_EQ(result.first, "");
  EXPECT_EQ(result.second, "");
}

TEST(EdgeTest, VeryLongStrings) {
  std::string longId(10000, 'a');
  std::string longLabel(10000, 'b');
  AsymmetricConnections conn{longId, longLabel};

  Edge edge(longId, longLabel, EdgeType::ASYMMETRIC, conn);

  EXPECT_EQ(edge.getId(), longId);
  EXPECT_EQ(edge.getLabel(), longLabel);

  const auto &result = std::get<AsymmetricConnections>(edge.getConnections());
  EXPECT_EQ(result.first, longId);
  EXPECT_EQ(result.second, longLabel);
}

TEST(EdgeTest, EmptySymmetricConnections) {
  SymmetricConnections conn;
  Edge edge("e1", "empty", EdgeType::SYMMETRIC, conn);

  const auto &result = std::get<SymmetricConnections>(edge.getConnections());
  EXPECT_TRUE(result.empty());
}

TEST(EdgeTest, SingleNodeSymmetricConnection) {
  SymmetricConnections conn({"luffy"});
  Edge edge("e1", "self", EdgeType::SYMMETRIC, conn);

  const auto &result = std::get<SymmetricConnections>(edge.getConnections());
  EXPECT_EQ(result.size(), 1);
  EXPECT_TRUE(result.find("luffy") != result.end());
}

TEST(EdgeTest, MultipleUpdatesToMetadata) {
  AsymmetricConnections conn{"luffy", "shanks"};
  Edge edge("e1", "relation", EdgeType::ASYMMETRIC, conn);

  edge.updateMetadata("key1", "value1");
  edge.updateMetadata("key2", "value2");
  edge.updateMetadata("key1", "new_value");

  EXPECT_EQ(edge.getMetadata()["key1"], "new_value");
  EXPECT_EQ(edge.getMetadata()["key2"], "value2");
  EXPECT_EQ(edge.getMetadata().size(), 2);
}

// Const Correctness Tests
TEST(EdgeTest, ConstEdgeGetters) {
  AsymmetricConnections conn{"luffy", "shanks"};
  const Edge edge("e1", "relation", EdgeType::ASYMMETRIC, conn);

  // These should compile and work on const edge
  EXPECT_EQ(edge.getId(), "e1");
  EXPECT_EQ(edge.getLabel(), "relation");
  EXPECT_EQ(edge.getType(), EdgeType::ASYMMETRIC);
  EXPECT_EQ(edge.getWeight(), 1.0f);

  // This would NOT compile (good - const correctness)
  // edge.setLabel("new");  // Error: cannot call non-const method on const
  // object
}

// Variant Type Tests
TEST(EdgeTest, HoldsAlternativeCheck) {
  AsymmetricConnections asym_conn{"luffy", "shanks"};
  Edge asym_edge("e1", "asym", EdgeType::ASYMMETRIC, asym_conn);

  SymmetricConnections sym_conn({"luffy", "zoro"});
  Edge sym_edge("e2", "sym", EdgeType::SYMMETRIC, sym_conn);

  EXPECT_TRUE(std::holds_alternative<AsymmetricConnections>(
      asym_edge.getConnections()));
  EXPECT_FALSE(
      std::holds_alternative<SymmetricConnections>(asym_edge.getConnections()));

  EXPECT_TRUE(
      std::holds_alternative<SymmetricConnections>(sym_edge.getConnections()));
  EXPECT_FALSE(
      std::holds_alternative<AsymmetricConnections>(sym_edge.getConnections()));
}
