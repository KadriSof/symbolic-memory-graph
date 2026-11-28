#include "memory_graph/node.hpp"
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <string>

using namespace memory_graph;
using json = nlohmann::json;

// Constructor Tests
TEST(NodeTest, DefaultMetadataConstructor) {
  Node node("luffy", "Monkey D. Luffy");

  EXPECT_EQ(node.getId(), "luffy");
  EXPECT_EQ(node.getLabel(), "Monkey D. Luffy");
  EXPECT_TRUE(node.getMetadata().is_object());
  EXPECT_TRUE(node.getMetadata().empty());
}

TEST(NodeTest, ConstructorWithMetadata) {
  json metadata = {{"type", "character"}, {"bounty", 3000000000}};
  Node node("luffy", "Monkey D. Luffy", metadata);

  EXPECT_EQ(node.getId(), "luffy");
  EXPECT_EQ(node.getLabel(), "Monkey D. Luffy");
  EXPECT_EQ(node.getMetadata()["type"], "character");
  EXPECT_EQ(node.getMetadata()["bounty"], 3000000000);
}

// Getter Tests
TEST(NodeTest, Getters) {
  Node node("zoro", "Roronoa Zoro", json{{"role", "swordsman"}});

  EXPECT_EQ(node.getId(), "zoro");
  EXPECT_EQ(node.getLabel(), "Roronoa Zoro");
  EXPECT_EQ(node.getMetadata()["role"], "swordsman");
}

// Setter Tests
TEST(NodeTest, SetLabel) {
  Node node("nami", "Nami");
  EXPECT_EQ(node.getLabel(), "Nami");

  node.setLabel("Cat Burglar Nami");
  EXPECT_EQ(node.getLabel(), "Cat Burglar Nami");
}

TEST(NodeTest, SetMetadata) {
  Node node("chopper", "Tony Tony Chopper");
  json newMetadata = {{"type", "doctor"}, {"bounty", 1000}};

  node.setMetadata(newMetadata);
  EXPECT_EQ(node.getMetadata()["type"], "doctor");
  EXPECT_EQ(node.getMetadata()["bounty"], 1000);
}

TEST(NodeTest, UpdateMetadata) {
  Node node("robin", "Nico Robin", json{{"type", "archaeologist"}});

  node.updateMetadata("age", 30);
  EXPECT_EQ(node.getMetadata()["age"], 30);
  EXPECT_EQ(node.getMetadata()["type"],
            "archaeologist"); // Existing key unchanged

  node.updateMetadata("type", "devil_fruit_user");
  EXPECT_EQ(node.getMetadata()["type"], "devil_fruit_user"); // Overwrites
}

// Serialization Tests
TEST(NodeTest, ToJson) {
  Node node("luffy", "Monkey D. Luffy", json{{"bounty", 3000000000}});

  json nodeJson = node.toJson();

  EXPECT_EQ(nodeJson["id"], "luffy");
  EXPECT_EQ(nodeJson["label"], "Monkey D. Luffy");
  EXPECT_EQ(nodeJson["metadata"]["bounty"], 3000000000);

  // FIX #4: connections are no longer serialized from Node
  EXPECT_FALSE(nodeJson.contains("connections"));
}

TEST(NodeTest, FromJson) {
  json nodeJson = {{"id", "luffy"},
                   {"label", "Monkey D. Luffy"},
                   {"metadata", json::object()}};

  Node node = Node::fromJson(nodeJson);

  EXPECT_EQ(node.getId(), "luffy");
  EXPECT_EQ(node.getLabel(), "Monkey D. Luffy");
  EXPECT_TRUE(node.getMetadata().is_object());
}

TEST(NodeTest, FromJsonIgnoresLegacyConnections) {
  // If legacy JSON has connections, it should be safely ignored
  json nodeJson = {{"id", "luffy"},
                   {"label", "Monkey D. Luffy"},
                   {"connections", json::array()},
                   {"metadata", json::object()}};

  Node node = Node::fromJson(nodeJson);

  EXPECT_EQ(node.getId(), "luffy");
  EXPECT_EQ(node.getLabel(), "Monkey D. Luffy");

  // Verify it doesn't leak into new serialization
  EXPECT_FALSE(node.toJson().contains("connections"));
}

// Edge Case Tests
TEST(NodeTest, EmptyId) {
  Node node("", "Empty ID");
  EXPECT_EQ(node.getId(), "");
  EXPECT_EQ(node.getLabel(), "Empty ID");
}

TEST(NodeTest, EmptyLabel) {
  Node node("test", "");
  EXPECT_EQ(node.getId(), "test");
  EXPECT_EQ(node.getLabel(), "");
}

TEST(NodeTest, VeryLongStrings) {
  std::string longId(10000, 'a');
  std::string longLabel(10000, 'b');

  Node node(longId, longLabel);

  EXPECT_EQ(node.getId(), longId);
  EXPECT_EQ(node.getLabel(), longLabel);
}

// Const Correctness Tests
TEST(NodeTest, ConstNodeGetters) {
  const Node node("const_test", "Const Node", json{{"test", true}});

  // These should compile and work on const node
  EXPECT_EQ(node.getId(), "const_test");
  EXPECT_EQ(node.getLabel(), "Const Node");
  EXPECT_EQ(node.getMetadata()["test"], true);
}
