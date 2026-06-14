#include "memory_graph/node.hpp"
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

using namespace memory_graph;
using json = nlohmann::json;

// Constructor Tests
TEST(NodeTest, DefaultMetadataConstructor) {
  Node node("luffy", "Monkey D. Luffy");

  EXPECT_EQ(node.getId(), "luffy");
  EXPECT_EQ(node.getLabel(), "Monkey D. Luffy");
  EXPECT_TRUE(node.getMetadata().is_null()); // Default metadata should be null
  EXPECT_TRUE(node.getConnections().empty());
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
  EXPECT_TRUE(node.getConnections().empty());
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

// Connection Tests
TEST(NodeTest, AddConnection) {
  Node node("luffy", "Monkey D. Luffy");

  node.addConnection("shanks");
  EXPECT_EQ(node.getConnections().size(), 1);
  EXPECT_EQ(node.getConnections()[0], "shanks");
}

TEST(NodeTest, AddDuplicateConnection) {
  Node node("luffy", "Monkey D. Luffy");

  node.addConnection("shanks");
  node.addConnection("shanks"); // Duplicate

  EXPECT_EQ(node.getConnections().size(), 1); // Should still be 1
  EXPECT_EQ(node.getConnections()[0], "shanks");
}

TEST(NodeTest, AddMultipleConnections) {
  Node node("luffy", "Monkey D. Luffy");

  node.addConnection("shanks");
  node.addConnection("zoro");
  node.addConnection("nami");

  EXPECT_EQ(node.getConnections().size(), 3);

  // Verify all connections exist
  const auto &connections = node.getConnections();
  EXPECT_TRUE(std::find(connections.begin(), connections.end(), "shanks") !=
              connections.end());
  EXPECT_TRUE(std::find(connections.begin(), connections.end(), "zoro") !=
              connections.end());
  EXPECT_TRUE(std::find(connections.begin(), connections.end(), "nami") !=
              connections.end());
}

TEST(NodeTest, RemoveConnection) {
  Node node("luffy", "Monkey D. Luffy");
  node.addConnection("shanks");
  node.addConnection("zoro");
  node.addConnection("nami");

  node.removeConnection("zoro");
  EXPECT_EQ(node.getConnections().size(), 2);

  const auto &connections = node.getConnections();
  EXPECT_TRUE(std::find(connections.begin(), connections.end(), "shanks") !=
              connections.end());
  EXPECT_FALSE(std::find(connections.begin(), connections.end(), "zoro") !=
               connections.end());
  EXPECT_TRUE(std::find(connections.begin(), connections.end(), "nami") !=
              connections.end());
}

TEST(NodeTest, RemoveNonExistentConnection) {
  Node node("luffy", "Monkey D. Luffy");
  node.addConnection("shanks");

  // Removing non-existent connection should do nothing (no crash)
  node.removeConnection("zoro");
  EXPECT_EQ(node.getConnections().size(), 1);
  EXPECT_EQ(node.getConnections()[0], "shanks");
}

TEST(NodeTest, RemoveFromEmptyConnections) {
  Node node("luffy", "Monkey D. Luffy");

  // Removing from empty vector should not crash
  EXPECT_NO_THROW(node.removeConnection("anything"));
  EXPECT_TRUE(node.getConnections().empty());
}

// Serialization Tests
TEST(NodeTest, ToJsonBasic) {
  Node node("luffy", "Monkey D. Luffy", json{{"bounty", 3000000000}});
  node.addConnection("shanks");
  node.addConnection("zoro");

  json nodeJson = node.toJson();

  EXPECT_EQ(nodeJson["id"], "luffy");
  EXPECT_EQ(nodeJson["label"], "Monkey D. Luffy");
  EXPECT_EQ(nodeJson["metadata"]["bounty"], 3000000000);
  EXPECT_EQ(nodeJson["connections"].size(), 2);
  EXPECT_EQ(nodeJson["connections"][0], "shanks");
  EXPECT_EQ(nodeJson["connections"][1], "zoro");
}

TEST(NodeTest, ToJsonNoConnections) {
  Node node("luffy", "Monkey D. Luffy");

  json nodeJson = node.toJson();

  EXPECT_EQ(nodeJson["connections"].size(), 0);
  EXPECT_TRUE(nodeJson["connections"].is_array());
}

TEST(NodeTest, FromJsonBasic) {
  json nodeJson = {{"id", "luffy"},
                   {"label", "Monkey D. Luffy"},
                   {"connections", {"shanks", "zoro"}},
                   {"metadata", {{"bounty", 3000000000}}}};

  Node node = Node::fromJson(nodeJson);

  EXPECT_EQ(node.getId(), "luffy");
  EXPECT_EQ(node.getLabel(), "Monkey D. Luffy");
  EXPECT_EQ(node.getMetadata()["bounty"], 3000000000);
  EXPECT_EQ(node.getConnections().size(), 2);
  EXPECT_EQ(node.getConnections()[0], "shanks");
  EXPECT_EQ(node.getConnections()[1], "zoro");
}

TEST(NodeTest, FromJsonNoMetadata) {
  json nodeJson = {
      {"id", "luffy"}, {"label", "Monkey D. Luffy"}, {"connections", {"shanks"}}
      // No metadata field
  };

  Node node = Node::fromJson(nodeJson);

  EXPECT_EQ(node.getId(), "luffy");
  EXPECT_EQ(node.getLabel(), "Monkey D. Luffy");
  EXPECT_TRUE(node.getMetadata().is_object()); // Should default to empty object
  EXPECT_EQ(node.getConnections().size(), 1);
}

TEST(NodeTest, FromJsonEmptyConnections) {
  json nodeJson = {{"id", "luffy"},
                   {"label", "Monkey D. Luffy"},
                   {"connections", json::array()},
                   {"metadata", json::object()}};

  Node node = Node::fromJson(nodeJson);

  EXPECT_TRUE(node.getConnections().empty());
}

TEST(NodeTest, SerializationRoundTrip) {
  // Original node
  Node original("luffy", "Monkey D. Luffy", json{{"bounty", 3000000000}});
  original.addConnection("shanks");
  original.addConnection("zoro");

  // Serialize and deserialize
  json serialized = original.toJson();
  Node deserialized = Node::fromJson(serialized);

  // Compare
  EXPECT_EQ(original.getId(), deserialized.getId());
  EXPECT_EQ(original.getLabel(), deserialized.getLabel());
  EXPECT_EQ(original.getMetadata()["bounty"],
            deserialized.getMetadata()["bounty"]);
  EXPECT_EQ(original.getConnections().size(),
            deserialized.getConnections().size());
  EXPECT_EQ(original.getConnections()[0], deserialized.getConnections()[0]);
  EXPECT_EQ(original.getConnections()[1], deserialized.getConnections()[1]);
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
  EXPECT_TRUE(node.getConnections().empty());

  // The following would NOT compile (good - const correctness)
  // node.setLabel("new");  // Error: cannot call non-const method on const
  // object
}
