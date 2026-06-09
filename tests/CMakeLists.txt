# Find GoogleTest
find_package(GTest REQUIRED)

# Add test executable
add_executable(test_memory_graph
  test_node.cpp
  test_edge.cpp
  test_memory_graph.cpp
  test_serialization.cpp
)

# Link to the library and GTest
target_link_libraries(test_memory_graph PRIVATE
  memory_graph
  GTest::GTest
  GTest::Main
)

# Enable testing
gtest_discover_tests(test_memory_graph)
