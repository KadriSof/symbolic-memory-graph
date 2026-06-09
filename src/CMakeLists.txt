# Build the memory_graph library
add_library(memory_graph STATIC
  memory_graph/node.cpp
  memory_graph/edge.cpp
  memory_graph/memory_graph.cpp
  memory_graph/exceptions.cpp
  memory_graph/utils/serialization.cpp
  memory_graph/utils/traversal.cpp
)

# Specify include directories
target_include_directories(memory_graph
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link dependencies
target_link_libraries(memory_graph PRIVATE nlohmann_json::nlohmann_json)

# Install headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/memory_graph
  DESTINATION include
  FILES_MATCHING PATTERN "*.hpp"
)
