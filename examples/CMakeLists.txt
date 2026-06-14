# memory_graph/examples/CMakeLists.txt

# Simple example
add_executable(simple_graph simple_graph.cpp)
target_link_libraries(simple_graph PRIVATE memory_graph)

# LLM integration example (future)
add_executable(llm_integration llm_integration.cpp)
target_link_libraries(llm_integration PRIVATE memory_graph)
