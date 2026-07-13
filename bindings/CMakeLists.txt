# bindings/CMakeLists.txt
cmake_minimum_required(VERSION 3.15)

# find pybind11
add_subdirectory(pybind11)

pybind11_add_module(memory_graph_core memory_graph.cpp)

# link with the memory_graph library
target_link_libraries(memory_graph_core PRIVATE memory_graph)
target_include_directories(memory_graph_core PRIVATE ${CMAKE_SOURCE_DIR}/include)

# set python module properties
target_compile_definitions(memory_graph_core PRIVATE VERSION_INFO=$"{PROJECT_VERSION}")

# intall to Python site-packages
# install(TARGETS memory_graph_core DESTINATION ${PYTHON_SITELIB}/memory_graph)
