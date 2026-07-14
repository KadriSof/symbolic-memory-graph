# bindings/CMakeLists.txt
cmake_minimum_required(VERSION 3.15)

if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/pybind11/CMakeLists.txt")
    message(FATAL_ERROR 
        "pybind11 submodule not found!\n"
        "Please initialize the submodule with:\n"
        "  git submodule update --init --recursive\n"
        "Or clone with:\n"
        "  git clone --recursive <repository-url>"
    )
endif()

add_subdirectory(pybind11)

set(BINDING_SOURCES
    module.cpp
    exceptions_binding.cpp
    node_binding.cpp
    edge_binding.cpp
    graph_binding.cpp
)

pybind11_add_module(memory_graph_core ${BINDING_SOURCES})

# Link with the memory_graph library
target_link_libraries(memory_graph_core PRIVATE memory_graph)
target_include_directories(memory_graph_core PRIVATE ${CMAKE_SOURCE_DIR}/include)

# VERSION_INFO as a proper C++ string literal
target_compile_definitions(memory_graph_core PRIVATE 
    VERSION_INFO=\"${PROJECT_VERSION}\"
)

# Install to Python site-packages
install(TARGETS memory_graph_core DESTINATION ${Python_SITELIB}/memory_graph)
