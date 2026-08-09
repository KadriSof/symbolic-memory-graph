# bindings/CMakeLists.txt
cmake_minimum_required(VERSION 3.15)

# Check if pybind11 submodule is initialized
if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/pybind11/CMakeLists.txt")
    message(WARNING "pybind11 submodule not found!")
    message(WARNING "Please run: git submodule update --init --recursive")
    return()
endif()

add_subdirectory(pybind11)

set(BINDING_SOURCES
    module.cpp
    exceptions_binding.cpp
    node_binding.cpp
    edge_binding.cpp
    graph_binding.cpp
    traversal_binding.cpp
    serialization_binding.cpp
)

pybind11_add_module(memory_graph_core ${BINDING_SOURCES})

# Link with the memory_graph library
target_link_libraries(memory_graph_core PRIVATE memory_graph)
target_include_directories(memory_graph_core PRIVATE ${CMAKE_SOURCE_DIR}/include)

# VERSION_INFO as a proper C++ string literal
target_compile_definitions(memory_graph_core PRIVATE 
    VERSION_INFO=\"${PROJECT_VERSION}\"
)

# Auto-copy .so/.pyd/.dylib to Python package on build (Cross-platform & ABI-safe)
add_custom_command(
  TARGET memory_graph_core 
  POST_BUILD
    # 1. Ensure the destination directory exists
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/python/src/memory_graph
    
    # 2. Copy the file, preserving its platform-specific name (e.g., .cpython-312-x86_64-linux-gnu.so)
    COMMAND ${CMAKE_COMMAND} -E copy 
        $<TARGET_FILE:memory_graph_core> 
        ${CMAKE_SOURCE_DIR}/python/src/memory_graph/
        
    COMMENT "Copying shared library to Python package..."
)

install(TARGETS memory_graph_core DESTINATION ${Python_SITELIB}/memory_graph)
