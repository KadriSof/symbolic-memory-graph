# tests/CMakeLists.txt

# FetchContent to get GTest automatically
include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.0  # Use a stable release
)

# For Windows to prevent overriding parent project settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

add_executable(test_memory_graph
    test_node.cpp
    test_edge.cpp
    test_memory_graph.cpp
    test_serialization.cpp
    test_traversal.cpp
)

target_link_libraries(test_memory_graph
    PRIVATE
        memory_graph
        GTest::gtest_main
)

# Discover tests
include(GoogleTest)
gtest_discover_tests(test_memory_graph)
