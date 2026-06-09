cmake_minimum_required(VERSION 3.15)
project (MemoryGraph LANGUAGES CXX VERSION 1.0)

# Set C++ standard (C++17)
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_EXTENSIONS OFF)

# Enable testing
enable_testing()


# nlohmann_json interface
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)

# Directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Main library
add_subdirectory(src)

# Tests:
add_subdirectory(tests)

# Examples:
add_subdirectory(examples)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
