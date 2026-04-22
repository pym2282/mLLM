cmake_minimum_required(VERSION 3.20)
project(MiniVLLM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Output executable name
add_executable(MiniVLLM
    src/main.cpp
)

# Include headers
target_include_directories(MiniVLLM PRIVATE
    include
)

# Windows: UTF-8 source handling (optional)
if (MSVC)
    target_compile_options(MiniVLLM PRIVATE /utf-8)
endif()
