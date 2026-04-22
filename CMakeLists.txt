cmake_minimum_required(VERSION 3.20)
project(mLLM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# LibTorch 경로
set(Torch_DIR "C:/libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)

add_executable(mLLM
        src/main.cpp
        src/models/llama/LlamaRunner.cpp
)

target_include_directories(mLLM PRIVATE
        src
        include
        external
)

target_link_libraries(mLLM PRIVATE
        ${TORCH_LIBRARIES}
)

if (MSVC)
    target_compile_options(mLLM PRIVATE /utf-8)
endif()