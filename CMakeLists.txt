cmake_minimum_required(VERSION 3.20)
project(mLLM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# LibTorch 경로 — CUDA 버전 우선, 없으면 CPU fallback
if(EXISTS "C:/libtorch-cuda/share/cmake/Torch")
    set(Torch_DIR "C:/libtorch-cuda/share/cmake/Torch")
    message(STATUS "Using CUDA LibTorch: C:/libtorch-cuda")
else()
    set(Torch_DIR "C:/libtorch/share/cmake/Torch")
    message(STATUS "Using CPU LibTorch: C:/libtorch")
endif()

find_package(Torch REQUIRED)

add_executable(mLLM
        src/main.cpp

        # runtime
        src/core/runtime/Sampler.cpp

        # tokenizer
        src/tokenizer/BpeTokenizer.cpp
        src/tokenizer/LlamaTokenizer.cpp
        src/tokenizer/QwenTokenizer.cpp
        src/tokenizer/TokenizerJsonLoader.cpp

        # model
        src/models/llama/LlamaRunner.cpp
        src/models/qwen/QwenRunner.cpp

        # serving
        src/serving/Scheduler.cpp
        src/serving/HttpServer.cpp
)

target_precompile_headers(
        mLLM
        PRIVATE
        src/pch.h
)

# httplib.h manages its own Windows socket include order; skip the PCH so
# torch's transitive <windows.h> doesn't pull in winsock 1.x first.
set_source_files_properties(
        src/serving/HttpServer.cpp
        PROPERTIES SKIP_PRECOMPILE_HEADERS ON
)

target_include_directories(mLLM PRIVATE
        src
        include
        external
)

target_link_libraries(mLLM PRIVATE
        ${TORCH_LIBRARIES}
        $<$<PLATFORM_ID:Windows>:ws2_32>
)

if(EXISTS "C:/libtorch-cuda/share/cmake/Torch")
    set(LIBTORCH_LIB_DIR "C:/libtorch-cuda/lib")
else()
    set(LIBTORCH_LIB_DIR "C:/libtorch/lib")
endif()

add_custom_command(
        TARGET mLLM
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${LIBTORCH_LIB_DIR}"
        $<TARGET_FILE_DIR:mLLM>
        COMMENT "Copying LibTorch DLLs from ${LIBTORCH_LIB_DIR}..."
)

if (MSVC)
    target_compile_options(mLLM PRIVATE /utf-8)
endif()