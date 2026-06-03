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

set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe")
# CUDA 12.6 supports MSVC up to 2022; use VS2022 cl.exe as nvcc host compiler
set(CMAKE_CUDA_HOST_COMPILER "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe")
# RTX 3060 = sm_86; skip auto-detection (try_run fails with unsupported MSVC)
set(TORCH_CUDA_ARCH_LIST "8.6")
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
        src/models/gemma/GemmaRunner.cpp

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

# Minimal DLL set — empirically verified via regression_test.py.
# Excludes cuDNN (~1GB), cufft, cusparse, cusolver, curand, cupti, nvperf (unused at inference).
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