cmake_minimum_required(VERSION 3.20)
project(mLLM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# LibTorch 경로
set(Torch_DIR "C:/libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(mLLM ${SOURCES})

target_precompile_headers(
        mLLM
        PRIVATE
        src/pch.h
)

target_include_directories(mLLM PRIVATE
        src
        include
        external
)

target_link_libraries(mLLM PRIVATE
        ${TORCH_LIBRARIES}
)

add_custom_command(
        TARGET mLLM
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "C:/libtorch/lib"
        $<TARGET_FILE_DIR:mLLM>
        COMMENT "Copying LibTorch DLLs..."
)

if (MSVC)
    target_compile_options(mLLM PRIVATE /utf-8)
endif()