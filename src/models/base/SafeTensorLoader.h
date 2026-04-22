#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>

namespace mllm
{
    class SafeTensorLoader
    {
    public:
        // Step 1:
        // model.safetensors 파일 존재 여부 확인
        static bool Exists(const std::string& model_dir)
        {
            const std::string file_path = model_dir + "/model.safetensors";

            if (!std::filesystem::exists(file_path))
            {
                std::cerr << "[SafeTensorLoader] File not found: "
                          << file_path << std::endl;
                return false;
            }

            std::cout << "[SafeTensorLoader] Found: "
                      << file_path << std::endl;

            return true;
        }

        // Step 2:
        // 이후 구현 예정
        // - header parsing
        // - tensor metadata parsing
        // - actual tensor loading
    };
}
