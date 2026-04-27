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
        static bool Exists(
            const std::string& model_path)
        {
            namespace fs = std::filesystem;

            const fs::path single_file =
                fs::path(model_path) /
                "model.safetensors";

            const fs::path sharded_index =
                fs::path(model_path) /
                "model.safetensors.index.json";

            if (fs::exists(single_file))
            {
                std::cout
                    << "[SafeTensorLoader] Found single safetensor: "
                    << single_file.string()
                    << std::endl;

                return true;
            }

            if (fs::exists(sharded_index))
            {
                std::cout
                    << "[SafeTensorLoader] Found sharded safetensor index: "
                    << sharded_index.string()
                    << std::endl;

                return true;
            }

            std::cerr
                << "[SafeTensorLoader] File not found:\n"
                << "  "
                << single_file.string()
                << "\n"
                << "  "
                << sharded_index.string()
                << std::endl;

            return false;
        }

        // Step 2:
        // 이후 구현 예정
        // - header parsing
        // - tensor metadata parsing
        // - actual tensor loading
    };
}
