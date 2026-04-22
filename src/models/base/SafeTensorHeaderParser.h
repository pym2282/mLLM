#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <cstdint>

#include <nlohmann/json.hpp>

namespace mllm
{
    struct TensorMeta
    {
        std::string dtype;
        std::vector<int64_t> shape;
        std::vector<int64_t> data_offsets;
    };

    class SafeTensorHeaderParser
    {
    public:
        static bool Parse(
            const std::string& model_dir,
            std::unordered_map<std::string, TensorMeta>& out_tensors)
        {
            const std::string file_path = model_dir + "/model.safetensors";

            std::ifstream file(file_path, std::ios::binary);
            if (!file.is_open())
            {
                std::cerr << "[SafeTensorHeaderParser] Failed to open: "
                          << file_path << std::endl;
                return false;
            }

            // 첫 8 bytes = header size (little endian uint64)
            uint64_t header_size = 0;
            file.read(reinterpret_cast<char*>(&header_size), sizeof(uint64_t));

            if (header_size == 0)
            {
                std::cerr << "[SafeTensorHeaderParser] Invalid header size"
                          << std::endl;
                return false;
            }

            std::string header_json;
            header_json.resize(header_size);
            file.read(header_json.data(), static_cast<std::streamsize>(header_size));

            nlohmann::json j = nlohmann::json::parse(header_json);

            for (auto it = j.begin(); it != j.end(); ++it)
            {
                if (it.key() == "__metadata__")
                    continue;

                TensorMeta meta;

                meta.dtype = it.value().value("dtype", "UNKNOWN");
                meta.shape = it.value().value("shape", std::vector<int64_t>{});
                meta.data_offsets = it.value().value("data_offsets", std::vector<int64_t>{});

                out_tensors[it.key()] = meta;
            }

            std::cout << "[SafeTensorHeaderParser] Parsed tensor count: "
                      << out_tensors.size() << std::endl;

            return true;
        }
    };
}
