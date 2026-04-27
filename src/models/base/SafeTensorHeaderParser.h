#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <filesystem>

#include <nlohmann/json.hpp>

namespace mllm
{
    struct TensorMeta
    {
        std::string dtype;
        std::vector<int64_t> shape;
        std::vector<int64_t> data_offsets;

        // sharded safetensors support
        // example:
        // model-00003-of-00005.safetensors
        std::string shard_file;
    };

    class SafeTensorHeaderParser
    {
    public:
        static bool Parse(
            const std::string& model_dir,
            std::unordered_map<std::string, TensorMeta>& out_tensors)
        {
            namespace fs = std::filesystem;

            const fs::path single_file =
                fs::path(model_dir) /
                "model.safetensors";

            const fs::path index_file =
                fs::path(model_dir) /
                "model.safetensors.index.json";

            // =====================================================
            // Case 1
            // Single safetensor
            // =====================================================

            if (fs::exists(single_file))
            {
                return ParseSingleFile(
                    single_file.string(),
                    "",
                    out_tensors
                );
            }

            // =====================================================
            // Case 2
            // Sharded safetensors
            // =====================================================

            if (fs::exists(index_file))
            {
                return ParseShardedIndex(
                    model_dir,
                    index_file.string(),
                    out_tensors
                );
            }

            std::cerr
                << "[SafeTensorHeaderParser] No safetensor found:\n"
                << "  "
                << single_file.string()
                << "\n"
                << "  "
                << index_file.string()
                << std::endl;

            return false;
        }

    private:
        static bool ParseSingleFile(
            const std::string& file_path,
            const std::string& shard_name,
            std::unordered_map<std::string, TensorMeta>& out_tensors)
        {
            std::ifstream file(
                file_path,
                std::ios::binary
            );

            if (!file.is_open())
            {
                std::cerr
                    << "[SafeTensorHeaderParser] Failed to open: "
                    << file_path
                    << std::endl;

                return false;
            }

            uint64_t header_size = 0;

            file.read(
                reinterpret_cast<char*>(&header_size),
                sizeof(uint64_t)
            );

            if (header_size == 0)
            {
                std::cerr
                    << "[SafeTensorHeaderParser] Invalid header size"
                    << std::endl;

                return false;
            }

            std::string header_json;
            header_json.resize(header_size);

            file.read(
                header_json.data(),
                static_cast<std::streamsize>(header_size)
            );

            nlohmann::json j =
                nlohmann::json::parse(
                    header_json
                );

            for (
                auto it = j.begin();
                it != j.end();
                ++it
            )
            {
                if (it.key() == "__metadata__")
                {
                    continue;
                }

                TensorMeta meta;

                meta.dtype =
                    it.value().value(
                        "dtype",
                        "UNKNOWN"
                    );

                meta.shape =
                    it.value().value(
                        "shape",
                        std::vector<int64_t>{}
                    );

                meta.data_offsets =
                    it.value().value(
                        "data_offsets",
                        std::vector<int64_t>{}
                    );

                meta.shard_file =
                    shard_name;

                out_tensors[it.key()] =
                    meta;
            }

            return true;
        }

        static bool ParseShardedIndex(
            const std::string& model_dir,
            const std::string& index_path,
            std::unordered_map<std::string, TensorMeta>& out_tensors)
        {
            std::ifstream file(
                index_path
            );

            if (!file.is_open())
            {
                std::cerr
                    << "[SafeTensorHeaderParser] Failed to open index: "
                    << index_path
                    << std::endl;

                return false;
            }

            nlohmann::json j;
            file >> j;

            if (
                !j.contains("weight_map") ||
                !j["weight_map"].is_object()
            )
            {
                std::cerr
                    << "[SafeTensorHeaderParser] Invalid index.json: missing weight_map"
                    << std::endl;

                return false;
            }

            std::unordered_map<
                std::string,
                std::string
            > weight_to_shard;

            for (
                auto it = j["weight_map"].begin();
                it != j["weight_map"].end();
                ++it
            )
            {
                weight_to_shard[it.key()] =
                    it.value().get<std::string>();
            }

            std::unordered_map<
                std::string,
                bool
            > parsed_shards;

            for (const auto& [weight_name, shard_name] : weight_to_shard)
            {
                if (parsed_shards.count(shard_name))
                {
                    continue;
                }

                const std::string shard_path =
                    model_dir +
                    "/" +
                    shard_name;

                if (
                    !ParseSingleFile(
                        shard_path,
                        shard_name,
                        out_tensors
                    )
                )
                {
                    return false;
                }

                parsed_shards[shard_name] = true;
            }

            std::cout
                << "[SafeTensorHeaderParser] Parsed tensor count: "
                << out_tensors.size()
                << std::endl;

            return true;
        }
    };
}