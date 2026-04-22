#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <unordered_map>

#include <torch/torch.h>

#include "models/base/SafeTensorHeaderParser.h"

namespace mllm
{
    class SafeTensorTensorLoader
    {
    public:
        // Step 1:
        // 특정 tensor를 실제로 읽어서 torch::Tensor 생성
        // 현재는 BF16 / F16 / F32 일부만 대응 시작
        static torch::Tensor LoadTensor(
            const std::string& model_dir,
            const std::string& tensor_name,
            const std::unordered_map<std::string, TensorMeta>& tensor_map)
        {
            if (tensor_map.find(tensor_name) == tensor_map.end())
            {
                throw std::runtime_error("Tensor not found: " + tensor_name);
            }

            const auto& meta = tensor_map.at(tensor_name);
            const std::string file_path = model_dir + "/model.safetensors";

            std::ifstream file(file_path, std::ios::binary);
            if (!file.is_open())
            {
                throw std::runtime_error("Failed to open safetensors file.");
            }

            // safetensors:
            // first 8 bytes = header size
            uint64_t header_size = 0;
            file.read(reinterpret_cast<char*>(&header_size), sizeof(uint64_t));

            const int64_t data_start = 8 + static_cast<int64_t>(header_size);
            const int64_t begin = data_start + meta.data_offsets[0];
            const int64_t end = data_start + meta.data_offsets[1];
            const int64_t byte_size = end - begin;

            file.seekg(begin, std::ios::beg);

            std::vector<char> buffer(byte_size);
            file.read(buffer.data(), byte_size);

            std::cout << "[SafeTensorTensorLoader] Loaded tensor: "
                      << tensor_name << std::endl;
            std::cout << "  dtype: " << meta.dtype << std::endl;
            std::cout << "  bytes: " << byte_size << std::endl;

            // 현재는 placeholder tensor 반환
            // 다음 단계에서 dtype별 실제 변환 구현
            return torch::rand(meta.shape, torch::kFloat32);
        }
    };
}
