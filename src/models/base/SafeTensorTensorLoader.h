#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <stdexcept>

#include <torch/torch.h>

#include "models/base/SafeTensorHeaderParser.h"

namespace mllm
{
    class SafeTensorTensorLoader
    {
    public:
        // Actual safetensors tensor byte loading.
        // Decodes dtype → torch::Tensor. Currently supports BF16 / F16 / F32.
        // Buffer bytes are copied into an owning tensor via clone(), so the
        // returned tensor does not depend on caller-side memory.
        static torch::Tensor LoadTensor(
            const std::string& model_dir,
            const std::string& tensor_name,
            const std::unordered_map<std::string, TensorMeta>& tensor_map)
        {
            auto it = tensor_map.find(tensor_name);
            if (it == tensor_map.end())
            {
                throw std::runtime_error("Tensor not found: " + tensor_name);
            }

            const auto& meta = it->second;
            const std::string file_path = model_dir + "/model.safetensors";

            std::ifstream file(file_path, std::ios::binary);
            if (!file.is_open())
            {
                throw std::runtime_error("Failed to open safetensors file.");
            }

            // safetensors layout:
            //   [ 8 bytes little-endian header_size ]
            //   [ header_size bytes JSON header     ]
            //   [ raw tensor bytes (per-tensor offsets are relative here) ]
            uint64_t header_size = 0;
            file.read(reinterpret_cast<char*>(&header_size), sizeof(uint64_t));

            const int64_t data_start = 8 + static_cast<int64_t>(header_size);
            const int64_t begin = data_start + meta.data_offsets[0];
            const int64_t end = data_start + meta.data_offsets[1];
            const int64_t byte_size = end - begin;

            if (byte_size <= 0)
            {
                throw std::runtime_error(
                    "Invalid tensor byte size for: " + tensor_name);
            }

            file.seekg(begin, std::ios::beg);

            std::vector<char> buffer(static_cast<size_t>(byte_size));
            file.read(buffer.data(), byte_size);

            const auto dtype = ResolveDtype(meta.dtype);
            const size_t elem_size = DtypeSize(meta.dtype);

            int64_t num_elements = 1;
            for (auto d : meta.shape) num_elements *= d;

            if (static_cast<int64_t>(elem_size) * num_elements != byte_size)
            {
                throw std::runtime_error(
                    "Byte size mismatch for tensor: " + tensor_name);
            }

            // from_blob borrows memory; clone() copies into owning storage so
            // the returned tensor outlives the local buffer.
            auto tensor = torch::from_blob(
                buffer.data(),
                meta.shape,
                torch::TensorOptions().dtype(dtype)
            ).clone();

            std::cout << "[SafeTensorTensorLoader] Loaded tensor: "
                      << tensor_name
                      << "  dtype=" << meta.dtype
                      << "  bytes=" << byte_size
                      << "  shape=" << tensor.sizes()
                      << std::endl;

            return tensor;
        }

    private:
        static torch::ScalarType ResolveDtype(const std::string& name)
        {
            if (name == "BF16") return torch::kBFloat16;
            if (name == "F16")  return torch::kFloat16;
            if (name == "F32")  return torch::kFloat32;
            if (name == "F64")  return torch::kFloat64;
            if (name == "I32")  return torch::kInt32;
            if (name == "I64")  return torch::kInt64;
            if (name == "U8")   return torch::kUInt8;
            if (name == "I8")   return torch::kInt8;

            throw std::runtime_error("Unsupported safetensors dtype: " + name);
        }

        static size_t DtypeSize(const std::string& name)
        {
            if (name == "BF16") return 2;
            if (name == "F16")  return 2;
            if (name == "F32")  return 4;
            if (name == "F64")  return 8;
            if (name == "I32")  return 4;
            if (name == "I64")  return 8;
            if (name == "U8")   return 1;
            if (name == "I8")   return 1;

            throw std::runtime_error("Unsupported safetensors dtype: " + name);
        }
    };
}
