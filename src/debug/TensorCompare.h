// src/debug/TensorCompare.h
// txt 기반 parity compare 버전
// torch::load 제거
// ifstream 기반 plain text compare

#pragma once

#include <torch/torch.h>

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cmath>

namespace mllm
{
    class TensorCompare
    {
    public:
        static bool CompareTensor(
            const torch::Tensor& actual,
            const std::string& reference_path,
            float atol = 1e-2f,
            float rtol = 1e-2f)
        {
            try
            {
                auto actual_f =
                    actual.detach()
                        .to(torch::kFloat32)
                        .cpu()
                        .flatten();

                std::vector<float> reference_values;

                {
                    std::ifstream file(reference_path);

                    if (!file.is_open())
                    {
                        std::cerr
                            << "[Parity] Failed to open file: "
                            << reference_path
                            << std::endl;
                        return false;
                    }

                    float value = 0.0f;

                    while (file >> value)
                    {
                        reference_values.push_back(value);
                    }
                }

                if (reference_values.empty())
                {
                    std::cerr
                        << "[Parity] Empty reference file: "
                        << reference_path
                        << std::endl;
                    return false;
                }

                if (actual_f.numel() !=
                    static_cast<int64_t>(
                        reference_values.size()))
                {
                    std::cerr
                        << "[Parity] Size mismatch\n"
                        << "actual numel : "
                        << actual_f.numel()
                        << "\nref numel    : "
                        << reference_values.size()
                        << "\nfile         : "
                        << reference_path
                        << std::endl;

                    return false;
                }

                float max_diff = 0.0f;
                double mean_diff = 0.0;
                bool pass = true;

                auto accessor =
                    actual_f.accessor<float, 1>();

                for (int64_t i = 0;
                     i < actual_f.numel();
                     ++i)
                {
                    const float a =
                        accessor[i];

                    const float b =
                        reference_values[i];

                    const float abs_diff =
                        std::fabs(a - b);

                    const float allowed =
                        atol + rtol * std::fabs(b);

                    if (abs_diff > allowed)
                    {
                        pass = false;
                    }

                    if (abs_diff > max_diff)
                    {
                        max_diff = abs_diff;
                    }

                    mean_diff += abs_diff;
                }

                mean_diff /=
                    static_cast<double>(
                        actual_f.numel());

                if (pass)
                {
                    std::cout
                        << "[Parity] PASS : "
                        << reference_path
                        << "\n"
                        << "max diff  = "
                        << max_diff
                        << "\n"
                        << "mean diff = "
                        << mean_diff
                        << std::endl;
                }
                else
                {
                    std::cerr
                        << "[Parity] FAIL : "
                        << reference_path
                        << "\n"
                        << "max diff  = "
                        << max_diff
                        << "\n"
                        << "mean diff = "
                        << mean_diff
                        << std::endl;
                }

                return pass;
            }
            catch (const std::exception& e)
            {
                std::cerr
                    << "[Parity] Exception while comparing tensor\n"
                    << "file: "
                    << reference_path
                    << "\n"
                    << e.what()
                    << std::endl;

                return false;
            }
        }
    };
}