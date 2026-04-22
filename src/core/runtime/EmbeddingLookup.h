#pragma once

#include <torch/torch.h>
#include <iostream>

namespace mllm
{
    class EmbeddingLookup
    {
    public:
        // input_ids: [batch, seq]
        // embedding_weight: [vocab_size, hidden_size]
        // output: [batch, seq, hidden_size]
        static torch::Tensor Forward(
            const torch::Tensor& input_ids,
            const torch::Tensor& embedding_weight)
        {
            if (embedding_weight.dim() != 2)
            {
                throw std::runtime_error(
                    "Embedding weight must be 2D tensor.");
            }

            auto output = torch::nn::functional::embedding(
                input_ids,
                embedding_weight
            );

            std::cout << "[EmbeddingLookup] Success" << std::endl;
            std::cout << "  input_ids shape: " << input_ids.sizes() << std::endl;
            std::cout << "  output shape: " << output.sizes() << std::endl;

            return output;
        }
    };
}
