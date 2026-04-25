#include <iostream>
#include <memory>

#include "models/llama/LlamaRunner.h"

int main()
{
    std::cout << "===== mLLM Runtime Start =====" << std::endl;

    auto runner = std::make_shared<mllm::LlamaRunner>();

    const std::string model_path = "../models/TinyLlama";

    if (!runner->Load(model_path))
    {
        std::cerr << "Failed to load model." << std::endl;
        return -1;
    }

    runner->InitKVCache(1, 512);

    // 테스트용 dummy input
    // 이후 tokenizer 연결 예정
    auto input_ids = torch::tensor({{15043, 6796, 263, 1243}}, torch::kInt64);
    auto attention_mask = torch::ones({1, 4}, torch::kInt64);

    try
    {
        auto logits = runner->Forward(input_ids, attention_mask);

        std::cout << "Forward success!" << std::endl;
        std::cout << "Logits shape: " << logits.sizes() << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Forward failed:\n" << e.what() << std::endl;
        return -1;
    }

    std::cout << "===== mLLM Runtime End =====" << std::endl;

    return 0;
}
