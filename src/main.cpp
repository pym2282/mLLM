// src/main.cpp
// Persistent Chat History 적용 버전

#include <iostream>
#include <string>
#include <memory>
#include <vector>

#include "models/llama/LlamaRunner.h"
#include "models/base/GenerateOptions.h"

#include "tokenizer/ITokenizer.h"
#include "tokenizer/LlamaTokenizer.h"

constexpr size_t MAX_HISTORY_CHARS = 8000;

// --parity: fixed forward pass for regression_test.py
// matches verify_full_forward.py input: [[15043, 6796, 263, 1243]]
static int RunParityCheck()
{
    mllm::LlamaRunner runner;

    if (!runner.Load("../models/TinyLlama"))
    {
        std::cerr << "Parity: model load failed." << std::endl;
        return -1;
    }

    auto ids = torch::tensor(
        std::vector<int64_t>{15043, 6796, 263, 1243},
        torch::kInt64
    ).unsqueeze(0);

    auto mask = torch::ones({1, 4}, torch::kInt64);

    runner.Forward(ids, mask);
    return 0;
}

int main(int argc, char* argv[])
{
    if (argc > 1 && std::string(argv[1]) == "--parity")
    {
        return RunParityCheck();
    }

    if (argc > 1 && std::string(argv[1]) == "--tokenize")
    {
        // Text is read from stdin (one line) so the caller can send
        // UTF-8 bytes directly via pipe, avoiding Windows argv
        // code-page conversion that corrupts non-ASCII characters.
        std::unique_ptr<mllm::ITokenizer> tok =
            std::make_unique<mllm::LlamaTokenizer>();

        if (!tok->Load("../models/TinyLlama"))
        {
            std::cerr << "Tokenizer load failed." << std::endl;
            return -1;
        }

        std::string text;
        std::getline(std::cin, text);

        const auto ids = tok->Encode(text);

        for (size_t i = 0; i < ids.size(); ++i)
        {
            if (i > 0) std::cout << ' ';
            std::cout << ids[i];
        }
        std::cout << std::endl;

        return 0;
    }


    std::cout
        << "===== mLLM Runtime Start ====="
        << std::endl;

    // --------------------------------
    // Model Runner
    // --------------------------------

    mllm::LlamaRunner runner;

    if (!runner.Load("../models/TinyLlama"))
    {
        std::cout
            << "Failed to load model."
            << std::endl;

        return -1;
    }

    // --------------------------------
    // Tokenizer
    // --------------------------------

    std::unique_ptr<mllm::ITokenizer> tokenizer =
        std::make_unique<mllm::LlamaTokenizer>();

    if (!tokenizer->Load("../models/TinyLlama"))
    {
        std::cout
            << "Failed to load tokenizer."
            << std::endl;

        return -1;
    }

    // --------------------------------
    // Generation Options
    // --------------------------------

    mllm::GenerateOptions options;

    options.max_new_tokens = 64;
    options.temperature = 0.8f;
    options.top_k = 40;
    options.top_p = 0.9f;
    options.use_greedy = false;
    options.repetition_penalty = 1.1f;
    options.eos_token_id = tokenizer->GetEOSTokenId();

    // --------------------------------
    // Persistent Chat History
    // --------------------------------

    std::string chat_history =
        "You are a helpful assistant.\n\n";

    // --------------------------------
    // Interactive CLI
    // --------------------------------

    while (true)
    {
        std::cout
            << "\nUser (q to quit): ";

        std::string user_text;

        if (!std::getline(std::cin, user_text))
        {
            break;
        }

        if (user_text == "q")
        {
            break;
        }

        if (user_text.empty())
        {
            continue;
        }

        // --------------------------------
        // Build conversation prompt
        // --------------------------------

        chat_history +=
            "User: " + user_text + "\n";

        chat_history +=
            "Assistant: ";

        std::cout
            << "\n[Prompt]\n"
            << chat_history
            << std::endl;

        // --------------------------------
        // Encode full history
        // --------------------------------

        auto input_ids =
            tokenizer->Encode(chat_history);

        if (input_ids.empty())
        {
            std::cout
                << "Tokenization failed."
                << std::endl;

            continue;
        }

        std::cout
            << "Input token count: "
            << input_ids.size()
            << std::endl;

        // --------------------------------
        // Generate response
        // --------------------------------

        auto generated =
            runner.Generate(
                input_ids,
                options
            );

        if (generated.empty())
        {
            std::cout
                << "Generation failed."
                << std::endl;

            continue;
        }

        // --------------------------------
        // Decode response
        // --------------------------------

        auto assistant_text =
            tokenizer->Decode(generated);

        std::cout
            << "\nAssistant: "
            << assistant_text
            << std::endl;

        // --------------------------------
        // Save assistant response
        // --------------------------------

        chat_history +=
            assistant_text + "\n";
        // --------------------------------
        // Sliding Window
        // --------------------------------

        if (chat_history.size() > MAX_HISTORY_CHARS)
        {
            chat_history =
                chat_history.substr(
                    chat_history.size() - MAX_HISTORY_CHARS
                );

            std::cout
                << "[History trimmed]"
                << std::endl;
        }
    }

    std::cout
        << "\n===== mLLM Runtime End ====="
        << std::endl;

    return 0;
}