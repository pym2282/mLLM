// src/main.cpp

#include <iostream>
#include <string>
#include <memory>
#include <vector>

#include "models/base/ModelRunnerFactory.h"
#include "models/base/GenerateOptions.h"
#include "tokenizer/ITokenizer.h"

constexpr size_t MAX_HISTORY_CHARS = 8000;

static const std::string DEFAULT_MODEL_PATH = "../models/Qwen3-8B-FP16";

// Extract the first non-flag argument as model path, or return default.
static std::string ParseModelPath(int argc, char* argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a.rfind("--", 0) != 0)
            return a;
    }
    return DEFAULT_MODEL_PATH;
}

// --parity: fixed forward pass used by regression_test.py
static int RunParityCheck(const std::string& model_path)
{
    auto bundle = mllm::ModelRunnerFactory::Create(model_path);

    if (!bundle.runner->Load(model_path))
    {
        std::cerr << "Parity: model load failed." << std::endl;
        return -1;
    }

    auto ids = torch::tensor(
        std::vector<int64_t>{15043, 6796, 263, 1243},
        torch::kInt64
    ).unsqueeze(0);

    auto mask = torch::ones({1, 4}, torch::kInt64);

    bundle.runner->Forward(ids, mask);
    return 0;
}

int main(int argc, char* argv[])
{
    const std::string model_path = ParseModelPath(argc, argv);

    // --------------------------------
    // --parity
    // --------------------------------
    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == "--parity")
            return RunParityCheck(model_path);
    }

    // --------------------------------
    // --tokenize: read one line from stdin and print token IDs
    // --------------------------------
    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == "--tokenize")
        {
            auto bundle = mllm::ModelRunnerFactory::Create(model_path);

            if (!bundle.tokenizer->Load(model_path))
            {
                std::cerr << "Tokenizer load failed." << std::endl;
                return -1;
            }

            std::string text;
            std::getline(std::cin, text);

            const auto ids = bundle.tokenizer->Encode(text);
            for (size_t j = 0; j < ids.size(); ++j)
            {
                if (j > 0) std::cout << ' ';
                std::cout << ids[j];
            }
            std::cout << std::endl;
            return 0;
        }
    }

    // --------------------------------
    // Interactive mode
    // --------------------------------

    std::cout << "===== mLLM Runtime Start =====" << std::endl;

    auto bundle = mllm::ModelRunnerFactory::Create(model_path);

    if (!bundle.runner->Load(model_path))
    {
        std::cerr << "Failed to load model." << std::endl;
        return -1;
    }

    if (!bundle.tokenizer->Load(model_path))
    {
        std::cerr << "Failed to load tokenizer." << std::endl;
        return -1;
    }

    std::cout
        << "[Tokenizer] EOS token id = "
        << bundle.tokenizer->GetEOSTokenId()
        << std::endl;

    // --------------------------------
    // Generation Options
    // --------------------------------

    mllm::GenerateOptions options;
    options.max_new_tokens    = 16;
    options.temperature       = 0.0f;
    options.top_k             = 1;
    options.top_p             = 1.0f;
    options.use_greedy        = true;
    options.repetition_penalty = 1.0f;
    options.eos_token_id      = bundle.tokenizer->GetEOSTokenId();

    std::cout
        << "[GenerateOptions]"
        << " max_new_tokens=" << options.max_new_tokens
        << " greedy=" << (options.use_greedy ? "true" : "false")
        << " eos=" << options.eos_token_id
        << std::endl;

    // --------------------------------
    // Persistent Chat History
    // --------------------------------

    std::string chat_history;

    // --------------------------------
    // Interactive CLI
    // --------------------------------

    while (true)
    {
        std::cout << "\nUser (q to quit): ";

        std::string user_text;
        if (!std::getline(std::cin, user_text))
            break;

        if (user_text == "q")
            break;

        if (user_text.empty())
            continue;

        chat_history = bundle.tokenizer->BuildChatPrompt(
            "You are a concise assistant.\n"
            "Answer with only the final answer.\n"
            "Do not explain.\n"
            "Do not repeat.\n"
            "One short sentence only.",
            user_text
        );

        std::cout << "\n[Prompt]\n" << chat_history << std::endl;

        auto input_ids = bundle.tokenizer->Encode(chat_history);

        if (input_ids.empty())
        {
            std::cout << "Tokenization failed." << std::endl;
            continue;
        }

        std::cout << "Input token count: " << input_ids.size() << std::endl;

        auto generated = bundle.runner->Generate(input_ids, options);

        if (generated.empty())
        {
            std::cout << "Generation failed." << std::endl;
            continue;
        }

        auto assistant_text = bundle.tokenizer->Decode(generated);

        std::cout << "\nAssistant: " << assistant_text << std::endl;

        chat_history += assistant_text + "\n";

        if (chat_history.size() > MAX_HISTORY_CHARS)
        {
            chat_history = chat_history.substr(chat_history.size() - MAX_HISTORY_CHARS);
            std::cout << "[History trimmed]" << std::endl;
        }
    }

    std::cout << "\n===== mLLM Runtime End =====" << std::endl;
    return 0;
}
