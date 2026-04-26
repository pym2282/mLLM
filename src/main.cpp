// src/main.cpp

#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <memory>
#include <array>
#include <cstdio>

#include "models/llama/LlamaRunner.h"
#include "models/base/GenerateOptions.h"

static std::string RunCommand(
    const std::string& command)
{
    std::array<char, 512> buffer{};
    std::string result;

#ifdef _WIN32
    FILE* pipe = _popen(command.c_str(), "r");
#else
    FILE* pipe = popen(command.c_str(), "r");
#endif

    if (pipe == nullptr)
    {
        return "";
    }

    while (true)
    {
        if (
            fgets(
                buffer.data(),
                static_cast<int>(buffer.size()),
                pipe
            ) == nullptr
        )
        {
            break;
        }

        result += buffer.data();
    }

#ifdef _WIN32
    int rc = _pclose(pipe);
#else
    int rc = pclose(pipe);
#endif

    (void)rc;

    return result;
}

static std::vector<int64_t> ParseTokenIds(const std::string& text)
{
    std::vector<int64_t> ids;
    std::stringstream ss(text);

    int64_t token_id;
    while (ss >> token_id)
    {
        ids.push_back(token_id);
    }

    return ids;
}

static std::vector<int64_t> EncodeText(const std::string& text)
{
    std::string command =
        "python ../scripts/tokenizer_helper.py encode \"" +
        text + "\"";

    std::string output = RunCommand(command);

    return ParseTokenIds(output);
}

static std::string DecodeTokens(
    const std::vector<int64_t>& tokens)
{
    std::stringstream ss;

    ss << "python ../scripts/tokenizer_helper.py decode";

    for (auto token : tokens)
    {
        ss << " " << token;
    }

    return RunCommand(ss.str());
}

int main()
{
    std::cout
        << "===== mLLM Runtime Start ====="
        << std::endl;

    mllm::LlamaRunner runner;

    if (!runner.Load("../models/TinyLlama"))
    {
        std::cout
            << "Failed to load model."
            << std::endl;
        return -1;
    }

    mllm::GenerateOptions options;
    options.max_new_tokens = 32;
    options.temperature = 1.0f;
    options.top_k = 40;
    options.use_greedy = false;

    while (true)
    {
        std::cout
            << "\nUser text (q to quit): ";

        std::string user_text;
        std::getline(std::cin, user_text);

        if (user_text == "q")
        {
            break;
        }

        if (user_text.empty())
        {
            continue;
        }

        auto input_ids =
            EncodeText(user_text);

        if (input_ids.empty())
        {
            std::cout
                << "Tokenization failed."
                << std::endl;
            continue;
        }

        std::cout << "Input token ids: ";
        for (auto id : input_ids)
        {
            std::cout << id << " ";
        }
        std::cout << std::endl;

        auto generated =
            runner.Generate(
                input_ids,
                options
            );

        std::cout << "Generated token ids: ";
        for (auto token : generated)
        {
            std::cout << token << " ";
        }
        std::cout << std::endl;

        auto decoded =
            DecodeTokens(generated);

        std::cout
            << "Model output: "
            << decoded
            << std::endl;
    }

    std::cout
        << "===== mLLM Runtime End ====="
        << std::endl;

    return 0;
}