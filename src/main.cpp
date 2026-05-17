// src/main.cpp

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <exception>

#include "models/base/ModelRunnerFactory.h"
#include "models/base/GenerateOptions.h"
#include "models/base/GenerateResult.h"
#include "tokenizer/ITokenizer.h"
#include "serving/Scheduler.h"
#include "serving/HttpServer.h"

// Trim history from the front when it exceeds this many tokens.
constexpr size_t MAX_CONTEXT_TOKENS = 2048;

static const std::string DEFAULT_MODEL_PATH = "../models/Qwen3-8B-FP16";
static const std::string DEFAULT_PARITY_DIR = "../scripts/parity";

static const std::string SYSTEM_PROMPT =
    "You are a helpful assistant.\n"
    "Always respond in the same language as the user.\n"
    "Be concise and direct.";

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

static std::string ParseOptionValue(
    int argc,
    char* argv[],
    const std::string& name,
    const std::string& default_value)
{
    const std::string prefix = name + "=";
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == name && i + 1 < argc)
            return argv[i + 1];
        if (arg.rfind(prefix, 0) == 0)
            return arg.substr(prefix.size());
    }
    return default_value;
}

static bool HasFlag(int argc, char* argv[], const std::string& flag)
{
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == flag) return true;
    return false;
}

// --parity: fixed forward pass used by regression_test.py
static int RunParityCheck(
    const std::string& model_path,
    const std::string& parity_dir)
{
    auto bundle = mllm::ModelRunnerFactory::Create(model_path);
    bundle.runner->SetParityMode(true);
    bundle.runner->SetParityReferenceDir(parity_dir);

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

    auto logits = bundle.runner->Forward(ids, mask);
    auto last_logits = logits.index({
        0,
        logits.size(1) - 1
    }).to(torch::kFloat32);

    std::cout
        << "last-token argmax token_id: "
        << torch::argmax(last_logits, -1).item<int64_t>()
        << std::endl;

    return 0;
}

int main(int argc, char* argv[])
{
    try
    {
    const std::string model_path = ParseModelPath(argc, argv);

    // --------------------------------
    // --parity
    // --------------------------------
    if (HasFlag(argc, argv, "--parity"))
    {
        return RunParityCheck(
            model_path,
            ParseOptionValue(argc, argv, "--parity-dir", DEFAULT_PARITY_DIR)
        );
    }

    // --------------------------------
    // --serve: start HTTP inference server
    // --------------------------------
    if (HasFlag(argc, argv, "--serve"))
    {
        const int port = std::stoi(
            ParseOptionValue(argc, argv, "--port", "8080"));

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

        bundle.runner->InitKVCache(1, bundle.runner->GetConfig().max_position_embeddings);

        mllm::Scheduler scheduler(*bundle.runner);
        scheduler.Start();

        mllm::HttpServer server(scheduler, *bundle.tokenizer, port);
        server.Run();      // blocks; Ctrl-C is the only exit signal in v1 (no SIGINT handler)

        scheduler.Stop();
        return 0;
    }

    // --------------------------------
    // --tokenize: read one line from stdin and print token IDs
    // --------------------------------
    if (HasFlag(argc, argv, "--tokenize"))
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

    // --------------------------------
    // --tokenize-batch: read stdin lines and print one token-ID line per input
    // --------------------------------
    if (HasFlag(argc, argv, "--tokenize-batch"))
    {
        auto bundle = mllm::ModelRunnerFactory::Create(model_path);

        if (!bundle.tokenizer->Load(model_path))
        {
            std::cerr << "Tokenizer load failed." << std::endl;
            return -1;
        }

        std::string text;
        while (std::getline(std::cin, text))
        {
            const auto ids = bundle.tokenizer->Encode(text);
            for (size_t j = 0; j < ids.size(); ++j)
            {
                if (j > 0) std::cout << ' ';
                std::cout << ids[j];
            }
            std::cout << '\n';
        }
        return 0;
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
    options.max_new_tokens     = 1024;
    options.temperature        = 0.0f;
    options.top_k              = 1;
    options.top_p              = 1.0f;
    options.use_greedy         = true;
    options.repetition_penalty = 1.0f;
    options.eos_token_id       = bundle.tokenizer->GetEOSTokenId();
    options.enable_thinking    = bundle.tokenizer->SupportsThinking();

    std::cout
        << "[GenerateOptions]"
        << " max_new_tokens=" << options.max_new_tokens
        << " greedy=" << (options.use_greedy ? "true" : "false")
        << " eos=" << options.eos_token_id
        << std::endl;

    // --------------------------------
    // Token-based conversation history
    // Each turn appends to history_ids so prior context is visible to the
    // model. Older tokens are trimmed from the front when the window fills.
    // --------------------------------

    std::vector<int64_t> history_ids;
    bool first_turn = true;

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

        if (first_turn)
        {
            const std::string prompt =
                bundle.tokenizer->BuildChatPrompt(SYSTEM_PROMPT, user_text, options.enable_thinking);
            history_ids = bundle.tokenizer->Encode(prompt);
            first_turn = false;
        }
        else
        {
            const std::string cont =
                bundle.tokenizer->BuildNextUserTurn(user_text, options.enable_thinking);
            const auto cont_ids = bundle.tokenizer->Encode(cont);
            history_ids.insert(history_ids.end(), cont_ids.begin(), cont_ids.end());
        }

        // Trim oldest tokens when the context window is full
        if (history_ids.size() > MAX_CONTEXT_TOKENS)
        {
            const size_t excess = history_ids.size() - MAX_CONTEXT_TOKENS;
            history_ids.erase(history_ids.begin(), history_ids.begin() + static_cast<ptrdiff_t>(excess));
            std::cout << "[History trimmed to " << MAX_CONTEXT_TOKENS << " tokens]" << std::endl;
        }

        std::cout << "Input token count: " << history_ids.size() << std::endl;

        if (history_ids.empty())
        {
            std::cout << "Tokenization failed." << std::endl;
            continue;
        }

        mllm::GenerateResult result =
            bundle.runner->Generate(history_ids, options);

        if (result.tokens.empty())
        {
            std::cout << "Generation produced no tokens." << std::endl;
            continue;
        }

        const std::string assistant_text =
            bundle.tokenizer->Decode(result.tokens);

        std::cout << "\nAssistant: " << assistant_text << std::endl;

        // Append generated tokens excluding EOS: BuildNextUserTurn already
        // adds the turn separator, so keeping EOS would double <|im_end|>.
        auto hist_end = result.tokens.end();
        if (!result.tokens.empty() &&
            result.tokens.back() == static_cast<int64_t>(options.eos_token_id))
            --hist_end;
        history_ids.insert(history_ids.end(), result.tokens.begin(), hist_end);
    }

    std::cout << "\n===== mLLM Runtime End =====" << std::endl;
    return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return -1;
    }
}
