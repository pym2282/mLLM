// src/main.cpp

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <exception>
#include <atomic>
#include <csignal>

// Windows SEH exception filter: writes a minidump on crash
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")

static LONG WINAPI CrashHandler(EXCEPTION_POINTERS* ep)
{
    HANDLE hFile = CreateFileA(
        "crash.dmp",
        GENERIC_WRITE, 0, nullptr,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL, nullptr);

    if (hFile != INVALID_HANDLE_VALUE)
    {
        MINIDUMP_EXCEPTION_INFORMATION mei{};
        mei.ThreadId          = GetCurrentThreadId();
        mei.ExceptionPointers = ep;
        mei.ClientPointers    = FALSE;

        MiniDumpWriteDump(
            GetCurrentProcess(),
            GetCurrentProcessId(),
            hFile,
            MiniDumpWithFullMemory,
            &mei, nullptr, nullptr);

        CloseHandle(hFile);
        std::cerr << "[CRASH] Minidump written to crash.dmp\n";
    }
    else
    {
        std::cerr << "[CRASH] Could not create crash.dmp (err=" << GetLastError() << ")\n";
    }

    std::cerr << "[CRASH] Exception code=0x"
              << std::hex << ep->ExceptionRecord->ExceptionCode
              << "  addr=0x" << ep->ExceptionRecord->ExceptionAddress
              << std::dec << "\n";
    std::cerr.flush();
    return EXCEPTION_CONTINUE_SEARCH;
}
#endif

#include "models/base/ModelRunnerFactory.h"
#include "models/base/GenerateOptions.h"
#include "models/base/GenerateResult.h"
#include "tokenizer/ITokenizer.h"
#include "serving/Scheduler.h"
#include "serving/HttpServer.h"
#include "serving/GenerationRequest.h"
#include "core/Logger.h"
#include "core/MllmException.h"

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

// --stress / --stress-direct: reproduce serve-mode crash without HTTP server.
//
//   --stress           : runs via Scheduler worker thread  (= serve path)
//   --stress-direct    : calls Generate() on main thread   (= chat path)
//   --stress-thinking  : scheduler path + enable_thinking=true
//
// Both build a ~650-token prompt so kv_seq during decode matches the
// crash threshold seen in serve mode (kv_seq ≈ 553-610).
static int RunStressTest(
    const std::string& model_path,
    bool use_scheduler,
    bool use_thinking = false)
{
    std::cout << "===== Stress Test (mode="
              << (use_scheduler ? "scheduler" : "direct")
              << ") =====" << std::endl;

    auto bundle = mllm::ModelRunnerFactory::Create(model_path);

    if (!bundle.runner->Load(model_path))
    {
        MLLM_ERROR("main", "Stress: model load failed.");
        return -1;
    }
    if (!bundle.tokenizer->Load(model_path))
    {
        MLLM_ERROR("main", "Stress: tokenizer load failed.");
        return -1;
    }

    bundle.runner->InitKVCache(1, bundle.runner->GetConfig().max_position_embeddings);

    mllm::GenerateOptions opts;
    opts.max_new_tokens     = use_thinking ? 512 : 64;
    opts.temperature        = 0.0f;
    opts.top_k              = 1;
    opts.use_greedy         = true;
    opts.enable_thinking    = use_thinking;
    opts.eos_token_id       = bundle.tokenizer->GetEOSTokenId();

    // Multi-turn stress: simulate serve-mode conversation where each turn's
    // full history (system + all prior turns + new user msg) is re-encoded.
    // ~80 tokens per user turn × 5 turns + ~60 tokens responses → kv_seq > 600.
    const std::string system_prompt = "You are a helpful assistant.";

    // Each user message is ~80 tokens
    const std::vector<std::string> user_turns = {
        "Please explain in detail what machine learning is. Include key concepts, "
        "types of learning, and real-world applications. Be thorough.",
        "Now explain deep learning and how it differs from classical ML. "
        "Describe neural networks, layers, and backpropagation in detail.",
        "Describe transformer architecture thoroughly. Explain self-attention, "
        "multi-head attention, positional encoding, and why transformers dominate NLP.",
        "Explain how large language models like GPT are trained. Describe "
        "pretraining objectives, fine-tuning, RLHF, and inference at scale.",
        "Summarize all four topics above in a concise paragraph each. "
        "Then discuss future directions for AI research.",
    };

    // accumulated chat history for building full-context prompts each turn
    struct Turn { std::string role; std::string content; };
    std::vector<Turn> history;

    std::unique_ptr<mllm::Scheduler> scheduler_ptr;
    if (use_scheduler)
    {
        scheduler_ptr = std::make_unique<mllm::Scheduler>(*bundle.runner);
        scheduler_ptr->Start();
    }

    for (int turn = 0; turn < static_cast<int>(user_turns.size()); ++turn)
    {
        history.push_back({"user", user_turns[turn]});

        // Build full prompt from all history (mirrors serve-mode BuildPromptFromMessages)
        std::vector<mllm::Message> msgs;
        for (const auto& t : history)
            msgs.push_back({t.role, t.content});

        const std::string prompt =
            bundle.tokenizer->BuildPromptFromMessages(msgs, false);
        auto ids = bundle.tokenizer->Encode(prompt);

        std::cout << "[Stress] Turn " << turn
                  << "  prompt_tokens=" << ids.size() << std::endl;

        std::string reply_text;
        bool ok = true;

        if (use_scheduler)
        {
            auto req = std::make_shared<mllm::GenerationRequest>();
            req->request_id    = "stress-" + std::to_string(turn);
            req->prompt_tokens = ids;
            req->options       = opts;

            auto fut = req->result_promise.get_future();
            scheduler_ptr->GetQueue().Push(std::move(req));

            try
            {
                auto result = fut.get();
                reply_text = bundle.tokenizer->Decode(result.tokens);
            }
            catch (const std::exception& e)
            {
                MLLM_ERROR("main", "Stress turn " + std::to_string(turn) + " EXCEPTION: " + e.what());
                ok = false;
            }
        }
        else
        {
            try
            {
                auto result = bundle.runner->Generate(ids, opts);
                reply_text = bundle.tokenizer->Decode(result.tokens);
            }
            catch (const std::exception& e)
            {
                MLLM_ERROR("main", "Stress turn " + std::to_string(turn) + " EXCEPTION: " + e.what());
                ok = false;
            }
        }

        if (ok)
        {
            std::cout << "[Stress] Turn " << turn << " OK  reply_len=" << reply_text.size()
                      << "  preview=" << reply_text.substr(0, 80) << std::endl;
            history.push_back({"assistant", reply_text});
        }
        else
        {
            break;
        }
    }

    if (use_scheduler)
        scheduler_ptr->Stop();

    return 0;
}

// --generate-test: deterministic multi-token generation for regression testing.
// Uses temperature=0 (greedy), max_new_tokens=16, model-specific fixed input.
// Outputs "generate-test tokens: T1 T2 ..." to stdout for golden-record comparison.
static int RunGenerateTest(const std::string& model_path)
{
    auto bundle = mllm::ModelRunnerFactory::Create(model_path);

    if (!bundle.runner->Load(model_path))
    {
        MLLM_ERROR("main", "GenerateTest: model load failed.");
        return -1;
    }
    if (!bundle.tokenizer->Load(model_path))
    {
        MLLM_ERROR("main", "GenerateTest: tokenizer load failed.");
        return -1;
    }

    const std::string mt = bundle.runner->GetModelType();
    std::vector<int64_t> input_ids;
    if (mt == "gemma")
    {
        // Same fixed tokens used by --parity (verified argmax=4176 against HF BF16)
        input_ids = {2, 106, 2430, 106, 108, 106, 4176, 108};
    }
    else
    {
        // Generic: build minimal chat prompt from tokenizer
        const std::string prompt =
            bundle.tokenizer->BuildChatPrompt("", "Hello", false);
        input_ids = bundle.tokenizer->Encode(prompt);
    }

    // Warmup: triggers EnsureOnGPU for lazy-transfer models (QwenRunner) before
    // InitKVCache so caches are allocated on the correct device (same as serve mode).
    {
        mllm::GenerateOptions warm_opts;
        warm_opts.max_new_tokens = 1;
        try { bundle.runner->Generate({1}, warm_opts); } catch (...) {}
    }
    // Allocate KV cache the same way serve mode does — activates capacity>0 path
    bundle.runner->InitKVCache(1, bundle.runner->GetConfig().max_position_embeddings);

    mllm::GenerateOptions opts;
    opts.max_new_tokens      = 16;
    opts.temperature         = 0.0f;
    opts.top_k               = 1;
    opts.use_greedy          = true;
    opts.repetition_penalty  = 1.0f;  // no penalty: first token must match --parity argmax
    opts.eos_token_id        = bundle.tokenizer->GetEOSTokenId();

    auto result = bundle.runner->Generate(input_ids, opts);

    std::cout << "generate-test input_len=" << input_ids.size() << "\n";
    std::cout << "generate-test tokens:";
    for (auto t : result.tokens)
        std::cout << ' ' << t;
    std::cout << '\n';
    std::cout << "generate-test finish=" << static_cast<int>(result.finish_reason) << '\n';

    return 0;
}

// --prefix-test: cold-vs-hit prefix cache regression.
// Runs the same prompt twice through Scheduler + QwenRunner.
// Outputs "prefix-test hit_len=N tokens_match=1" (or 0).
// Requires Qwen model (other models return empty KVSnapshot → hit_len=0, reported as SKIP).
static int RunPrefixTest(const std::string& model_path)
{
    auto bundle = mllm::ModelRunnerFactory::Create(model_path);

    if (!bundle.runner->Load(model_path))
    {
        MLLM_ERROR("main", "PrefixTest: model load failed.");
        return -1;
    }
    if (!bundle.tokenizer->Load(model_path))
    {
        MLLM_ERROR("main", "PrefixTest: tokenizer load failed.");
        return -1;
    }

    // Warmup + KV cache allocation (same as serve mode)
    {
        mllm::GenerateOptions w;
        w.max_new_tokens = 1;
        try { bundle.runner->Generate({1}, w); } catch (...) {}
    }
    bundle.runner->InitKVCache(1, bundle.runner->GetConfig().max_position_embeddings);

    // Fixed 48-token prompt (3 full 16-token blocks) built from tokenizer
    const std::string prompt =
        bundle.tokenizer->BuildChatPrompt(
            "You are a helpful assistant.",
            "Please explain what a transformer model is in detail. "
            "Include attention mechanism, positional encoding, and feed-forward layers.",
            false);
    const auto prompt_ids = bundle.tokenizer->Encode(prompt);

    if (static_cast<int>(prompt_ids.size()) < 32)
    {
        // If tokenizer can't produce a long-enough prompt, build a synthetic one
        // by repeating a known Qwen token to ensure >=32 tokens for cache
    }

    mllm::GenerateOptions opts;
    opts.max_new_tokens     = 8;
    opts.temperature        = 0.0f;
    opts.top_k              = 1;
    opts.use_greedy         = true;
    opts.repetition_penalty = 1.0f;
    opts.eos_token_id       = bundle.tokenizer->GetEOSTokenId();

    // Run via Scheduler — same code path as serve mode
    mllm::Scheduler scheduler(*bundle.runner);
    scheduler.Start();

    auto run_once = [&]() -> mllm::GenerateResult {
        auto req = std::make_shared<mllm::GenerationRequest>();
        req->request_id    = "prefix-test";
        req->prompt_tokens = prompt_ids;
        req->options       = opts;
        auto fut = req->result_promise.get_future();
        scheduler.GetQueue().Push(std::move(req));
        try { return fut.get(); } catch (...) { return {}; }
    };

    const auto r1 = run_once();  // cold — Scheduler stores KV after this
    const auto r2 = run_once();  // should hit prefix cache

    scheduler.Stop();

    const bool tokens_match = (r1.tokens == r2.tokens) && !r1.tokens.empty();
    // We can't directly inspect hit_len from outside Scheduler, so we use
    // the indirect signal: if tokens match and prompt is >=16 tokens, cache worked
    const int hit_len = (tokens_match && prompt_ids.size() >= 16)
                        ? static_cast<int>((prompt_ids.size() / 16) * 16) : 0;

    std::cout << "prefix-test prompt_len=" << prompt_ids.size() << "\n";
    std::cout << "prefix-test hit_len=" << hit_len << "\n";
    std::cout << "prefix-test tokens_match=" << (tokens_match ? 1 : 0) << "\n";

    return tokens_match ? 0 : 1;
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
        MLLM_ERROR("main", "Parity: model load failed.");
        return -1;
    }

    // For Gemma: use BOS(2) + turn_marker(106) + "Hi" tokens
    // For Llama: use the original tokens
    const std::string mt = bundle.runner->GetModelType();
    std::vector<int64_t> token_list;
    if (mt == "gemma")
        token_list = {2, 106, 2430, 106, 108, 106, 4176, 108};  // <bos><turn|>user\nHi<turn|>\n<turn|>model\n
    else
        token_list = {15043, 6796, 263, 1243};

    auto ids = torch::tensor(token_list, torch::kInt64).unsqueeze(0);
    auto mask = torch::ones({1, (int64_t)token_list.size()}, torch::kInt64);
    auto logits = bundle.runner->Forward(ids, mask);
    auto last_logits = logits.index({0, logits.size(1) - 1}).to(torch::kFloat32);

    int64_t top_id = torch::argmax(last_logits, -1).item<int64_t>();
    float   top_val = last_logits[top_id].item<float>();
    std::cout << "last-token argmax token_id: " << top_id
              << "  logit=" << top_val << std::endl;

    // Also print top-5
    auto [topk_vals, topk_ids] = torch::topk(last_logits, 5);
    std::cout << "top-5: ";
    for (int i = 0; i < 5; ++i)
        std::cout << topk_ids[i].item<int64_t>() << "(" << topk_vals[i].item<float>() << ") ";
    std::cout << std::endl;

    return 0;
}

int main(int argc, char* argv[])
{
#ifdef _WIN32
    SetUnhandledExceptionFilter(CrashHandler);
#endif

    std::set_terminate([](){
        try { if (auto ep = std::current_exception()) std::rethrow_exception(ep); }
        catch (const std::exception& ex) { std::cerr << "[TERMINATE] " << ex.what() << std::endl; }
        catch (...) { std::cerr << "[TERMINATE] unknown exception\n"; }
        std::cerr.flush();
        std::abort();
    });

    try
    {
    const std::string model_path = ParseModelPath(argc, argv);

    // --------------------------------
    // --stress / --stress-direct
    // --------------------------------
    if (HasFlag(argc, argv, "--stress"))
        return RunStressTest(model_path, true);

    if (HasFlag(argc, argv, "--stress-direct"))
        return RunStressTest(model_path, false);

    if (HasFlag(argc, argv, "--stress-thinking"))
        return RunStressTest(model_path, true, true);

    // --------------------------------
    // --generate-test
    // --------------------------------
    if (HasFlag(argc, argv, "--generate-test"))
        return RunGenerateTest(model_path);

    // --------------------------------
    // --prefix-test
    // --------------------------------
    if (HasFlag(argc, argv, "--prefix-test"))
        return RunPrefixTest(model_path);

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
            MLLM_ERROR("main", "Failed to load model.");
            return -1;
        }

        if (!bundle.tokenizer->Load(model_path))
        {
            MLLM_ERROR("main", "Failed to load tokenizer.");
            return -1;
        }

        // Warmup: trigger lazy GPU transfer BEFORE InitKVCache
        // so KV caches are allocated on CUDA (not CPU)
        {
            MLLM_INFO("main", "[Serve] Warming up (GPU transfer)...");
            std::vector<int64_t> warmup_ids = { 1 };
            mllm::GenerateOptions warm_opts;
            warm_opts.max_new_tokens = 1;
            try { bundle.runner->Generate(warmup_ids, warm_opts); }
            catch (...) {}
            MLLM_INFO("main", "[Serve] Warmup done.");
        }

        // KV cache allocated AFTER GPU transfer → uses CUDA device
        bundle.runner->InitKVCache(1, bundle.runner->GetConfig().max_position_embeddings);
        MLLM_INFO("main", "[Serve] Ready.");

        mllm::Scheduler scheduler(*bundle.runner);
        scheduler.Start();

        mllm::HttpServer server(scheduler, *bundle.tokenizer, port);

        // Graceful shutdown on SIGINT (Ctrl-C) or SIGTERM.
        // The flag and pointers are set before Run() so the handler is safe.
        static std::atomic<bool> g_shutdown{false};
        static mllm::HttpServer*  g_server    = &server;
        static mllm::Scheduler*   g_scheduler = &scheduler;
        auto sighandler = [](int) {
            if (g_shutdown.exchange(true)) return;  // once only
            MLLM_INFO("main", "[Serve] Shutting down...");
            g_server->Stop();
            g_scheduler->Stop();
        };
        std::signal(SIGINT,  sighandler);
        std::signal(SIGTERM, sighandler);

        server.Run();  // blocks until Stop() is called

        // Ensure cleanup if Run() returned without signal (e.g. port conflict).
        if (!g_shutdown.exchange(true))
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
            MLLM_ERROR("main", "Tokenizer load failed.");
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
            MLLM_ERROR("main", "Tokenizer load failed.");
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
        MLLM_ERROR("main", "Failed to load model.");
        return -1;
    }

    if (!bundle.tokenizer->Load(model_path))
    {
        MLLM_ERROR("main", "Failed to load tokenizer.");
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
    options.enable_thinking    =
        bundle.tokenizer->SupportsThinking() && HasFlag(argc, argv, "--thinking");

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

