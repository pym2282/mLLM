// src/core/Logger.h
//
// Minimal structured logger.  Usage:
//   MLLM_INFO("Component", "message")
//   MLLM_WARN("Component", "message")
//   MLLM_ERROR("Component", "message")
//   MLLM_DEBUG("Component", "message")   // only when level <= Debug
//
// Level is set via MLLM_LOG_LEVEL env var at startup:
//   debug | info (default) | warn | error | none
//
// All output goes to stderr.

#pragma once

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>

namespace mllm
{
    enum class LogLevel : int { Debug = 0, Info = 1, Warn = 2, Error = 3, None = 4 };

    namespace detail
    {
        inline LogLevel& CurrentLevel()
        {
            static LogLevel level = []() -> LogLevel {
                const char* env = std::getenv("MLLM_LOG_LEVEL");
                if (!env) return LogLevel::Info;
                if (std::strcmp(env, "debug") == 0) return LogLevel::Debug;
                if (std::strcmp(env, "warn")  == 0) return LogLevel::Warn;
                if (std::strcmp(env, "error") == 0) return LogLevel::Error;
                if (std::strcmp(env, "none")  == 0) return LogLevel::None;
                return LogLevel::Info;
            }();
            return level;
        }

        inline std::mutex& LogMutex()
        {
            static std::mutex mu;
            return mu;
        }

        inline const char* LevelTag(LogLevel l)
        {
            switch (l) {
                case LogLevel::Debug: return "DEBUG";
                case LogLevel::Info:  return "INFO ";
                case LogLevel::Warn:  return "WARN ";
                case LogLevel::Error: return "ERROR";
                default:              return "?    ";
            }
        }

        inline void Log(LogLevel level, const char* component, const std::string& msg)
        {
            if (level < CurrentLevel()) return;
            std::lock_guard<std::mutex> lk(LogMutex());
            std::cerr << "[" << LevelTag(level) << "][" << component << "] " << msg << "\n";
        }
    }

    inline void SetLogLevel(LogLevel l) { detail::CurrentLevel() = l; }
}

#define MLLM_DEBUG(comp, msg) ::mllm::detail::Log(::mllm::LogLevel::Debug, comp, msg)
#define MLLM_INFO(comp, msg)  ::mllm::detail::Log(::mllm::LogLevel::Info,  comp, msg)
#define MLLM_WARN(comp, msg)  ::mllm::detail::Log(::mllm::LogLevel::Warn,  comp, msg)
#define MLLM_ERROR(comp, msg) ::mllm::detail::Log(::mllm::LogLevel::Error, comp, msg)
