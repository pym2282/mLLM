// src/core/MllmException.h
//
// Exception hierarchy for mLLM runtime errors.
// Callers catch MllmException for any runtime error,
// or a subtype to handle specific failure categories.

#pragma once
#include <stdexcept>
#include <string>

namespace mllm
{
    struct MllmException    : std::runtime_error { using runtime_error::runtime_error; };
    struct ModelLoadError   : MllmException      { using MllmException::MllmException; };
    struct InferenceError   : MllmException      { using MllmException::MllmException; };
    struct TokenizerError   : MllmException      { using MllmException::MllmException; };
    struct QueueError       : MllmException      { using MllmException::MllmException; };
}
