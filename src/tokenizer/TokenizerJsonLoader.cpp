#include "tokenizer/TokenizerJsonLoader.h"

#include <fstream>
#include <stdexcept>

namespace mllm
{
    nlohmann::json TokenizerJsonLoader::Load(
        const std::string& model_path
    )
    {
        const std::string file_path =
            model_path + "/tokenizer.json";

        std::ifstream file(file_path);

        if (!file.is_open())
        {
            throw std::runtime_error(
                "Failed to open tokenizer.json: " + file_path
            );
        }

        nlohmann::json j;
        file >> j;

        return j;
    }
}