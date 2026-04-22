#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>

/*
Mini LLM Tokenizer (C++)
------------------------
목표:
- 문자열 -> Token IDs (encode)
- Token IDs -> 문자열 (decode)
- 실제 LLM의 입력 흐름 이해

하드코딩 logits 실험은 제외하고,
바로 tokenizer 단계부터 시작합니다.
*/

class MiniTokenizer {
private:
    std::unordered_map<std::string, int> vocab = {
        {"1", 16},
        {"+", 488},
        {"=", 284},
        {"2", 17},
        {"hello", 1001},
        {"world", 1002}
    };

    std::unordered_map<int, std::string> reverse_vocab = {
        {16, "1"},
        {488, "+"},
        {284, "="},
        {17, "2"},
        {1001, "hello"},
        {1002, "world"}
    };

public:
    std::vector<int> encode(const std::vector<std::string>& words) {
        std::vector<int> result;

        for (const auto& word : words) {
            if (vocab.find(word) != vocab.end()) {
                result.push_back(vocab[word]);
            }
            else {
                result.push_back(-1); // unknown token
            }
        }

        return result;
    }

    std::string decode(const std::vector<int>& ids) {
        std::string result;

        for (int id : ids) {
            if (reverse_vocab.find(id) != reverse_vocab.end()) {
                result += reverse_vocab[id] + " ";
            }
            else {
                result += "[UNK] ";
            }
        }

        return result;
    }
};

int main() {
    MiniTokenizer tokenizer;

    std::vector<std::string> input = {"1", "+", "1", "="};

    std::cout << "Input Text:\n";
    for (const auto& word : input) {
        std::cout << word << " ";
    }

    std::cout << "\n\n=== Encode ===\n";
    std::vector<int> tokenIds = tokenizer.encode(input);

    for (int id : tokenIds) {
        std::cout << id << " ";
    }

    std::cout << "\n\n=== Decode ===\n";
    std::cout << tokenizer.decode(tokenIds) << "\n";

    std::cout << "\n핵심:\n";
    std::cout << "모델은 문자열이 아니라 Token IDs를 입력으로 받습니다.\n";

    return 0;
}
