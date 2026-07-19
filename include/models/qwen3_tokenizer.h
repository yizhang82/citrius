#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace citrius::models {

class Qwen3Tokenizer {
  public:
    explicit Qwen3Tokenizer(const std::filesystem::path& model_directory);

    std::vector<std::int64_t> encode(std::string_view text) const;
    std::string decode(const std::vector<std::int64_t>& token_ids,
                       bool skip_special_tokens = false) const;
    std::string token_display_name(std::int64_t token_id) const;
    std::vector<std::int64_t> encode_chat_prompt(std::string_view user_prompt) const;

    static constexpr std::int64_t end_of_text_token_id = 151643;
    static constexpr std::int64_t im_start_token_id = 151644;
    static constexpr std::int64_t im_end_token_id = 151645;

  private:
    struct PairHash {
        std::size_t operator()(const std::pair<std::string, std::string>& pair) const noexcept;
    };

    std::vector<std::int64_t> encode_piece(std::string_view piece) const;

    std::unordered_map<std::string, std::int64_t> vocabulary_;
    std::vector<std::string> tokens_;
    std::unordered_map<std::pair<std::string, std::string>, std::size_t, PairHash> merge_ranks_;
    std::unordered_map<std::string, std::int64_t> special_tokens_;
    std::unordered_map<std::int64_t, std::string> special_token_text_;
    std::vector<std::string> byte_to_symbol_;
    std::unordered_map<std::string, unsigned char> symbol_to_byte_;
};

} // namespace citrius::models
