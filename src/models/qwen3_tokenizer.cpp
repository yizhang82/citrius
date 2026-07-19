#include "models/qwen3_tokenizer.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace citrius::models {
namespace {

void append_utf8(std::string& output, std::uint32_t codepoint) {
    if (codepoint <= 0x7f)
        output.push_back(static_cast<char>(codepoint));
    else if (codepoint <= 0x7ff) {
        output.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else if (codepoint <= 0xffff) {
        output.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else {
        output.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
}

struct Rune {
    std::uint32_t value;
    std::size_t begin;
    std::size_t end;
};

std::vector<Rune> utf8_runes(std::string_view text) {
    std::vector<Rune> runes;
    for (std::size_t index = 0; index < text.size();) {
        const std::size_t begin = index;
        const auto first = static_cast<unsigned char>(text[index++]);
        std::uint32_t value = first;
        int remaining = 0;
        if ((first & 0xe0) == 0xc0) {
            value = first & 0x1f;
            remaining = 1;
        } else if ((first & 0xf0) == 0xe0) {
            value = first & 0x0f;
            remaining = 2;
        } else if ((first & 0xf8) == 0xf0) {
            value = first & 0x07;
            remaining = 3;
        } else if (first >= 0x80)
            throw std::invalid_argument("invalid UTF-8 input");
        for (int count = 0; count < remaining; ++count) {
            if (index >= text.size())
                throw std::invalid_argument("truncated UTF-8 input");
            const auto next = static_cast<unsigned char>(text[index++]);
            if ((next & 0xc0) != 0x80)
                throw std::invalid_argument("invalid UTF-8 continuation byte");
            value = (value << 6) | (next & 0x3f);
        }
        runes.push_back({value, begin, index});
    }
    return runes;
}

bool is_space(std::uint32_t value) {
    if (value <= 0x7f)
        return std::isspace(static_cast<unsigned char>(value)) != 0;
#ifdef _WIN32
    WORD type = 0;
    if (value <= 0xffff) {
        const wchar_t character = static_cast<wchar_t>(value);
        GetStringTypeW(CT_CTYPE1, &character, 1, &type);
        return (type & C1_SPACE) != 0;
    }
#endif
    return value == 0x85 || value == 0xa0 || value == 0x1680 ||
           (value >= 0x2000 && value <= 0x200a) || value == 0x2028 || value == 0x2029 ||
           value == 0x202f || value == 0x205f || value == 0x3000;
}

bool is_letter(std::uint32_t value) {
    if (value <= 0x7f)
        return std::isalpha(static_cast<unsigned char>(value)) != 0;
#ifdef _WIN32
    WORD type = 0;
    if (value <= 0xffff) {
        const wchar_t character = static_cast<wchar_t>(value);
        GetStringTypeW(CT_CTYPE1, &character, 1, &type);
        return (type & C1_ALPHA) != 0;
    }
#endif
    // All common non-ASCII scripts are represented as letters here. Emoji and
    // symbol blocks are excluded so byte-BPE punctuation grouping remains stable.
    return value < 0x2000 || (value >= 0x2e80 && value < 0xd800);
}

bool is_number(std::uint32_t value) {
    if (value <= 0x7f)
        return std::isdigit(static_cast<unsigned char>(value)) != 0;
#ifdef _WIN32
    WORD type = 0;
    if (value <= 0xffff) {
        const wchar_t character = static_cast<wchar_t>(value);
        GetStringTypeW(CT_CTYPE1, &character, 1, &type);
        return (type & C1_DIGIT) != 0;
    }
#endif
    return false;
}

bool is_newline(std::uint32_t value) {
    return value == '\r' || value == '\n';
}
bool is_punctuation(std::uint32_t value) {
    return !is_space(value) && !is_letter(value) && !is_number(value);
}

std::string normalize_nfc(std::string_view text) {
#ifdef _WIN32
    if (text.empty())
        return {};
    const int wide_size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(),
                                              static_cast<int>(text.size()), nullptr, 0);
    if (wide_size <= 0)
        throw std::invalid_argument("invalid UTF-8 input");
    std::wstring wide(static_cast<std::size_t>(wide_size), L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()),
                        wide.data(), wide_size);
    const int normalized_size = NormalizeString(NormalizationC, wide.data(), wide_size, nullptr, 0);
    if (normalized_size <= 0)
        throw std::runtime_error("failed to normalize tokenizer input");
    std::wstring normalized(static_cast<std::size_t>(normalized_size), L'\0');
    const int normalized_written =
        NormalizeString(NormalizationC, wide.data(), wide_size, normalized.data(), normalized_size);
    if (normalized_written <= 0)
        throw std::runtime_error("failed to normalize tokenizer input");
    const int utf8_size = WideCharToMultiByte(CP_UTF8, 0, normalized.data(), normalized_written,
                                              nullptr, 0, nullptr, nullptr);
    std::string output(static_cast<std::size_t>(utf8_size), '\0');
    WideCharToMultiByte(CP_UTF8, 0, normalized.data(), normalized_written, output.data(), utf8_size,
                        nullptr, nullptr);
    return output;
#else
    return std::string(text);
#endif
}

std::vector<std::string_view> pretokenize(std::string_view text) {
    const auto runes = utf8_runes(text);
    std::vector<std::string_view> pieces;
    std::size_t index = 0;
    const auto emit = [&](std::size_t first, std::size_t last) {
        pieces.emplace_back(text.data() + runes[first].begin,
                            runes[last - 1].end - runes[first].begin);
    };
    while (index < runes.size()) {
        const std::size_t start = index;
        if (runes[index].value == '\'' && index + 1 < runes.size()) {
            static constexpr const char* suffixes[] = {"s", "t", "re", "ve", "m", "ll", "d"};
            bool matched = false;
            for (const char* suffix : suffixes) {
                std::size_t cursor = index + 1;
                const char* character = suffix;
                while (*character && cursor < runes.size() && runes[cursor].value <= 0x7f &&
                       std::tolower(static_cast<unsigned char>(runes[cursor].value)) ==
                           *character) {
                    ++cursor;
                    ++character;
                }
                if (!*character) {
                    emit(start, cursor);
                    index = cursor;
                    matched = true;
                    break;
                }
            }
            if (matched)
                continue;
        }
        if (is_letter(runes[index].value)) {
            while (index < runes.size() && is_letter(runes[index].value))
                ++index;
            emit(start, index);
        } else if (!is_newline(runes[index].value) && !is_letter(runes[index].value) &&
                   !is_number(runes[index].value) && index + 1 < runes.size() &&
                   is_letter(runes[index + 1].value)) {
            index += 2;
            while (index < runes.size() && is_letter(runes[index].value))
                ++index;
            emit(start, index);
        } else if (is_number(runes[index].value)) {
            emit(index, index + 1);
            ++index;
        } else if ((runes[index].value == ' ' && index + 1 < runes.size() &&
                    is_punctuation(runes[index + 1].value)) ||
                   is_punctuation(runes[index].value)) {
            if (runes[index].value == ' ')
                ++index;
            while (index < runes.size() && is_punctuation(runes[index].value))
                ++index;
            while (index < runes.size() && is_newline(runes[index].value))
                ++index;
            emit(start, index);
        } else {
            while (index < runes.size() && is_space(runes[index].value))
                ++index;
            emit(start, index);
        }
    }
    return pieces;
}

std::string parse_json_string(std::string_view json, std::size_t& index) {
    if (index >= json.size() || json[index++] != '"')
        throw std::runtime_error("invalid tokenizer vocabulary JSON");
    std::string output;
    while (index < json.size()) {
        const char character = json[index++];
        if (character == '"')
            return output;
        if (character != '\\') {
            output.push_back(character);
            continue;
        }
        if (index >= json.size())
            break;
        const char escaped = json[index++];
        if (escaped == '"' || escaped == '\\' || escaped == '/')
            output.push_back(escaped);
        else if (escaped == 'b')
            output.push_back('\b');
        else if (escaped == 'f')
            output.push_back('\f');
        else if (escaped == 'n')
            output.push_back('\n');
        else if (escaped == 'r')
            output.push_back('\r');
        else if (escaped == 't')
            output.push_back('\t');
        else if (escaped == 'u') {
            if (index + 4 > json.size())
                throw std::runtime_error("invalid JSON Unicode escape");
            std::uint32_t value = 0;
            for (int digit = 0; digit < 4; ++digit) {
                const char hex = json[index++];
                value = value * 16 + (hex >= '0' && hex <= '9'   ? hex - '0'
                                      : hex >= 'a' && hex <= 'f' ? hex - 'a' + 10
                                                                 : hex - 'A' + 10);
            }
            append_utf8(output, value);
        } else
            throw std::runtime_error("unsupported JSON escape in tokenizer vocabulary");
    }
    throw std::runtime_error("unterminated tokenizer vocabulary string");
}

std::unordered_map<std::string, std::int64_t> load_vocabulary(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("cannot open tokenizer vocabulary: " + path.string());
    const std::string json((std::istreambuf_iterator<char>(stream)),
                           std::istreambuf_iterator<char>());
    std::unordered_map<std::string, std::int64_t> vocabulary;
    std::size_t index = 0;
    while (index < json.size() && json[index] != '{')
        ++index;
    if (index == json.size())
        throw std::runtime_error("invalid tokenizer vocabulary JSON");
    ++index;
    while (index < json.size()) {
        while (index < json.size() && std::isspace(static_cast<unsigned char>(json[index])))
            ++index;
        if (index < json.size() && json[index] == '}')
            break;
        std::string token = parse_json_string(json, index);
        while (index < json.size() &&
               (std::isspace(static_cast<unsigned char>(json[index])) || json[index] == ':'))
            ++index;
        std::int64_t id = 0;
        if (index >= json.size() || !std::isdigit(static_cast<unsigned char>(json[index])))
            throw std::runtime_error("invalid tokenizer vocabulary ID");
        while (index < json.size() && std::isdigit(static_cast<unsigned char>(json[index])))
            id = id * 10 + json[index++] - '0';
        vocabulary.emplace(std::move(token), id);
        while (index < json.size() && std::isspace(static_cast<unsigned char>(json[index])))
            ++index;
        if (index < json.size() && json[index] == ',')
            ++index;
    }
    return vocabulary;
}

} // namespace

std::size_t Qwen3Tokenizer::PairHash::operator()(
    const std::pair<std::string, std::string>& pair) const noexcept {
    return std::hash<std::string>{}(pair.first) ^ (std::hash<std::string>{}(pair.second) << 1);
}

Qwen3Tokenizer::Qwen3Tokenizer(const std::filesystem::path& model_directory) {
    vocabulary_ = load_vocabulary(model_directory / "vocab.json");
    tokens_.resize(vocabulary_.size());
    for (const auto& [token, id] : vocabulary_) {
        if (id < 0 || static_cast<std::size_t>(id) >= tokens_.size())
            throw std::runtime_error("non-contiguous tokenizer vocabulary IDs");
        tokens_[static_cast<std::size_t>(id)] = token;
    }
    std::ifstream merges(model_directory / "merges.txt");
    if (!merges)
        throw std::runtime_error("cannot open tokenizer merges: " +
                                 (model_directory / "merges.txt").string());
    std::string line;
    std::size_t rank = 0;
    while (std::getline(merges, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line.empty() || line[0] == '#')
            continue;
        const auto separator = line.find(' ');
        if (separator == std::string::npos)
            throw std::runtime_error("invalid tokenizer merge rule");
        merge_ranks_.emplace(std::make_pair(line.substr(0, separator), line.substr(separator + 1)),
                             rank++);
    }

    byte_to_symbol_.resize(256);
    std::vector<int> bytes;
    for (int value = 33; value <= 126; ++value)
        bytes.push_back(value);
    for (int value = 161; value <= 172; ++value)
        bytes.push_back(value);
    for (int value = 174; value <= 255; ++value)
        bytes.push_back(value);
    std::vector<bool> present(256, false);
    for (const int value : bytes)
        present[static_cast<std::size_t>(value)] = true;
    int extra = 0;
    for (int value = 0; value < 256; ++value) {
        const std::uint32_t symbol =
            present[static_cast<std::size_t>(value)] ? value : 256 + extra++;
        append_utf8(byte_to_symbol_[static_cast<std::size_t>(value)], symbol);
        symbol_to_byte_.emplace(byte_to_symbol_[static_cast<std::size_t>(value)],
                                static_cast<unsigned char>(value));
    }

    special_tokens_ = {
        {"<|endoftext|>", 151643},
        {"<|im_start|>", 151644},
        {"<|im_end|>", 151645},
        {"<|object_ref_start|>", 151646},
        {"<|object_ref_end|>", 151647},
        {"<|box_start|>", 151648},
        {"<|box_end|>", 151649},
        {"<|quad_start|>", 151650},
        {"<|quad_end|>", 151651},
        {"<|vision_start|>", 151652},
        {"<|vision_end|>", 151653},
        {"<|vision_pad|>", 151654},
        {"<|image_pad|>", 151655},
        {"<|video_pad|>", 151656},
        {"<tool_call>", 151657},
        {"</tool_call>", 151658},
        {"<|fim_prefix|>", 151659},
        {"<|fim_middle|>", 151660},
        {"<|fim_suffix|>", 151661},
        {"<|fim_pad|>", 151662},
        {"<|repo_name|>", 151663},
        {"<|file_sep|>", 151664},
        {"<tool_response>", 151665},
        {"</tool_response>", 151666},
        {"<think>", 151667},
        {"</think>", 151668},
    };
    for (const auto& [text, id] : special_tokens_)
        special_token_text_.emplace(id, text);
}

std::vector<std::int64_t> Qwen3Tokenizer::encode_piece(std::string_view piece) const {
    std::vector<std::string> symbols;
    symbols.reserve(piece.size());
    for (const unsigned char byte : piece)
        symbols.push_back(byte_to_symbol_[byte]);
    while (symbols.size() > 1) {
        std::size_t best_rank = std::numeric_limits<std::size_t>::max();
        std::pair<std::string, std::string> best;
        for (std::size_t index = 0; index + 1 < symbols.size(); ++index) {
            const auto found = merge_ranks_.find({symbols[index], symbols[index + 1]});
            if (found != merge_ranks_.end() && found->second < best_rank) {
                best_rank = found->second;
                best = found->first;
            }
        }
        if (best_rank == std::numeric_limits<std::size_t>::max())
            break;
        std::vector<std::string> merged;
        for (std::size_t index = 0; index < symbols.size();) {
            if (index + 1 < symbols.size() && symbols[index] == best.first &&
                symbols[index + 1] == best.second) {
                merged.push_back(symbols[index] + symbols[index + 1]);
                index += 2;
            } else
                merged.push_back(symbols[index++]);
        }
        symbols = std::move(merged);
    }
    std::vector<std::int64_t> ids;
    ids.reserve(symbols.size());
    for (const auto& symbol : symbols) {
        const auto found = vocabulary_.find(symbol);
        if (found == vocabulary_.end())
            throw std::runtime_error("tokenizer produced a symbol absent from the vocabulary");
        ids.push_back(found->second);
    }
    return ids;
}

std::vector<std::int64_t> Qwen3Tokenizer::encode(std::string_view input) const {
    const std::string text = normalize_nfc(input);
    std::vector<std::int64_t> ids;
    for (std::size_t offset = 0; offset < text.size();) {
        std::size_t special_offset = std::string::npos;
        const std::pair<const std::string, std::int64_t>* special = nullptr;
        for (const auto& candidate : special_tokens_) {
            const auto found = text.find(candidate.first, offset);
            if (found < special_offset) {
                special_offset = found;
                special = &candidate;
            }
        }
        const std::size_t plain_end = special ? special_offset : text.size();
        if (plain_end > offset) {
            const std::string_view plain(text.data() + offset, plain_end - offset);
            for (const auto piece : pretokenize(plain)) {
                auto piece_ids = encode_piece(piece);
                ids.insert(ids.end(), piece_ids.begin(), piece_ids.end());
            }
        }
        if (!special)
            break;
        ids.push_back(special->second);
        offset = special_offset + special->first.size();
    }
    return ids;
}

std::string Qwen3Tokenizer::decode(const std::vector<std::int64_t>& token_ids,
                                   bool skip_special_tokens) const {
    std::string output;
    std::string encoded_bytes;
    const auto flush = [&]() {
        const auto runes = utf8_runes(encoded_bytes);
        for (const auto& rune : runes) {
            const std::string symbol(encoded_bytes.data() + rune.begin, rune.end - rune.begin);
            const auto found = symbol_to_byte_.find(symbol);
            if (found == symbol_to_byte_.end())
                throw std::runtime_error("invalid byte-level tokenizer symbol");
            output.push_back(static_cast<char>(found->second));
        }
        encoded_bytes.clear();
    };
    for (const auto id : token_ids) {
        const auto special = special_token_text_.find(id);
        if (special != special_token_text_.end()) {
            flush();
            if (!skip_special_tokens)
                output += special->second;
        } else {
            if (id < 0 || static_cast<std::size_t>(id) >= tokens_.size())
                throw std::out_of_range("token ID is outside the Qwen3 vocabulary");
            encoded_bytes += tokens_[static_cast<std::size_t>(id)];
        }
    }
    flush();
    return output;
}

std::string Qwen3Tokenizer::token_display_name(std::int64_t token_id) const {
    const auto special = special_token_text_.find(token_id);
    if (special != special_token_text_.end())
        return special->second;
    if (token_id < 0 || static_cast<std::size_t>(token_id) >= tokens_.size())
        throw std::out_of_range("token ID is outside the Qwen3 vocabulary");

    const std::string decoded = decode({token_id});
    std::string display = "\"";
    constexpr char hex[] = "0123456789ABCDEF";
    for (const unsigned char byte : decoded) {
        if (byte == '\\')
            display += "\\\\";
        else if (byte == '"')
            display += "\\\"";
        else if (byte == '\n')
            display += "\\n";
        else if (byte == '\r')
            display += "\\r";
        else if (byte == '\t')
            display += "\\t";
        else if (byte >= 0x20 && byte <= 0x7e)
            display.push_back(static_cast<char>(byte));
        else {
            display += "\\x";
            display.push_back(hex[byte >> 4]);
            display.push_back(hex[byte & 0x0f]);
        }
    }
    display += '"';
    return display;
}

std::vector<std::int64_t> Qwen3Tokenizer::encode_chat_prompt(std::string_view user_prompt) const {
    return encode("<|im_start|>user\n" + std::string(user_prompt) +
                  "<|im_end|>\n<|im_start|>assistant\n");
}

} // namespace citrius::models
