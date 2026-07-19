#include "models/qwen3_tokenizer.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>

namespace {

std::filesystem::path tokenizer_directory() {
    const char* configured = std::getenv("CITRIUS_QWEN3_MODEL_DIR");
    return configured ? std::filesystem::path(configured) : std::filesystem::path();
}

TEST(Qwen3TokenizerTest, EncodesOfficialHelloChatPrompt) {
    const auto directory = tokenizer_directory();
    if (directory.empty())
        GTEST_SKIP() << "set CITRIUS_QWEN3_MODEL_DIR to run tokenizer integration tests";
    const citrius::models::Qwen3Tokenizer tokenizer(directory);
    EXPECT_EQ(tokenizer.encode_chat_prompt("Hello"),
              (std::vector<std::int64_t>{151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198}));
}

TEST(Qwen3TokenizerTest, RoundTripsUtf8Text) {
    const auto directory = tokenizer_directory();
    if (directory.empty())
        GTEST_SKIP() << "set CITRIUS_QWEN3_MODEL_DIR to run tokenizer integration tests";
    const citrius::models::Qwen3Tokenizer tokenizer(directory);
    const std::string text = "Hello, café — 中文 🚀\n";
    EXPECT_EQ(tokenizer.decode(tokenizer.encode(text)), text);
}

TEST(Qwen3TokenizerTest, DecodesQwen3ThinkingTokens) {
    const auto directory = tokenizer_directory();
    if (directory.empty())
        GTEST_SKIP() << "set CITRIUS_QWEN3_MODEL_DIR to run tokenizer integration tests";
    const citrius::models::Qwen3Tokenizer tokenizer(directory);
    EXPECT_EQ(tokenizer.decode({151667, 198, 151668}), "<think>\n</think>");
    EXPECT_EQ(tokenizer.decode({151667, 198, 151668}, true), "\n");
    EXPECT_EQ(tokenizer.token_display_name(151667), "<think>");
    EXPECT_EQ(tokenizer.token_display_name(198), "\"\\n\"");
    EXPECT_EQ(tokenizer.token_display_name(525), "\" are\"");
}

} // namespace
