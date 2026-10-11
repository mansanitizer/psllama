// OpenVLA control step (see vla_policy.h), following the reference
// implementation's predict_action: prompt "In: What action should the robot
// take to {instruction}?\nOut:", image patches right after BOS, greedy
// decoding of seven tokens from the last 256 ids of the Llama-2 vocabulary.
// SPDX-License-Identifier: GPL-3.0-or-later
#include "vla_policy.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <vector>

namespace {

constexpr llama_token kEmptyToken = 29871;   // "▁": the reference appends it after "Out:"
constexpr int kBaseVocab = 32000;            // Llama-2 vocabulary without OpenVLA's pad token
constexpr int kBins = 256;

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Decodes tokens or embeddings at explicit positions; logits for the last row only.
bool decode(llama_context *context, const llama_token *tokens, const float *embd, int n_embd, int count,
            llama_pos first_pos, bool want_logits, int n_batch) {
    for (int at = 0; at < count; at += n_batch) {
        const int n = std::min(n_batch, count - at);
        llama_batch batch = llama_batch_init(n, embd ? n_embd : 0, 1);
        batch.n_tokens = n;
        for (int i = 0; i < n; ++i) {
            if (embd) std::copy(embd + static_cast<size_t>(at + i) * n_embd, embd + static_cast<size_t>(at + i + 1) * n_embd,
                                batch.embd + static_cast<size_t>(i) * n_embd);
            else batch.token[i] = tokens[at + i];
            batch.pos[i] = first_pos + at + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = want_logits && at + i == count - 1;
        }
        const int rc = llama_decode(context, batch);
        llama_batch_free(batch);
        if (rc != 0) return false;
    }
    return true;
}

}  // namespace

bool vla_step(VlaVision *vision, llama_context *context, const uint8_t *rgb, const std::string &instruction,
              int n_batch, VlaStep &step, std::string &error) {
    const llama_model *model = llama_get_model(context);
    const llama_vocab *vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::string task = instruction;
    std::transform(task.begin(), task.end(), task.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const std::string prompt = "In: What action should the robot take to " + task + "?\nOut:";
    std::vector<llama_token> tokens(prompt.size() + 8);
    const int n = llama_tokenize(vocab, prompt.data(), static_cast<int32_t>(prompt.size()), tokens.data(),
                                 static_cast<int32_t>(tokens.size()), true, false);
    if (n < 2) { error = "the instruction could not be tokenized"; return false; }
    tokens.resize(static_cast<size_t>(n));
    if (tokens.back() != kEmptyToken) tokens.push_back(kEmptyToken);

    double t = now_s();
    std::vector<float> embeddings;
    int n_embd = 0;
    if (!vla_vision_encode(vision, rgb, embeddings, &n_embd, error)) return false;
    if (n_embd != llama_model_n_embd(model)) { error = "the projector does not match the language model"; return false; }
    step.vision_s = now_s() - t;

    // BOS, then the 256 image patches, then the rest of the prompt.
    t = now_s();
    llama_memory_clear(llama_get_memory(context), true);
    const int n_text = static_cast<int>(tokens.size()) - 1;
    step.prompt_tokens = 1 + kVlaPatches + n_text;
    if (step.prompt_tokens + kVlaActionDim > static_cast<int>(llama_n_ctx(context))) { error = "the instruction is too long for the context"; return false; }
    if (!decode(context, tokens.data(), nullptr, 0, 1, 0, false, n_batch) ||
        !decode(context, nullptr, embeddings.data(), n_embd, kVlaPatches, 1, false, n_batch) ||
        !decode(context, tokens.data() + 1, nullptr, 0, n_text, 1 + kVlaPatches, true, n_batch)) {
        error = "prompt evaluation failed";
        return false;
    }
    step.prompt_s = now_s() - t;

    t = now_s();
    llama_pos pos = step.prompt_tokens;
    for (int i = 0; i < kVlaActionDim; ++i) {
        const float *logits = llama_get_logits_ith(context, -1);
        const llama_token token = static_cast<llama_token>(std::max_element(logits, logits + n_vocab) - logits);
        step.tokens[i] = token;
        // Token id -> bin (counted down from the end of the base vocabulary) -> bin centre.
        const int bin = std::clamp(kBaseVocab - token - 1, 0, kBins - 2);
        step.normalized[i] = -1.0f + (2.0f * bin + 1.0f) / (kBins - 1);
        if (i + 1 < kVlaActionDim && !decode(context, &token, nullptr, 0, 1, pos++, true, n_batch)) {
            error = "action decoding failed";
            return false;
        }
    }
    step.decode_s = now_s() - t;
    return true;
}
