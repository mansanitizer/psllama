// Speech for the PS5 title: speech to text with whisper.cpp and text to speech
// with llama.cpp's mtmd audio generation (e.g. Qwen3-TTS), both on the same
// ggml/Vulkan device as the LLM. One model of each kind at a time; the caller
// unloads other engines first when GPU memory is needed.
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// Decodes a RIFF/WAVE file (PCM 8/16/24/32-bit or float 32, any channel
// count and sample rate) to mono float samples at `rate` Hz.
bool decode_wav(const std::string &bytes, int rate, std::vector<float> &samples, std::string &error);

struct SttSegment {
    double start_s = 0.0, end_s = 0.0;
    std::string text;
};

struct SttResult {
    std::string text, language;
    std::vector<SttSegment> segments;
    double audio_s = 0.0, load_s = 0.0, transcribe_s = 0.0;
};

// model: a whisper.cpp model file under /app0 (e.g. "whisper/ggml-large-v3-turbo-q5_0.bin").
bool stt_load(const std::string &model, std::string &error, double *load_s);
void stt_unload();
const std::string &stt_loaded_model();
// language: "" or "auto" to detect, else a code such as "en".
bool stt_transcribe(const std::vector<float> &samples16k, const std::string &language, bool translate,
                    SttResult &result, std::string &error);

struct TtsRequest {
    std::string text;
    std::string language;           // optional, model-specific (e.g. "en")
    std::string speaker_wav;        // optional reference voice, WAV bytes
    int max_frames = 1024;
    int top_k = 50;
    float top_p = 0.9f;
    float temperature = 0.9f;
    int64_t seed = -1;              // -1: random
};

struct TtsResult {
    std::string wav;                // WAV, PCM 16-bit mono
    int sample_rate = 0;
    double audio_s = 0.0, load_s = 0.0, generate_s = 0.0;
    int frames = 0;
};

// model: a folder under /app0 holding the backbone GGUF and its mmproj-*.gguf.
bool tts_load(const std::string &model, std::string &error, double *load_s);
void tts_unload();
const std::string &tts_loaded_model();
// progress(frames generated so far, max frames)
bool tts_generate(const TtsRequest &request, TtsResult &result, std::string &error,
                  const std::function<void(int, int)> &progress);
