// Speech to text (whisper.cpp) and text to speech (llama.cpp mtmd audio
// generation) for the PS5 title.
// SPDX-License-Identifier: GPL-3.0-or-later
#include "audio_service.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "whisper.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <random>

extern "C" int sceKernelDebugOutText(int, const char *);

namespace {
whisper_context *stt = nullptr;
std::string stt_name;
llama_model *tts_model = nullptr;
llama_context *tts_context = nullptr;
mtmd_context *tts_mtmd = nullptr;
std::string tts_name;

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void klog_line(const char *prefix, const char *text) {
    if (!text || !*text) return;
    char line[600];
    std::snprintf(line, sizeof(line), "%s%s%s", prefix, text, text[std::strlen(text) - 1] == '\n' ? "" : "\n");
    sceKernelDebugOutText(0, line);
}

void whisper_log(ggml_log_level level, const char *text, void *) {
    if (level >= GGML_LOG_LEVEL_INFO) klog_line("[whisper] ", text);
}

void mtmd_log(ggml_log_level level, const char *text, void *) {
    if (level >= GGML_LOG_LEVEL_INFO) klog_line("[mtmd] ", text);
}

uint32_t le32(const unsigned char *p) { return p[0] | p[1] << 8 | p[2] << 16 | static_cast<uint32_t>(p[3]) << 24; }
uint16_t le16(const unsigned char *p) { return static_cast<uint16_t>(p[0] | p[1] << 8); }
} // namespace

bool decode_wav(const std::string &bytes, int rate, std::vector<float> &samples, std::string &error) {
    const auto *data = reinterpret_cast<const unsigned char *>(bytes.data());
    const std::size_t size = bytes.size();
    if (size < 12 || std::memcmp(data, "RIFF", 4) != 0 || std::memcmp(data + 8, "WAVE", 4) != 0) {
        error = "audio must be a RIFF/WAVE file";
        return false;
    }
    int format = 0, channels = 0, bits = 0;
    uint32_t source_rate = 0;
    const unsigned char *pcm = nullptr;
    std::size_t pcm_size = 0;
    for (std::size_t at = 12; at + 8 <= size;) {
        const uint32_t chunk = le32(data + at + 4);
        const unsigned char *body = data + at + 8;
        const std::size_t available = std::min<std::size_t>(chunk, size - at - 8);
        if (std::memcmp(data + at, "fmt ", 4) == 0 && available >= 16) {
            format = le16(body);
            channels = le16(body + 2);
            source_rate = le32(body + 4);
            bits = le16(body + 14);
            if (format == 0xFFFE && available >= 26) format = le16(body + 24);  // WAVE_FORMAT_EXTENSIBLE
        } else if (std::memcmp(data + at, "data", 4) == 0) {
            pcm = body;
            pcm_size = available;
        }
        at += 8 + chunk + (chunk & 1);
    }
    const int bytes_per_sample = bits / 8;
    if (!pcm || channels < 1 || source_rate == 0 || bytes_per_sample < 1 ||
        !((format == 1 && bits >= 8 && bits <= 32) || (format == 3 && bits == 32))) {
        error = "unsupported WAV encoding (PCM 8/16/24/32-bit or float 32 expected)";
        return false;
    }
    const std::size_t frames = pcm_size / (static_cast<std::size_t>(bytes_per_sample) * channels);
    std::vector<float> mono(frames);
    for (std::size_t i = 0; i < frames; ++i) {
        float sum = 0.0f;
        for (int c = 0; c < channels; ++c) {
            const unsigned char *s = pcm + (i * channels + c) * bytes_per_sample;
            float v;
            if (format == 3) {
                std::memcpy(&v, s, 4);
            } else if (bits == 8) {
                v = (s[0] - 128) / 128.0f;
            } else if (bits == 16) {
                v = static_cast<int16_t>(le16(s)) / 32768.0f;
            } else if (bits == 24) {
                const int32_t x = static_cast<int32_t>(s[0] << 8 | s[1] << 16 | static_cast<uint32_t>(s[2]) << 24) >> 8;
                v = x / 8388608.0f;
            } else {
                v = static_cast<int32_t>(le32(s)) / 2147483648.0f;
            }
            sum += v;
        }
        mono[i] = sum / channels;
    }
    if (static_cast<int>(source_rate) == rate) {
        samples = std::move(mono);
        return true;
    }
    // Linear resampling is enough for speech recognition input.
    const double step = static_cast<double>(source_rate) / rate;
    const std::size_t out = static_cast<std::size_t>(frames / step);
    samples.resize(out);
    for (std::size_t i = 0; i < out; ++i) {
        const double position = i * step;
        const std::size_t j = static_cast<std::size_t>(position);
        const float frac = static_cast<float>(position - j);
        samples[i] = j + 1 < frames ? mono[j] * (1.0f - frac) + mono[j + 1] * frac : mono[frames - 1];
    }
    return true;
}

const std::string &stt_loaded_model() { return stt_name; }

void stt_unload() {
    if (stt) whisper_free(stt);
    stt = nullptr;
    stt_name.clear();
}

bool stt_load(const std::string &model, std::string &error, double *load_s) {
    if (stt && stt_name == model) return true;
    stt_unload();
    whisper_log_set(whisper_log, nullptr);
    whisper_context_params params = whisper_context_default_params();
    params.use_gpu = true;
    params.flash_attn = true;
    params.gpu_device = 0;
    const double started = now_s();
    stt = whisper_init_from_file_with_params(("/app0/" + model).c_str(), params);
    if (load_s) *load_s = now_s() - started;
    if (!stt) {
        error = "speech-to-text model failed to load (see klog)";
        return false;
    }
    stt_name = model;
    return true;
}

bool stt_transcribe(const std::vector<float> &samples16k, const std::string &language, bool translate,
                    SttResult &result, std::string &error) {
    if (!stt) {
        error = "no speech-to-text model is loaded";
        return false;
    }
    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.n_threads = 4;
    params.translate = translate;
    params.language = language.empty() ? "auto" : language.c_str();
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;
    params.print_special = false;
    const double started = now_s();
    if (whisper_full(stt, params, samples16k.data(), static_cast<int>(samples16k.size())) != 0) {
        error = "transcription failed (see klog)";
        return false;
    }
    result.transcribe_s = now_s() - started;
    result.audio_s = samples16k.size() / static_cast<double>(WHISPER_SAMPLE_RATE);
    result.language = whisper_lang_str(whisper_full_lang_id(stt));
    result.text.clear();
    result.segments.clear();
    for (int i = 0; i < whisper_full_n_segments(stt); ++i) {
        SttSegment segment;
        segment.start_s = whisper_full_get_segment_t0(stt, i) / 100.0;
        segment.end_s = whisper_full_get_segment_t1(stt, i) / 100.0;
        segment.text = whisper_full_get_segment_text(stt, i);
        result.text += segment.text;
        result.segments.push_back(std::move(segment));
    }
    return true;
}

const std::string &tts_loaded_model() { return tts_name; }

void tts_unload() {
    if (tts_mtmd) mtmd_free(tts_mtmd);
    if (tts_context) llama_free(tts_context);
    if (tts_model) llama_model_free(tts_model);
    tts_mtmd = nullptr;
    tts_context = nullptr;
    tts_model = nullptr;
    tts_name.clear();
}

bool tts_load(const std::string &model, std::string &error, double *load_s) {
    if (tts_model && tts_name == model) return true;
    tts_unload();
    const std::string folder = "/app0/" + model;
    std::string backbone, projector;
    if (DIR *dir = opendir(folder.c_str())) {
        while (dirent *entry = readdir(dir)) {
            const std::string name = entry->d_name;
            if (name.size() < 5 || name.compare(name.size() - 5, 5, ".gguf") != 0) continue;
            (name.rfind("mmproj", 0) == 0 ? projector : backbone) = folder + "/" + name;
        }
        closedir(dir);
    }
    if (backbone.empty() || projector.empty()) {
        error = "text-to-speech folder needs a model .gguf and an mmproj-*.gguf";
        return false;
    }
    mtmd_helper_log_set(mtmd_log, nullptr);
    const double started = now_s();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 999;
    tts_model = llama_model_load_from_file(backbone.c_str(), model_params);
    if (tts_model) {
        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = 4096;
        context_params.n_batch = 128;   // one submit stays well under the 10 s GPU watchdog
        context_params.n_ubatch = 128;
        context_params.n_threads = 4;
        context_params.n_threads_batch = 4;
        context_params.embeddings = true;  // the audio stages read the backbone's hidden states
        tts_context = llama_init_from_model(tts_model, context_params);
    }
    if (tts_context) {
        mtmd_context_params mtmd_params = mtmd_context_params_default();
        mtmd_params.use_gpu = true;
        mtmd_params.n_threads = 4;
        mtmd_params.print_timings = false;
        tts_mtmd = mtmd_init_from_file(projector.c_str(), tts_model, mtmd_params);
    }
    if (load_s) *load_s = now_s() - started;
    if (!tts_mtmd || mtmd_gen_audio_get_info(tts_mtmd).type == MTMD_GEN_AUDIO_TYPE_NONE) {
        tts_unload();
        error = "text-to-speech model failed to load or cannot generate audio (see klog)";
        return false;
    }
    tts_name = model;
    return true;
}

bool tts_generate(const TtsRequest &request, TtsResult &result, std::string &error,
                  const std::function<void(int, int)> &progress) {
    if (!tts_mtmd) {
        error = "no text-to-speech model is loaded";
        return false;
    }
    llama_memory_clear(llama_get_memory(tts_context), true);
    mtmd::bitmap_ptr speaker;
    if (!request.speaker_wav.empty()) {
        auto wrapper = mtmd_helper_bitmap_init_from_buf(
            tts_mtmd, reinterpret_cast<const unsigned char *>(request.speaker_wav.data()), request.speaker_wav.size(),
            false, mtmd_helper_init_opt_default());
        if (!wrapper.bitmap) {
            error = "speaker reference audio could not be decoded";
            return false;
        }
        speaker.reset(wrapper.bitmap);
    }
    const uint32_t seed = request.seed >= 0 ? static_cast<uint32_t>(request.seed) : std::random_device{}();
    mtmd_helper::gen_audio gen(tts_context, tts_mtmd);
    mtmd_helper_gen_audio_inp input{};
    input.seq_id = 0;
    input.prompt = request.text.c_str();
    input.prompt_len = request.text.size();
    input.speaker_ref = speaker.get();
    input.lang = request.language.empty() ? nullptr : request.language.c_str();
    input.top_k = request.top_k;
    input.top_p = request.top_p;
    input.seed = seed;
    input.out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;
    const double started = now_s();
    if (gen.set_input(&input) != 0) {
        error = "text-to-speech input rejected";
        return false;
    }
    for (;;) {
        const int32_t remaining = gen.step_prompt(128);
        if (remaining < 0) {
            error = "text-to-speech prompt processing failed";
            return false;
        }
        if (remaining == 0) break;
    }
    // Semantic codes are sampled from the backbone like the llama.cpp tts tool does.
    llama_sampler *sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(request.top_k));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(request.top_p, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(request.temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));
    llama_token sampled = llama_sampler_sample(sampler, tts_context, -1);
    const float *h_state = llama_get_embeddings_ith(tts_context, -1);
    bool stop = false;
    int frames = 0;
    while (!stop && frames < request.max_frames) {
        const float *h_next = nullptr;
        if (gen.step_gen(sampled, h_state, &h_next, &stop) != 0) {
            llama_sampler_free(sampler);
            error = "text-to-speech generation failed";
            return false;
        }
        if (!h_next) break;
        ++frames;
        h_state = h_next;
        sampled = llama_sampler_sample(sampler, tts_context, -1);
        if (progress && frames % 12 == 0) progress(frames, request.max_frames);
    }
    llama_sampler_free(sampler);
    int32_t rate = 0;
    const char *wav = nullptr;
    size_t wav_size = 0;
    int64_t samples = 0;
    if (gen.get_output(&rate, &wav, &wav_size, &samples) != 0 || !wav || wav_size == 0) {
        error = "text-to-speech produced no audio";
        return false;
    }
    result.wav.assign(wav, wav_size);
    result.sample_rate = rate;
    result.audio_s = rate > 0 ? samples / static_cast<double>(rate) : 0.0;
    result.generate_s = now_s() - started;
    result.frames = frames;
    return true;
}
