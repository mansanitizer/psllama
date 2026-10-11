// OpenAI-compatible inference, model inventory, and dynamic loading endpoints.
// Deliberately single-threaded: one request at a time, no concurrent GPU work.
// SPDX-License-Identifier: GPL-3.0-or-later
#include "http_server.h"
#include "sd_service.h"
#include "audio_service.h"
#include "vla_policy.h"
#include "ggml-cpu.h"
#include "llama.h"
#include "chat.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "status_screen.h"
#include <arpa/inet.h>
#include <algorithm>
#include <cerrno>
#include <cstdarg>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <netinet/in.h>
#include <string>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#include <vector>
#include <xmmintrin.h>

extern "C" int sceKernelDebugOutText(int, const char *);

namespace {
constexpr std::size_t kMaxHeader = 8192;
constexpr std::size_t kMaxBody = 64u << 20;  // audio uploads arrive base64 in JSON
constexpr int kPrefillBatch = 128;
// An LLM up to this size shares the GPU heap with the speech models (voice
// agent: Whisper + a small LLM + TTS is ~6.5 GB); larger ones need it alone.
constexpr uint64_t kSmallLlmBytes = 5ull << 30;
constexpr int kMaxSafeBatch = 128;
// The API key is given at build time (PSLLAMA_API_KEY in the environment of
// build-gguf-vk-title.sh), so it never lives in the source tree.
#ifndef PSLLAMA_API_KEY
#error "build with -DPSLLAMA_API_KEY=\"<key>\" (see build-gguf-vk-title.sh)"
#endif
constexpr char kApiKey[] = PSLLAMA_API_KEY;

// Tokens between on-screen progress lines. Each line redraws, flushes the
// whole frame from the CPU cache and waits for vblank; PS5_PROGRESS_EVERY=0
// (vk-env.conf) turns them off.
int progress_every() {
    static const int every = [] {
        const char *value = std::getenv("PS5_PROGRESS_EVERY");
        return value ? std::max(0, std::atoi(value)) : 16;
    }();
    return every;
}

bool allowed_tailnet_peer(const struct sockaddr_in &peer) {
    const uint32_t address = ntohl(peer.sin_addr.s_addr);
    if ((address >> 24) == 127) return true; // in-process startup self-test
    // Tailscale IPv4 addresses come from CGNAT 100.64.0.0/10.
    return (address & 0xffc00000u) == 0x64400000u;
}

bool valid_api_key(const std::string &authorization) {
    const std::string expected = std::string("Bearer ") + kApiKey;
    unsigned diff = static_cast<unsigned>(authorization.size() ^ expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const unsigned char actual = i < authorization.size()
            ? static_cast<unsigned char>(authorization[i]) : 0;
        diff |= static_cast<unsigned>(actual ^ static_cast<unsigned char>(expected[i]));
    }
    return diff == 0;
}

void status(const char *message) {
    char line[512];
    std::snprintf(line, sizeof(line), "[llm-http] %s\n", message);
    sceKernelDebugOutText(0, line);
    status_screen::line(message);
}

void statusf(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void statusf(const char *fmt, ...) {
    char message[448];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(message, sizeof(message), fmt, args);
    va_end(args);
    status(message);
}

double clock_seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool send_all(int fd, const char *data, std::size_t size) {
    while (size) {
        const ssize_t sent = send(fd, data, size, 0);
        if (sent < 0 && errno == EINTR) continue;
        if (sent <= 0) return false;
        data += sent;
        size -= static_cast<std::size_t>(sent);
    }
    return true;
}

bool send_response(int fd, int code, const char *reason, const std::string &type,
                   const std::string &body) {
    char header[256];
    const int n = std::snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
        "Connection: close\r\nCache-Control: no-store\r\n\r\n",
        code, reason, type.c_str(), body.size());
    return n > 0 && send_all(fd, header, static_cast<std::size_t>(n)) &&
           send_all(fd, body.data(), body.size());
}

std::string json_escape(const std::string &in);

bool send_http_chunk(int fd, const std::string &body) {
    char length[32];
    const int n = std::snprintf(length, sizeof(length), "%lx\r\n",
                                static_cast<unsigned long>(body.size()));
    return n > 0 && send_all(fd, length, static_cast<std::size_t>(n)) &&
           send_all(fd, body.data(), body.size()) && send_all(fd, "\r\n", 2);
}

bool begin_sse(int fd) {
    static const char headers[] =
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\n"
        "Cache-Control: no-cache\r\nX-Accel-Buffering: no\r\n"
        "Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
    return send_all(fd, headers, sizeof(headers) - 1);
}

bool send_sse_data(int fd, const std::string &data) {
    return send_http_chunk(fd, "data: " + data + "\n\n");
}

bool end_sse(int fd) { return send_all(fd, "0\r\n\r\n", 5); }

std::size_t complete_utf8_prefix(const std::string &text) {
    std::size_t at = 0;
    while (at < text.size()) {
        const unsigned char lead = static_cast<unsigned char>(text[at]);
        std::size_t width = 1;
        if ((lead & 0xe0) == 0xc0) width = 2;
        else if ((lead & 0xf0) == 0xe0) width = 3;
        else if ((lead & 0xf8) == 0xf0) width = 4;
        if (at + width > text.size()) break;
        bool valid = true;
        for (std::size_t i = 1; i < width; ++i) {
            if ((static_cast<unsigned char>(text[at + i]) & 0xc0) != 0x80) {
                valid = false;
                break;
            }
        }
        at += valid ? width : 1;
    }
    return at;
}

std::string sse_chunk(const std::string &delta, const char *finish_reason) {
    std::string json = "{\"id\":\"chatcmpl-ps5\",\"object\":\"chat.completion.chunk\","
        "\"created\":0,\"model\":\"ps5-local\",\"choices\":[{\"index\":0,\"delta\":{";
    if (!delta.empty()) json += "\"content\":\"" + json_escape(delta) + "\"";
    if (finish_reason) {
        if (!delta.empty()) json += ",";
        json += "},\"finish_reason\":\"" + std::string(finish_reason) + "\"}]}";
    } else {
        json += "},\"finish_reason\":null}]}";
    }
    return json;
}

std::string json_escape(const std::string &in) {
    std::string out;
    out.reserve(in.size() + 16);
    for (unsigned char c : in) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (c < 0x20) out += ' ';
            else out += static_cast<char>(c);
        }
    }
    return out;
}

void append_utf8(std::string &out, unsigned cp) {
    if (cp <= 0x7f) out += static_cast<char>(cp);
    else if (cp <= 0x7ff) {
        out += static_cast<char>(0xc0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3f));
    } else if (cp <= 0xffff) {
        out += static_cast<char>(0xe0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3f));
        out += static_cast<char>(0x80 | (cp & 0x3f));
    } else {
        out += static_cast<char>(0xf0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3f));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3f));
        out += static_cast<char>(0x80 | (cp & 0x3f));
    }
}

bool parse_json_string(const std::string &s, std::size_t &at, std::string &out) {
    if (at >= s.size() || s[at++] != '"') return false;
    out.clear();
    while (at < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[at++]);
        if (c == '"') return true;
        if (c != '\\') {
            if (c < 0x20) return false;
            out += static_cast<char>(c);
            continue;
        }
        if (at == s.size()) return false;
        const char e = s[at++];
        switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
            if (at + 4 > s.size()) return false;
            unsigned cp = 0;
            for (int i = 0; i < 4; ++i) {
                const char h = s[at++];
                cp <<= 4;
                if (h >= '0' && h <= '9') cp |= unsigned(h - '0');
                else if (h >= 'a' && h <= 'f') cp |= unsigned(h - 'a' + 10);
                else if (h >= 'A' && h <= 'F') cp |= unsigned(h - 'A' + 10);
                else return false;
            }
            append_utf8(out, cp);
            break;
        }
        default: return false;
        }
    }
    return false;
}

std::size_t skip_ws(const std::string &s, std::size_t at) {
    while (at < s.size() && (s[at] == ' ' || s[at] == '\t' || s[at] == '\r' || s[at] == '\n')) ++at;
    return at;
}

bool find_property(const std::string &json, const std::string &name, std::size_t from,
                   std::size_t &value_at) {
    std::size_t at = from;
    while (at < json.size()) {
        at = json.find('"', at);
        if (at == std::string::npos) return false;
        std::string key;
        if (!parse_json_string(json, at, key)) return false;
        at = skip_ws(json, at);
        if (at >= json.size() || json[at++] != ':') continue;
        at = skip_ws(json, at);
        if (key == name) {
            value_at = at;
            return true;
        }
        // Skip the value so a matching string nested below another property is
        // not mistaken for a sibling property.
        if (at < json.size() && json[at] == '"') {
            std::string ignored;
            if (!parse_json_string(json, at, ignored)) return false;
        } else if (at < json.size() && (json[at] == '{' || json[at] == '[')) {
            const char open = json[at], close = open == '{' ? '}' : ']';
            int depth = 0;
            bool quoted = false, escaped = false;
            for (; at < json.size(); ++at) {
                const char c = json[at];
                if (quoted) {
                    if (escaped) escaped = false;
                    else if (c == '\\') escaped = true;
                    else if (c == '"') quoted = false;
                } else if (c == '"') quoted = true;
                else if (c == open) ++depth;
                else if (c == close && --depth == 0) { ++at; break; }
            }
        } else {
            while (at < json.size() && json[at] != ',' && json[at] != '}' && json[at] != ']') ++at;
        }
    }
    return false;
}

bool property_string(const std::string &json, const char *key, std::string &value,
                     std::size_t from = 0) {
    std::size_t at;
    return find_property(json, key, from, at) && parse_json_string(json, at, value);
}

bool property_int(const std::string &json, const char *key, int &value) {
    std::size_t at;
    if (!find_property(json, key, 0, at)) return false;
    char *end = nullptr;
    const long v = std::strtol(json.c_str() + at, &end, 10);
    if (end == json.c_str() + at || v < 0 || v > 1000000) return false;
    value = static_cast<int>(v);
    return true;
}

bool property_bool(const std::string &json, const char *key, bool &value) {
    std::size_t at;
    if (!find_property(json, key, 0, at)) return false;
    if (json.compare(at, 4, "true") == 0) { value = true; return true; }
    if (json.compare(at, 5, "false") == 0) { value = false; return true; }
    return false;
}

bool property_float(const std::string &json, const char *key, float &value) {
    std::size_t at;
    if (!find_property(json, key, 0, at)) return false;
    char *end = nullptr;
    const float v = std::strtof(json.c_str() + at, &end);
    if (end == json.c_str() + at || !std::isfinite(v)) return false;
    value = v;
    return true;
}

struct InferenceState {
    llama_model *model;
    llama_context *context;
    int context_tokens;
    int max_output_tokens;
    int threads;
    int threads_batch;
    int gpu_layers;
    int load_mode;
    int lazy_mode;
    bool use_extra_bufts;
    bool check_tensors;
    int batch_tokens;
    bool offload_kqv;
    bool op_offload;
    int flash_attention;
    ggml_type cache_type_k;
    ggml_type cache_type_v;
    float rope_freq_base;
    float rope_freq_scale;
    float default_temperature;
    int top_k = 0;
    float top_p = 1.0f;
    float min_p = 0.0f;
    int repeat_last_n = 64;
    float repeat_penalty = 1.0f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    uint32_t seed = LLAMA_DEFAULT_SEED;
    std::string model_file;
    bool loading = false;
};

// Load progress for a client that asked for it with "stream": true on
// /v1/models/load: SSE events while loading, then the usual JSON as the last
// event. A client that goes away stops the events, not the load.
// Image input for chat: the loaded LLM's vision projector, looked for next to
// the model as <model name without .gguf>.mmproj.gguf and loaded on the first
// request that carries an image. It is released whenever the LLM is.
mtmd_context *vision_context = nullptr;
const llama_model *vision_model = nullptr;

// OpenVLA's image encoder (two vision towers and a projector), loaded beside
// its Llama-2 language model on the first /v1/vla/act request.
VlaVision *vla_vision = nullptr;
std::string vla_vision_model;

void vla_release() {
    if (vla_vision) {
        vla_vision_free(vla_vision);
        status("VLA IMAGE ENCODER UNLOADED");
    }
    vla_vision = nullptr;
    vla_vision_model.clear();
}

void vision_release() {
    if (vision_context) {
        mtmd_free(vision_context);
        status("VISION PROJECTOR UNLOADED");
    }
    vision_context = nullptr;
    vision_model = nullptr;
}

mtmd_context *vision_for(const InferenceState &state, std::string &error) {
    if (vision_context && vision_model == state.model) return vision_context;
    vision_release();
    std::string stem = state.model_file;
    if (stem.size() > 5 && stem.compare(stem.size() - 5, 5, ".gguf") == 0) stem.resize(stem.size() - 5);
    const std::string path = "/app0/" + stem + ".mmproj.gguf";
    struct stat info{};
    if (stat(path.c_str(), &info) != 0) {
        error = "the loaded model cannot take images: no " + stem + ".mmproj.gguf in the app folder";
        return nullptr;
    }
    // A projector and its work buffers need 1.5 to 2.5 GB; the speech models
    // reload on their next use.
    stt_unload();
    tts_unload();
    statusf("LOADING VISION PROJECTOR %s", (stem + ".mmproj.gguf").c_str());
    mtmd_context_params params = mtmd_context_params_default();
    params.use_gpu = true;
    params.n_threads = 4;
    params.print_timings = false;
    // Dynamic-resolution projectors (Qwen-VL) otherwise size their buffers for
    // 1288x1288 input, which does not fit beside a 7B model. 1024 tokens is an
    // 896x896 image for Qwen-VL and what it needs for grounding.
    params.image_max_tokens = 1024;
    vision_context = mtmd_init_from_file(path.c_str(), state.model, params);
    if (!vision_context || !mtmd_support_vision(vision_context)) {
        vision_release();
        error = "the vision projector failed to load (see klog)";
        return nullptr;
    }
    vision_model = state.model;
    return vision_context;
}

struct LoadStream {
    int fd = -1;
    bool open = false;
    double started = 0.0;
    void event(const std::string &json) {
        if (open && !send_sse_data(fd, json)) open = false;
    }
    void phase(const char *name, const std::string &model) {
        char json[320];
        std::snprintf(json, sizeof(json), "{\"status\":\"%s\",\"model\":\"%s\",\"elapsed_s\":%.1f}",
                      name, json_escape(model).c_str(), clock_seconds() - started);
        event(json);
    }
};

struct ModelLoadProgress {
    int last_percent = -1;
    int last_streamed = -1;
    double started = 0.0;
    LoadStream *stream = nullptr;
    const char *phase = "loading";
    std::string model;
};

bool model_load_progress(float progress, void *user_data) {
    auto *state = static_cast<ModelLoadProgress *>(user_data);
    const int percent = static_cast<int>(progress * 100.0f);
    if (percent / 10 != state->last_percent / 10 || percent == 100) {
        state->last_percent = percent;
        char message[128];
        std::snprintf(message, sizeof(message), "LOADING MODEL %d%% T=%.0FS",
                      percent, clock_seconds() - state->started);
        status(message);
    }
    if (state->stream && percent != state->last_streamed) {
        state->last_streamed = percent;
        char json[320];
        std::snprintf(json, sizeof(json),
                      "{\"status\":\"%s\",\"model\":\"%s\",\"progress\":%d,\"elapsed_s\":%.1f}",
                      state->phase, json_escape(state->model).c_str(), percent,
                      clock_seconds() - state->stream->started);
        state->stream->event(json);
    }
    return true;
}

bool valid_model_filename(const std::string &name) {
    if (name.size() < 6 || name.size() >= 128 || name.substr(name.size() - 5) != ".gguf") return false;
    for (unsigned char c : name)
        if (!std::isalnum(c) && c != '-' && c != '_' && c != '.') return false;
    return name.find("..") == std::string::npos;
}

struct ModelOptions {
    int context_tokens;
    int max_output_tokens;
    int threads;
    int threads_batch;
    int gpu_layers;
    int load_mode;
    int lazy_mode;
    bool use_extra_bufts;
    bool check_tensors;
    int batch_tokens;
    bool offload_kqv;
    bool op_offload;
    int flash_attention;
    ggml_type cache_type_k;
    ggml_type cache_type_v;
    float rope_freq_base;
    float rope_freq_scale;
    float temperature;
    int top_k;
    float top_p;
    float min_p;
    int repeat_last_n;
    float repeat_penalty;
    float frequency_penalty;
    float presence_penalty;
    uint32_t seed;
};

ModelOptions options_from_state(const InferenceState &state) {
    return {state.context_tokens, state.max_output_tokens, state.threads,
        state.threads_batch, state.gpu_layers, state.load_mode, state.lazy_mode,
        state.use_extra_bufts, state.check_tensors, state.batch_tokens, state.offload_kqv,
        state.op_offload, state.flash_attention, state.cache_type_k, state.cache_type_v,
        state.rope_freq_base, state.rope_freq_scale, state.default_temperature,
        state.top_k, state.top_p, state.min_p, state.repeat_last_n, state.repeat_penalty,
        state.frequency_penalty, state.presence_penalty, state.seed};
}

void apply_options(InferenceState &state, const ModelOptions &options);

ggml_type startup_cache_type_k = GGML_TYPE_F16;
ggml_type startup_cache_type_v = GGML_TYPE_F16;
int startup_flash_attention = LLAMA_FLASH_ATTN_TYPE_AUTO;

InferenceState make_inference_state(llama_model *model, llama_context *context,
                                    int context_tokens, int max_output_tokens,
                                    int threads, int gpu_layers, float temperature,
                                    const char *model_file) {
    InferenceState state{};
    state.model = model;
    state.context = context;
    state.model_file = model_file ? model_file : "";
    ModelOptions options{context_tokens, max_output_tokens, threads, threads,
        gpu_layers, LLAMA_LOAD_MODE_AUTO, LLAMA_LAZY_MODE_AUTO, false, false,
        kPrefillBatch, true, true, startup_flash_attention,
        startup_cache_type_k, startup_cache_type_v, 0.0f, 0.0f, temperature,
        0, 1.0f, 0.0f, 64, 1.0f, 0.0f, 0.0f, LLAMA_DEFAULT_SEED};
    apply_options(state, options);
    return state;
}

void apply_options(InferenceState &state, const ModelOptions &options) {
    state.context_tokens = options.context_tokens;
    state.max_output_tokens = options.max_output_tokens;
    state.threads = options.threads;
    state.threads_batch = options.threads_batch;
    state.gpu_layers = options.gpu_layers;
    state.load_mode = options.load_mode;
    state.lazy_mode = options.lazy_mode;
    state.use_extra_bufts = options.use_extra_bufts;
    state.check_tensors = options.check_tensors;
    state.batch_tokens = options.batch_tokens;
    state.offload_kqv = options.offload_kqv;
    state.op_offload = options.op_offload;
    state.flash_attention = options.flash_attention;
    state.cache_type_k = options.cache_type_k;
    state.cache_type_v = options.cache_type_v;
    state.rope_freq_base = options.rope_freq_base;
    state.rope_freq_scale = options.rope_freq_scale;
    state.default_temperature = options.temperature;
    state.top_k = options.top_k;
    state.top_p = options.top_p;
    state.min_p = options.min_p;
    state.repeat_last_n = options.repeat_last_n;
    state.repeat_penalty = options.repeat_penalty;
    state.frequency_penalty = options.frequency_penalty;
    state.presence_penalty = options.presence_penalty;
    state.seed = options.seed;
}

const char *flash_attention_name(int value) {
    return value == LLAMA_FLASH_ATTN_TYPE_ENABLED ? "enabled" :
           value == LLAMA_FLASH_ATTN_TYPE_DISABLED ? "disabled" : "auto";
}

const char *backend_device_type_name(enum ggml_backend_dev_type type) {
    switch (type) {
    case GGML_BACKEND_DEVICE_TYPE_CPU: return "cpu";
    case GGML_BACKEND_DEVICE_TYPE_GPU: return "gpu";
    case GGML_BACKEND_DEVICE_TYPE_IGPU: return "integrated_gpu";
    case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "accelerator";
    case GGML_BACKEND_DEVICE_TYPE_META: return "meta";
    default: return "unknown";
    }
}

std::string options_json(const ModelOptions &o) {
    char values[1200];
    std::snprintf(values, sizeof(values),
        "{\"context_tokens\":%d,\"max_output_tokens\":%d,\"threads\":%d,\"threads_batch\":%d,"
        "\"gpu_layers\":%d,\"load_mode\":%d,\"lazy_mode\":%d,\"use_extra_bufts\":%s,\"check_tensors\":%s,"
        "\"batch_tokens\":%d,\"offload_kqv\":%s,\"op_offload\":%s,"
        "\"flash_attention\":\"%s\",\"cache_type_k\":\"%s\",\"cache_type_v\":\"%s\","
        "\"rope_freq_base\":%.6g,\"rope_freq_scale\":%.6g,\"temperature\":%.4f,"
        "\"top_k\":%d,\"top_p\":%.4f,\"min_p\":%.4f,\"repeat_last_n\":%d,"
        "\"repeat_penalty\":%.4f,\"frequency_penalty\":%.4f,\"presence_penalty\":%.4f,\"seed\":%s}",
        o.context_tokens, o.max_output_tokens, o.threads, o.threads_batch,
        o.gpu_layers, o.load_mode, o.lazy_mode, o.use_extra_bufts ? "true" : "false",
        o.check_tensors ? "true" : "false", o.batch_tokens, o.offload_kqv ? "true" : "false",
        o.op_offload ? "true" : "false", flash_attention_name(o.flash_attention),
        ggml_type_name(o.cache_type_k), ggml_type_name(o.cache_type_v),
        o.rope_freq_base, o.rope_freq_scale, o.temperature, o.top_k, o.top_p,
        o.min_p, o.repeat_last_n, o.repeat_penalty, o.frequency_penalty,
        o.presence_penalty, o.seed == LLAMA_DEFAULT_SEED ? "null" : std::to_string(o.seed).c_str());
    return values;
}

// role is LOAD or ROLLBACK; it tags the KV CACHE line so logs never attribute
// a rollback's cache to the model that failed.
bool create_model_context(const char *role, const std::string &filename, const ModelOptions &options,
                          llama_model **model_out, llama_context **context_out,
                          int *context_out_tokens, int *output_out_tokens,
                          LoadStream *stream = nullptr) {
    *model_out = nullptr;
    *context_out = nullptr;
    char path[160];
    std::snprintf(path, sizeof(path), "/app0/%s", filename.c_str());
    ModelLoadProgress progress;
    progress.started = clock_seconds();
    progress.stream = stream;
    progress.phase = std::strcmp(role, "ROLLBACK") == 0 ? "restoring_previous" : "loading";
    progress.model = filename;
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = options.gpu_layers;
    model_params.load_mode = static_cast<llama_load_mode>(options.load_mode);
    model_params.lazy_mode = static_cast<llama_lazy_mode>(options.lazy_mode);
    model_params.use_extra_bufts = options.use_extra_bufts;
    model_params.check_tensors = options.check_tensors;
    model_params.progress_callback = model_load_progress;
    model_params.progress_callback_user_data = &progress;
    status("LOADING MODEL FILE");
    llama_model *model = llama_model_load_from_file(path, model_params);
    if (!model) {
        statusf("%s FAILED READING MODEL %s", role, filename.c_str());
        return false;
    }

    int effective_context = options.context_tokens;
    const uint32_t trained_context = llama_model_n_ctx_train(model);
    if (trained_context && effective_context > static_cast<int>(trained_context))
        effective_context = static_cast<int>(trained_context);
    int effective_output = std::min(options.max_output_tokens, effective_context - 1);

    if (stream) stream->phase("creating_context", filename);
    auto context_params = llama_context_default_params();
    context_params.n_ctx = static_cast<uint32_t>(effective_context);
    context_params.n_batch = static_cast<uint32_t>(options.batch_tokens);
    context_params.n_ubatch = static_cast<uint32_t>(options.batch_tokens);
    context_params.n_threads = options.threads;
    context_params.n_threads_batch = options.threads_batch;
    context_params.offload_kqv = options.offload_kqv;
    context_params.op_offload = options.op_offload;
    context_params.flash_attn_type = static_cast<llama_flash_attn_type>(options.flash_attention);
    context_params.type_k = options.cache_type_k;
    context_params.type_v = options.cache_type_v;
    context_params.rope_freq_base = options.rope_freq_base;
    context_params.rope_freq_scale = options.rope_freq_scale;
    context_params.no_perf = false;
    llama_context *context = llama_init_from_model(model, context_params);
    if (!context) {
        statusf("%s FAILED CREATING CONTEXT %s K=%s V=%s CTX=%d FA=%s", role, filename.c_str(),
                ggml_type_name(options.cache_type_k), ggml_type_name(options.cache_type_v),
                effective_context, flash_attention_name(options.flash_attention));
        llama_model_free(model);
        return false;
    }
    statusf("KV CACHE %s %s K=%s V=%s CTX=%d FA=%s", role, filename.c_str(),
            ggml_type_name(options.cache_type_k), ggml_type_name(options.cache_type_v),
            effective_context, flash_attention_name(options.flash_attention));
    *model_out = model;
    *context_out = context;
    *context_out_tokens = effective_context;
    *output_out_tokens = effective_output;
    return true;
}

bool read_request(int fd, std::string &method, std::string &path, std::string &body,
                  std::string &authorization, int &error_code) {
    std::string raw;
    raw.reserve(4096);
    char chunk[2048];
    std::size_t boundary = std::string::npos;
    std::size_t content_length = 0;
    bool have_length = false;
    while (boundary == std::string::npos) {
        const ssize_t n = recv(fd, chunk, sizeof(chunk), 0);
        if (n < 0 && errno == EINTR) continue;
        // 0: the peer connected and left without sending anything (a port probe
        // or an abandoned request that waited behind a long generation).
        if (n <= 0) { error_code = raw.empty() ? 0 : 400; return false; }
        raw.append(chunk, static_cast<std::size_t>(n));
        boundary = raw.find("\r\n\r\n");
        if ((boundary == std::string::npos && raw.size() > kMaxHeader) || boundary > kMaxHeader) {
            error_code = 431;
            return false;
        }
    }
    const std::size_t line_end = raw.find("\r\n");
    if (line_end == std::string::npos) { error_code = 400; return false; }
    const std::string request_line = raw.substr(0, line_end);
    const std::size_t first = request_line.find(' '), second = request_line.find(' ', first + 1);
    if (first == std::string::npos || second == std::string::npos) { error_code = 400; return false; }
    method = request_line.substr(0, first);
    path = request_line.substr(first + 1, second - first - 1);
    std::size_t h = line_end + 2;
    while (h < boundary) {
        const std::size_t eol = raw.find("\r\n", h);
        if (eol == std::string::npos || eol > boundary) break;
        const std::string header = raw.substr(h, eol - h);
        const std::size_t colon = header.find(':');
        if (colon != std::string::npos) {
            std::string name = header.substr(0, colon);
            std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return char(std::tolower(c)); });
            std::size_t v = colon + 1;
            while (v < header.size() && (header[v] == ' ' || header[v] == '\t')) ++v;
            std::string value = header.substr(v);
            if (name == "content-length") {
                char *end = nullptr;
                const unsigned long n = std::strtoul(value.c_str(), &end, 10);
                if (end == value.c_str() || *end || n > kMaxBody) { error_code = 413; return false; }
                content_length = static_cast<std::size_t>(n);
                have_length = true;
            } else if (name == "transfer-encoding" && value != "identity") {
                error_code = 400;
                return false;
            } else if (name == "authorization") {
                authorization = value;
            }
        }
        h = eol + 2;
    }
    if (have_length && raw.size() - (boundary + 4) > content_length) {
        error_code = 400;
        return false;
    }
    if ((method == "POST") && !have_length) { error_code = 411; return false; }
    const std::size_t body_at = boundary + 4;
    while (raw.size() - body_at < content_length) {
        const ssize_t n = recv(fd, chunk, std::min(sizeof(chunk), content_length - (raw.size() - body_at)), 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { error_code = 400; return false; }
        raw.append(chunk, static_cast<std::size_t>(n));
    }
    body.assign(raw.data() + body_at, content_length);
    return true;
}

std::string base64_decode(const std::string &in, bool &ok);

// OpenAI-style content parts: [{"type":"text","text":...},
// {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}].
// Each image becomes a media marker in the text and its bytes in `images`.
bool parse_content_parts(const std::string &object, std::size_t at, std::string &content,
                         std::vector<std::string> &images) {
    ++at;  // past '['
    while ((at = skip_ws(object, at)) < object.size() && object[at] != ']') {
        if (object[at] == ',') { ++at; continue; }
        if (object[at] != '{') return false;
        const std::size_t start = at;
        int depth = 0;
        bool quoted = false, escaped = false;
        do {
            const char c = object[at++];
            if (quoted) {
                if (escaped) escaped = false;
                else if (c == '\\') escaped = true;
                else if (c == '"') quoted = false;
            } else if (c == '"') quoted = true;
            else if (c == '{') ++depth;
            else if (c == '}') --depth;
        } while (depth && at < object.size());
        if (depth) return false;
        const std::string part = object.substr(start, at - start);
        std::string type, text;
        if (!property_string(part, "type", type)) return false;
        if (type == "text") {
            if (!property_string(part, "text", text)) return false;
            content += text;
        } else if (type == "image_url") {
            // Base64 has no quotes or escapes, so the payload runs to the next quote.
            const std::size_t marker = part.find("base64,");
            const std::size_t end = marker == std::string::npos ? marker : part.find('"', marker);
            bool decoded = false;
            if (end == std::string::npos || images.size() >= 4) return false;
            images.push_back(base64_decode(part.substr(marker + 7, end - marker - 7), decoded));
            if (!decoded || images.back().empty()) return false;
            content += mtmd_default_marker();
            content += "\n";
        } else {
            return false;
        }
    }
    return at < object.size();
}

bool parse_messages(const std::string &body, std::vector<std::string> &roles,
                    std::vector<std::string> &contents, bool allow_empty = false,
                    std::vector<std::string> *images = nullptr) {
    std::size_t array_at;
    if (!find_property(body, "messages", 0, array_at) || array_at >= body.size() || body[array_at] != '[') return false;
    std::size_t at = array_at + 1;
    while ((at = skip_ws(body, at)) < body.size() && body[at] != ']') {
        if (body[at] == ',') { ++at; continue; }
        if (body[at++] != '{') return false;
        const std::size_t obj_start = at - 1;
        int depth = 1;
        bool quoted = false, escaped = false;
        for (; at < body.size() && depth; ++at) {
            const char c = body[at];
            if (quoted) {
                if (escaped) escaped = false;
                else if (c == '\\') escaped = true;
                else if (c == '"') quoted = false;
            } else if (c == '"') quoted = true;
            else if (c == '{') ++depth;
            else if (c == '}') --depth;
        }
        if (depth) return false;
        const std::string object = body.substr(obj_start, at - obj_start);
        std::string role, content;
        if (!property_string(object, "role", role)) return false;
        if (!property_string(object, "content", content)) {
            std::size_t parts_at = 0;
            if (!images || !find_property(object, "content", 0, parts_at) || parts_at >= object.size() ||
                object[parts_at] != '[' || !parse_content_parts(object, parts_at, content, *images))
                return false;
        }
        if (role != "system" && role != "user" && role != "assistant") return false;
        roles.push_back(std::move(role));
        contents.push_back(std::move(content));
        if (roles.size() > 64) return false;
    }
    return allow_empty || !roles.empty();
}

std::string model_list_json(const InferenceState &state, bool &ok) {
    DIR *directory = opendir("/app0");
    if (!directory) { ok = false; return {}; }
    std::string json = "{\"object\":\"list\",\"data\":[";
    bool first = true;
    struct dirent *entry;
    while ((entry = readdir(directory)) != nullptr) {
        const std::string filename(entry->d_name);
        if (!valid_model_filename(filename)) continue;
        char path[160];
        std::snprintf(path, sizeof(path), "/app0/%s", filename.c_str());
        struct stat info{};
        if (stat(path, &info) != 0 || !S_ISREG(info.st_mode)) continue;
        const bool loaded = filename == state.model_file && state.model != nullptr;
        char description[256] = {};
        if (loaded) llama_model_desc(state.model, description, sizeof(description));
        else std::snprintf(description, sizeof(description), "%s", filename.c_str());
        if (!first) json += ',';
        first = false;
        json += "{\"id\":\"" + std::string(loaded ? "ps5-local" : json_escape(filename)) +
                "\",\"object\":\"model\",\"owned_by\":\"local\",\"name\":\"" +
                json_escape(description) + "\",\"filename\":\"" + json_escape(filename) +
                "\",\"size_bytes\":" + std::to_string(static_cast<unsigned long long>(info.st_size)) +
                ",\"loaded\":" + (loaded ? "true" : "false") + "}";
    }
    closedir(directory);
    json += "]}";
    ok = true;
    return json;
}

std::string gpu_stats_json(const InferenceState &state) {
    std::string json = "{\"status\":\"ok\",\"memory_scope\":\"backend device heap; PS5 UMA may be shared with the system\",\"model\":{\"file\":\"" +
        json_escape(state.model_file) + "\",\"loaded\":" +
        std::string(state.model && state.context ? "true" : "false") +
        ",\"context_tokens\":" + std::to_string(state.context_tokens) +
        ",\"max_output_tokens\":" + std::to_string(state.max_output_tokens);
    if (state.model) {
        json += ",\"size_bytes\":" + std::to_string(static_cast<unsigned long long>(llama_model_size(state.model))) +
                ",\"parameters\":" + std::to_string(static_cast<unsigned long long>(llama_model_n_params(state.model)));
    }
    json += "}";
    struct DeviceStats {
        ggml_backend_dev_props props{};
        bool has_vulkan_budget = false;
        std::size_t heap_size = 0;
        std::size_t budget = 0;
        std::size_t usage = 0;
    };
    std::vector<DeviceStats> devices;
    devices.reserve(ggml_backend_dev_count());
    std::size_t gpu_free = 0, gpu_total = 0, gpu_budget = 0, gpu_usage = 0;
    int vulkan_device_index = 0;
    for (std::size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        if (!device) continue;
        DeviceStats entry{};
        ggml_backend_dev_get_props(device, &entry.props);
        const bool is_gpu = entry.props.type == GGML_BACKEND_DEVICE_TYPE_GPU ||
                            entry.props.type == GGML_BACKEND_DEVICE_TYPE_IGPU;
        if (is_gpu && entry.props.name && std::string(entry.props.name).find("Vulkan") != std::string::npos) {
            entry.has_vulkan_budget = ggml_backend_vk_get_device_memory_budget(
                vulkan_device_index++, &entry.budget, &entry.usage, &entry.heap_size);
        }
        if (is_gpu && entry.has_vulkan_budget) {
            gpu_free += entry.props.memory_free;
            gpu_total += entry.props.memory_total;
            gpu_budget += entry.budget;
            gpu_usage += entry.usage;
        } else if (is_gpu) {
            // Only use the generic backend values when the device is not Vulkan.
            gpu_free += entry.props.memory_free;
            gpu_total += entry.props.memory_total;
            gpu_budget += entry.props.memory_total;
            gpu_usage += entry.props.memory_total >= entry.props.memory_free
                ? entry.props.memory_total - entry.props.memory_free : entry.props.memory_total;
        }
        devices.push_back(entry);
    }
    const std::size_t gpu_used = gpu_total >= gpu_free ? gpu_total - gpu_free : gpu_total;
    const bool gpu_reported_usage_valid = gpu_usage <= gpu_total;
    json += ",\"vram\":{\"free_bytes\":" + std::to_string(static_cast<unsigned long long>(gpu_free)) +
            ",\"used_bytes\":" + std::to_string(static_cast<unsigned long long>(gpu_used)) +
            ",\"reported_usage_bytes\":" + std::to_string(static_cast<unsigned long long>(gpu_usage)) +
            ",\"total_bytes\":" + std::to_string(static_cast<unsigned long long>(gpu_total)) +
            ",\"budget_bytes\":" + std::to_string(static_cast<unsigned long long>(gpu_budget)) +
            ",\"used_estimate_source\":\"physical heap minus per-heap available budget\"" +
            ",\"reported_usage_valid\":" + (gpu_reported_usage_valid ? "true" : "false") +
            ",\"used_percent\":" + (gpu_total
                ? std::to_string(100.0 * static_cast<double>(gpu_used) / static_cast<double>(gpu_total))
                : "null") +
            ",\"reported_budget_usage_percent\":" + (gpu_budget
                ? std::to_string(100.0 * static_cast<double>(gpu_usage) / static_cast<double>(gpu_budget))
                : "null") +
            ",\"warning\":\"" + (gpu_reported_usage_valid
                ? "" : "Vulkan driver raw usage exceeds physical heap; used_bytes is a headroom-based estimate") +
            "\"},\"devices\":[";
    bool first = true;
    for (const auto &entry : devices) {
        const auto &props = entry.props;
        if (!first) json += ',';
        first = false;
        const bool is_gpu = props.type == GGML_BACKEND_DEVICE_TYPE_GPU ||
                            props.type == GGML_BACKEND_DEVICE_TYPE_IGPU;
        const std::size_t total = props.memory_total;
        const std::size_t budget = entry.has_vulkan_budget ? entry.budget : props.memory_total;
        const std::size_t used = props.memory_total >= props.memory_free
            ? props.memory_total - props.memory_free : props.memory_total;
        const std::size_t free = props.memory_free;
        const std::size_t reported_usage = entry.has_vulkan_budget ? entry.usage : used;
        const bool usage_valid = !is_gpu || !entry.has_vulkan_budget || reported_usage <= total;
        json += "{\"name\":\"" + json_escape(props.name ? props.name : "unknown") +
                "\",\"description\":\"" + json_escape(props.description ? props.description : "") +
                "\",\"device_id\":" + (props.device_id ? "\"" + json_escape(props.device_id) + "\"" : "null") +
                ",\"type\":\"" + backend_device_type_name(props.type) + "\"" +
                ",\"memory_free_bytes\":" + std::to_string(static_cast<unsigned long long>(free)) +
                ",\"memory_used_bytes\":" + (usage_valid
                    ? std::to_string(static_cast<unsigned long long>(used)) : "null") +
                ",\"reported_memory_usage_bytes\":" + std::to_string(static_cast<unsigned long long>(reported_usage)) +
                ",\"memory_total_bytes\":" + std::to_string(static_cast<unsigned long long>(total)) +
                ",\"memory_budget_bytes\":" + (entry.has_vulkan_budget || is_gpu
                    ? std::to_string(static_cast<unsigned long long>(budget)) : "null") +
                ",\"memory_usage_valid\":" + (usage_valid ? "true" : "false") +
                ",\"memory_used_percent\":" + (total
                    ? std::to_string(100.0 * static_cast<double>(used) / static_cast<double>(total))
                    : "null") + "}";
    }
    json += "]}";
    return json;
}

bool set_optional_int(const std::string &body, const char *key, int minimum, int maximum,
                      int &value) {
    std::size_t at = 0;
    if (!find_property(body, key, 0, at)) return true;
    return property_int(body, key, value) && value >= minimum && value <= maximum;
}

bool set_optional_float(const std::string &body, const char *key, float minimum, float maximum,
                        float &value) {
    std::size_t at = 0;
    if (!find_property(body, key, 0, at)) return true;
    return property_float(body, key, value) && value >= minimum && value <= maximum;
}

bool set_optional_bool(const std::string &body, const char *key, bool &value) {
    std::size_t at = 0;
    if (!find_property(body, key, 0, at)) return true;
    return property_bool(body, key, value);
}

bool set_optional_cache_type(const std::string &body, const char *key, ggml_type &value) {
    std::size_t at = 0;
    if (!find_property(body, key, 0, at)) return true;
    std::string name;
    if (!property_string(body, key, name)) return false;
    if (name == "f16") value = GGML_TYPE_F16;
    else if (name == "q8_0") value = GGML_TYPE_Q8_0;
    else if (name == "q4_0") value = GGML_TYPE_Q4_0;
    else return false;
    return true;
}

bool parse_load_options(const std::string &body, ModelOptions &o) {
    std::size_t max_tokens_at = 0, max_output_at = 0;
    const bool has_max_tokens = find_property(body, "max_tokens", 0, max_tokens_at);
    const bool has_max_output = find_property(body, "max_output_tokens", 0, max_output_at);
    int alias_max_tokens = o.max_output_tokens;
    int seed = 0;
    std::size_t seed_at = 0;
    const bool has_seed = find_property(body, "seed", 0, seed_at);
    if (has_max_tokens && !set_optional_int(body, "max_tokens", 1, 4096, alias_max_tokens)) return false;
    if (has_max_tokens && has_max_output) return false;
    if (has_max_tokens) o.max_output_tokens = alias_max_tokens;
    if (has_seed) {
        if (!set_optional_int(body, "seed", -1, 0x7fffffff, seed)) return false;
        o.seed = seed < 0 ? LLAMA_DEFAULT_SEED : static_cast<uint32_t>(seed);
    }
    return set_optional_int(body, "context_tokens", 512, 32768, o.context_tokens) &&
        set_optional_int(body, "max_output_tokens", 1, 4096, o.max_output_tokens) &&
        set_optional_int(body, "threads", 1, 4, o.threads) &&
        set_optional_int(body, "threads_batch", 1, 4, o.threads_batch) &&
        set_optional_int(body, "gpu_layers", 0, 999, o.gpu_layers) &&
        set_optional_int(body, "load_mode", LLAMA_LOAD_MODE_AUTO, LLAMA_LOAD_MODE_DIRECT_IO, o.load_mode) &&
        set_optional_int(body, "lazy_mode", LLAMA_LAZY_MODE_OFF, LLAMA_LAZY_MODE_ON, o.lazy_mode) &&
        set_optional_bool(body, "use_extra_bufts", o.use_extra_bufts) &&
        set_optional_bool(body, "check_tensors", o.check_tensors) &&
        set_optional_int(body, "batch_tokens", 16, kMaxSafeBatch, o.batch_tokens) &&
        set_optional_bool(body, "offload_kqv", o.offload_kqv) &&
        set_optional_bool(body, "op_offload", o.op_offload) &&
        set_optional_int(body, "flash_attention", -1, 1, o.flash_attention) &&
        set_optional_cache_type(body, "cache_type_k", o.cache_type_k) &&
        set_optional_cache_type(body, "cache_type_v", o.cache_type_v) &&
        set_optional_float(body, "rope_freq_base", 0.0f, 1000000.0f, o.rope_freq_base) &&
        set_optional_float(body, "rope_freq_scale", 0.0f, 100.0f, o.rope_freq_scale) &&
        set_optional_float(body, "temperature", 0.0f, 2.0f, o.temperature) &&
        set_optional_int(body, "top_k", 0, 1000, o.top_k) &&
        set_optional_float(body, "top_p", 0.01f, 1.0f, o.top_p) &&
        set_optional_float(body, "min_p", 0.0f, 1.0f, o.min_p) &&
        set_optional_int(body, "repeat_last_n", 0, 4096, o.repeat_last_n) &&
        set_optional_float(body, "repeat_penalty", 0.5f, 2.0f, o.repeat_penalty) &&
        set_optional_float(body, "frequency_penalty", -2.0f, 2.0f, o.frequency_penalty) &&
        set_optional_float(body, "presence_penalty", -2.0f, 2.0f, o.presence_penalty) &&
        o.max_output_tokens < o.context_tokens;
}

void handle_model_load(int fd, const std::string &body, InferenceState &state) {
    std::string filename;
    if (!property_string(body, "model", filename) || !valid_model_filename(filename)) {
        send_response(fd, 422, "Unprocessable Entity", "application/json",
                      "{\"error\":{\"message\":\"model must be a GGUF filename from GET /v1/models\",\"type\":\"invalid_request_error\"}}");
        status("MODEL LOAD REJECTED - INVALID NAME");
        return;
    }
    ModelOptions requested = options_from_state(state);
    // KV cache types and flash attention decide whether a model fits; never
    // inherit them from the previous model. Omitted means the defaults.
    requested.cache_type_k = GGML_TYPE_F16;
    requested.cache_type_v = GGML_TYPE_F16;
    requested.flash_attention = LLAMA_FLASH_ATTN_TYPE_AUTO;
    if (!parse_load_options(body, requested)) {
        send_response(fd, 422, "Unprocessable Entity", "application/json",
                      "{\"error\":{\"message\":\"invalid load parameters; see GET /health; supported options: context_tokens, max_tokens/max_output_tokens, threads/threads_batch, gpu_layers, load_mode (-1..4), lazy_mode (0..2), use_extra_bufts, check_tensors, batch_tokens (16..128), offload_kqv, op_offload, flash_attention (-1/0/1), cache_type_k/v (f16/q8_0/q4_0), rope_freq_base/scale, temperature, top_k/top_p/min_p, repeat_last_n/repeat_penalty, frequency/presence_penalty, seed\",\"type\":\"invalid_request_error\"}}");
        status("MODEL LOAD REJECTED - INVALID PARAMETERS");
        return;
    }
    char path[160];
    std::snprintf(path, sizeof(path), "/app0/%s", filename.c_str());
    struct stat info{};
    if (stat(path, &info) != 0 || !S_ISREG(info.st_mode) || info.st_size <= 0) {
        send_response(fd, 404, "Not Found", "application/json",
                      "{\"error\":{\"message\":\"GGUF model file is not available in the app folder\",\"type\":\"invalid_request_error\"}}");
        status("MODEL LOAD REJECTED - FILE NOT FOUND");
        return;
    }
    // Validation errors above stay plain HTTP errors; from here a streaming
    // client gets SSE, the last event being the JSON others get as the body.
    bool want_stream = false;
    set_optional_bool(body, "stream", want_stream);
    LoadStream stream;
    stream.fd = fd;
    stream.started = clock_seconds();
    if (want_stream) stream.open = begin_sse(fd);
    auto finish = [&](int code, const char *reason, const std::string &json) {
        if (!want_stream) {
            send_response(fd, code, reason, "application/json", json);
            return;
        }
        std::string last = json;
        last.insert(last.size() - 1, std::string(",\"http_status\":") + std::to_string(code) +
                    ",\"elapsed_s\":" + std::to_string(clock_seconds() - stream.started));
        stream.event(last);
        stream.event("[DONE]");
        if (stream.open) end_sse(fd);
    };
    const ModelOptions current = options_from_state(state);
    if (filename == state.model_file && state.model &&
        options_json(requested) == options_json(current)) {
        const std::string response = "{\"status\":\"already_loaded\",\"model_file\":\"" +
            json_escape(filename) + "\",\"parameters\":" + options_json(current) + "}";
        finish(200, "OK", response);
        status("MODEL ALREADY LOADED");
        return;
    }

    const std::string previous_file = state.model_file;
    const ModelOptions previous_options = current;
    llama_context *previous_context = state.context;
    llama_model *previous_model = state.model;
    state.loading = true;
    char start[192];
    std::snprintf(start, sizeof(start), "SWITCHING MODEL CTX=%d OUT=%d GPU=%d THR=%d",
                  requested.context_tokens, requested.max_output_tokens,
                  requested.gpu_layers, requested.threads);
    status(start);
    if (!sd_service_loaded_model().empty()) {
        status("UNLOADING IMAGE MODEL");
        stream.phase("unloading", sd_service_loaded_model());
        sd_service_unload();
    }
    if (static_cast<uint64_t>(info.st_size) > kSmallLlmBytes) {
        stt_unload();
        tts_unload();
    }
    status("UNLOADING PREVIOUS MODEL");
    if (!previous_file.empty()) stream.phase("unloading", previous_file);
    state.context = nullptr;
    state.model = nullptr;
    vision_release();
    vla_release();
    llama_free(previous_context);
    llama_model_free(previous_model);

    int effective_context = requested.context_tokens;
    int effective_output = requested.max_output_tokens;
    llama_model *new_model = nullptr;
    llama_context *new_context = nullptr;
    const bool loaded = create_model_context("LOAD", filename, requested, &new_model, &new_context,
                                              &effective_context, &effective_output, &stream);
    if (!loaded && previous_file.empty()) {
        state.loading = false;
        finish(503, "Service Unavailable",
               "{\"error\":{\"message\":\"requested model failed; no previous model to restore\",\"type\":\"model_load_error\"}}");
        status("MODEL LOAD FAILED - NO PREVIOUS MODEL");
        return;
    }
    if (!loaded) {
        status("MODEL LOAD FAILED - RESTORING PREVIOUS MODEL");
        int restored_context_tokens = previous_options.context_tokens;
        int restored_max_output = previous_options.max_output_tokens;
        llama_model *restored_model = nullptr;
        llama_context *restored_context = nullptr;
        stream.phase("restoring_previous", previous_file);
        const bool restored = create_model_context("ROLLBACK", previous_file, previous_options,
            &restored_model, &restored_context,
            &restored_context_tokens, &restored_max_output, &stream);
        state.model = restored_model;
        state.context = restored_context;
        state.model_file = previous_file;
        ModelOptions restored_options = previous_options;
        restored_options.context_tokens = restored_context_tokens;
        restored_options.max_output_tokens = restored_max_output;
        apply_options(state, restored_options);
        state.loading = false;
        if (restored) {
            char message[320];
            std::snprintf(message, sizeof(message),
                          "{\"error\":{\"message\":\"requested model failed; previous model restored\",\"type\":\"model_load_error\"},\"active_model\":\"%s\"}",
                          json_escape(previous_file).c_str());
            finish(503, "Service Unavailable", message);
            char loaded_status[192];
            std::snprintf(loaded_status, sizeof(loaded_status), "ROLLBACK LOADED %s", previous_file.c_str());
            status(loaded_status);
        } else {
            finish(500, "Internal Server Error",
                   "{\"error\":{\"message\":\"requested model and rollback model both failed to load\",\"type\":\"model_load_error\"}}");
            status("ROLLBACK MODEL LOAD FAILED");
        }
        return;
    }

    state.model = new_model;
    state.context = new_context;
    state.model_file = filename;
    requested.context_tokens = effective_context;
    requested.max_output_tokens = effective_output;
    apply_options(state, requested);
    state.loading = false;
    char description[192] = {};
    llama_model_desc(state.model, description, sizeof(description));
    status("MODEL LOADED");
    status(description);
    const std::string response = "{\"status\":\"loaded\",\"model_file\":\"" +
        json_escape(filename) + "\",\"model_name\":\"" + json_escape(description) +
        "\",\"parameters\":" + options_json(options_from_state(state)) + "}";
    finish(200, "OK", response);
    status("DYNAMIC MODEL SWITCH COMPLETE");
}

void free_gpu_for(const char *engine, InferenceState &state, LoadStream *stream);

// POST /v1/images/generations: OpenAI-style text to image. The image model and
// the LLM do not both fit, so this unloads the LLM; the image model then stays
// loaded for further image requests until an LLM is loaded again.
void handle_image_generation(int fd, const std::string &body, InferenceState &state) {
    SdImageRequest request;
    std::string model = sd_service_loaded_model().empty() ? "sd-turbo-fp16" : sd_service_loaded_model();
    std::string prediction, size;
    int seed = -1;
    bool want_stream = false;
    bool ok = property_string(body, "prompt", request.prompt) && !request.prompt.empty();
    std::size_t at = 0;
    if (ok && find_property(body, "model", 0, at)) ok = property_string(body, "model", model) && !model.empty() &&
        model.find("..") == std::string::npos && model[0] != '/';
    if (ok && find_property(body, "negative_prompt", 0, at)) ok = property_string(body, "negative_prompt", request.negative_prompt);
    if (ok && find_property(body, "sample_method", 0, at)) ok = property_string(body, "sample_method", request.sample_method);
    if (ok && find_property(body, "prediction", 0, at)) ok = property_string(body, "prediction", prediction);
    if (ok && find_property(body, "size", 0, at)) ok = property_string(body, "size", size) &&
        std::sscanf(size.c_str(), "%dx%d", &request.width, &request.height) == 2;
    // Omitted size/steps/cfg_scale (0 / -1) take the model's defaults.
    ok = ok && set_optional_int(body, "width", 256, 1024, request.width) &&
         set_optional_int(body, "height", 256, 1024, request.height) &&
         set_optional_int(body, "steps", 1, 100, request.steps) &&
         set_optional_float(body, "cfg_scale", 0.0f, 30.0f, request.cfg_scale) &&
         set_optional_int(body, "seed", -1, 0x7fffffff, seed) &&
         set_optional_bool(body, "stream", want_stream) &&
         set_optional_float(body, "strength", 0.0f, 1.0f, request.strength) &&
         request.width % 64 == 0 && request.height % 64 == 0 &&
         (request.width == 0) == (request.height == 0) &&
         (request.width == 0 || (request.width >= 256 && request.width <= 1024 &&
                                 request.height >= 256 && request.height <= 1024));
    // Editing: "init_image_b64" (image to image, with "strength" 0..1) and
    // "ref_images_b64" (up to 4 reference images, for models that take them).
    std::string encoded;
    bool decoded = false;
    if (ok && find_property(body, "init_image_b64", 0, at)) {
        ok = property_string(body, "init_image_b64", encoded);
        if (ok) { request.init_image = base64_decode(encoded, decoded); ok = decoded && !request.init_image.empty(); }
    }
    if (ok && find_property(body, "ref_images_b64", 0, at)) {
        // An array of base64 strings; base64 has no quotes or escapes to worry about.
        ok = at < body.size() && body[at] == '[';
        const std::size_t end = ok ? body.find(']', at) : std::string::npos;
        ok = ok && end != std::string::npos;
        for (std::size_t open = at; ok && (open = body.find('"', open + 1)) < end;) {
            const std::size_t close = body.find('"', open + 1);
            if (close == std::string::npos || close > end || request.ref_images.size() >= 4) { ok = false; break; }
            request.ref_images.push_back(base64_decode(body.substr(open + 1, close - open - 1), decoded));
            ok = decoded && !request.ref_images.back().empty();
            open = close;
        }
    }
    if (!ok) {
        send_response(fd, 422, "Unprocessable Entity", "application/json",
                      "{\"error\":{\"message\":\"invalid image request; prompt is required; optional: model (folder in the app directory), size WxH or width/height (multiples of 64, 256..1024 per side), steps 1..100, cfg_scale, seed, negative_prompt, sample_method, prediction, stream, init_image_b64 with strength 0..1, ref_images_b64 (up to 4); omitted values use the model folder defaults (sd-model.conf)\",\"type\":\"invalid_request_error\"}}");
        status("BAD IMAGE REQUEST");
        return;
    }
    request.seed = seed;
    struct stat info{};
    if (stat(("/app0/" + model).c_str(), &info) != 0) {
        send_response(fd, 404, "Not Found", "application/json",
                      "{\"error\":{\"message\":\"image model not found in the app folder\",\"type\":\"invalid_request_error\"}}");
        status("IMAGE MODEL NOT FOUND");
        return;
    }

    LoadStream stream;
    stream.fd = fd;
    stream.started = clock_seconds();
    if (want_stream) stream.open = begin_sse(fd);
    auto finish = [&](int code, const char *reason, const std::string &json) {
        if (!want_stream) {
            send_response(fd, code, reason, "application/json", json);
            return;
        }
        std::string last = json;
        last.insert(last.size() - 1, std::string(",\"http_status\":") + std::to_string(code));
        stream.event(last);
        stream.event("[DONE]");
        if (stream.open) end_sse(fd);
    };

    free_gpu_for("image", state, &stream);
    int last_percent = -1, last_step = -1;
    const SdProgress progress = [&](const char *phase, int value, int total) {
        if (std::strcmp(phase, "loading") == 0) {
            const int percent = total > 0 ? value * 100 / total : 0;
            if (percent == last_percent) return;
            if (percent / 10 != last_percent / 10) statusf("LOADING IMAGE MODEL %d%%", percent);
            last_percent = percent;
            char json[256];
            std::snprintf(json, sizeof(json), "{\"status\":\"loading\",\"model\":\"%s\",\"progress\":%d,\"elapsed_s\":%.1f}",
                          json_escape(model).c_str(), percent, clock_seconds() - stream.started);
            stream.event(json);
        } else {
            if (value == last_step) return;
            last_step = value;
            if (std::strcmp(phase, "generating") == 0) statusf("IMAGE STEP %d/%d", value, total);
            char json[192];
            std::snprintf(json, sizeof(json), "{\"status\":\"%s\",\"step\":%d,\"steps\":%d,\"elapsed_s\":%.1f}",
                          phase, value, total, clock_seconds() - stream.started);
            stream.event(json);
        }
    };
    std::string error;
    double load_s = 0.0;
    if (sd_service_loaded_model() != model) {
        statusf("LOADING IMAGE MODEL %s", model.c_str());
        if (!sd_service_load(model, prediction, error, progress, &load_s)) {
            finish(500, "Internal Server Error", "{\"error\":{\"message\":\"" + json_escape(error) +
                   "\",\"type\":\"model_load_error\"}}");
            status("IMAGE MODEL LOAD FAILED");
            return;
        }
        statusf("IMAGE MODEL LOADED IN %.1F S", load_s);
    }
    SdImageResult result;
    if (!sd_service_generate(request, result, error, progress)) {
        finish(500, "Internal Server Error", "{\"error\":{\"message\":\"" + json_escape(error) +
               "\",\"type\":\"server_error\"}}");
        status("IMAGE GENERATION FAILED");
        return;
    }
    statusf("IMAGE DONE %dX%d %d STEPS IN %.1F S", result.width, result.height, result.steps, result.generate_s);
    char head[512];
    std::snprintf(head, sizeof(head),
                  "{\"created\":0,\"model\":\"%s\",\"seed\":%lld,\"width\":%d,\"height\":%d,\"steps\":%d,"
                  "\"cfg_scale\":%.2f,\"timings\":{\"load_s\":%.2f,\"generate_s\":%.2f},\"data\":[{\"b64_json\":\"",
                  json_escape(model).c_str(), static_cast<long long>(result.seed), result.width, result.height,
                  result.steps, result.cfg_scale, load_s, result.generate_s);
    finish(200, "OK", head + base64_encode(result.png) + "\"}]}");
}

std::string base64_decode(const std::string &in, bool &ok) {
    static int8_t table[256];
    static bool ready = false;
    if (!ready) {
        std::memset(table, -1, sizeof(table));
        const char *alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) table[static_cast<unsigned char>(alphabet[i])] = static_cast<int8_t>(i);
        ready = true;
    }
    std::string out;
    out.reserve(in.size() / 4 * 3);
    uint32_t buffer = 0;
    int bits = 0;
    ok = true;
    for (unsigned char c : in) {
        if (c == '=' || c == '\n' || c == '\r' || c == ' ') continue;
        if (table[c] < 0) { ok = false; return {}; }
        buffer = buffer << 6 | static_cast<uint32_t>(table[c]);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out += static_cast<char>((buffer >> bits) & 0xff);
        }
    }
    return out;
}

// The GPU heap holds one big engine at a time: before an engine loads, every
// other one is unloaded (the LLM, the image/video model, speech models).
void free_gpu_for(const char *engine, InferenceState &state, LoadStream *stream) {
    const bool speech = std::strcmp(engine, "stt") == 0 || std::strcmp(engine, "tts") == 0 ||
                        std::strcmp(engine, "voice") == 0;
    const bool keep_llm = speech && state.model && llama_model_size(state.model) <= kSmallLlmBytes;
    if (state.model && !keep_llm) {
        statusf("UNLOADING LLM %s FOR %s", state.model_file.c_str(), engine);
        if (stream) stream->phase("unloading", state.model_file);
        vision_release();
        vla_release();
        llama_free(state.context);
        llama_model_free(state.model);
        state.context = nullptr;
        state.model = nullptr;
        state.model_file.clear();
    }
    if (std::strcmp(engine, "image") != 0 && !sd_service_loaded_model().empty()) {
        if (stream) stream->phase("unloading", sd_service_loaded_model());
        sd_service_unload();
    }
    // The two speech models are small and stay loaded together.
    if (!speech && !stt_loaded_model().empty()) {
        if (stream) stream->phase("unloading", stt_loaded_model());
        stt_unload();
    }
    if (!speech && !tts_loaded_model().empty()) {
        if (stream) stream->phase("unloading", tts_loaded_model());
        tts_unload();
    }
}

bool safe_model_path(const std::string &model) {
    return !model.empty() && model.find("..") == std::string::npos && model[0] != '/';
}

// POST /v1/audio/transcriptions: {"audio_b64": WAV, "model", "language", "translate"}.
void handle_transcription(int fd, const std::string &body, InferenceState &state) {
    std::string model = stt_loaded_model().empty() ? "whisper/ggml-large-v3-turbo-q5_0.bin" : stt_loaded_model();
    std::string audio_b64, language;
    bool translate = false;
    std::size_t at = 0;
    bool ok = property_string(body, "audio_b64", audio_b64) && !audio_b64.empty() &&
              set_optional_bool(body, "translate", translate);
    if (ok && find_property(body, "model", 0, at)) ok = property_string(body, "model", model) && safe_model_path(model);
    if (ok && find_property(body, "language", 0, at)) ok = property_string(body, "language", language);
    std::vector<float> samples;
    std::string error;
    if (ok) {
        bool decoded = false;
        const std::string wav = base64_decode(audio_b64, decoded);
        ok = decoded && decode_wav(wav, 16000, samples, error);
        if (!decoded) error = "audio_b64 is not valid base64";
    }
    if (!ok) {
        send_response(fd, 422, "Unprocessable Entity", "application/json",
                      "{\"error\":{\"message\":\"" + json_escape(error.empty() ? "audio_b64 (base64 WAV) is required; optional model, language, translate" : error) +
                      "\",\"type\":\"invalid_request_error\"}}");
        status("BAD TRANSCRIPTION REQUEST");
        return;
    }
    free_gpu_for("stt", state, nullptr);
    double load_s = 0.0;
    if (stt_loaded_model() != model) {
        statusf("LOADING STT MODEL %s", model.c_str());
        if (!stt_load(model, error, &load_s)) {
            send_response(fd, 500, "Internal Server Error", "application/json",
                          "{\"error\":{\"message\":\"" + json_escape(error) + "\",\"type\":\"model_load_error\"}}");
            status("STT MODEL LOAD FAILED");
            return;
        }
    }
    statusf("TRANSCRIBING %.1F S OF AUDIO", samples.size() / 16000.0);
    SttResult result;
    if (!stt_transcribe(samples, language, translate, result, error)) {
        send_response(fd, 500, "Internal Server Error", "application/json",
                      "{\"error\":{\"message\":\"" + json_escape(error) + "\",\"type\":\"server_error\"}}");
        status("TRANSCRIPTION FAILED");
        return;
    }
    result.load_s = load_s;
    std::string segments;
    for (const auto &segment : result.segments) {
        char head[96];
        std::snprintf(head, sizeof(head), "{\"start\":%.2f,\"end\":%.2f,\"text\":\"", segment.start_s, segment.end_s);
        segments += (segments.empty() ? "" : ",") + std::string(head) + json_escape(segment.text) + "\"}";
    }
    char timings[192];
    std::snprintf(timings, sizeof(timings), "{\"audio_s\":%.2f,\"load_s\":%.2f,\"transcribe_s\":%.2f}",
                  result.audio_s, result.load_s, result.transcribe_s);
    send_response(fd, 200, "OK", "application/json",
                  "{\"text\":\"" + json_escape(result.text) + "\",\"language\":\"" + json_escape(result.language) +
                  "\",\"model\":\"" + json_escape(model) + "\",\"segments\":[" + segments + "],\"timings\":" + timings + "}");
    statusf("TRANSCRIBED %.1F S AUDIO IN %.2F S", result.audio_s, result.transcribe_s);
}

// POST /v1/audio/speech: {"input", "model", "language", "speaker_b64" (WAV),
// "seed", "max_frames", "response_format": "json" (default, audio_b64) | "wav"}.
void handle_speech(int fd, const std::string &body, InferenceState &state) {
    std::string model = tts_loaded_model().empty() ? "qwen3-tts-1.7b" : tts_loaded_model();
    std::string speaker_b64, format = "json";
    TtsRequest request;
    int seed = -1;
    std::size_t at = 0;
    bool ok = property_string(body, "input", request.text) && !request.text.empty() &&
              set_optional_int(body, "seed", -1, 0x7fffffff, seed) &&
              set_optional_int(body, "max_frames", 12, 4096, request.max_frames);
    if (ok && find_property(body, "model", 0, at)) ok = property_string(body, "model", model) && safe_model_path(model);
    if (ok && find_property(body, "language", 0, at)) ok = property_string(body, "language", request.language);
    if (ok && find_property(body, "response_format", 0, at)) ok = property_string(body, "response_format", format) &&
        (format == "json" || format == "wav");
    if (ok && find_property(body, "speaker_b64", 0, at) && property_string(body, "speaker_b64", speaker_b64)) {
        bool decoded = false;
        request.speaker_wav = base64_decode(speaker_b64, decoded);
        ok = decoded;
    }
    if (!ok) {
        send_response(fd, 422, "Unprocessable Entity", "application/json",
                      "{\"error\":{\"message\":\"input is required; optional model, language, speaker_b64 (base64 WAV), seed, max_frames, response_format json|wav\",\"type\":\"invalid_request_error\"}}");
        status("BAD SPEECH REQUEST");
        return;
    }
    request.seed = seed;
    free_gpu_for("tts", state, nullptr);
    std::string error;
    double load_s = 0.0;
    if (tts_loaded_model() != model) {
        statusf("LOADING TTS MODEL %s", model.c_str());
        if (!tts_load(model, error, &load_s)) {
            send_response(fd, 500, "Internal Server Error", "application/json",
                          "{\"error\":{\"message\":\"" + json_escape(error) + "\",\"type\":\"model_load_error\"}}");
            status("TTS MODEL LOAD FAILED");
            return;
        }
    }
    statusf("SPEAKING %zu CHARACTERS", request.text.size());
    TtsResult result;
    if (!tts_generate(request, result, error, [](int frames, int) { statusf("TTS FRAMES %d", frames); })) {
        send_response(fd, 500, "Internal Server Error", "application/json",
                      "{\"error\":{\"message\":\"" + json_escape(error) + "\",\"type\":\"server_error\"}}");
        status("SPEECH FAILED");
        return;
    }
    result.load_s = load_s;
    statusf("SPOKE %.1F S AUDIO IN %.1F S", result.audio_s, result.generate_s);
    if (format == "wav") {
        send_response(fd, 200, "OK", "audio/wav", result.wav);
        return;
    }
    char head[320];
    std::snprintf(head, sizeof(head),
                  "{\"model\":\"%s\",\"sample_rate\":%d,\"frames\":%d,\"timings\":{\"audio_s\":%.2f,\"load_s\":%.2f,"
                  "\"generate_s\":%.2f},\"audio_b64\":\"", json_escape(model).c_str(), result.sample_rate, result.frames,
                  result.audio_s, result.load_s, result.generate_s);
    const std::vector<unsigned char> wav(result.wav.begin(), result.wav.end());
    send_response(fd, 200, "OK", "application/json", head + base64_encode(wav) + "\"}");
}

// POST /v1/videos/generations: like /v1/images/generations for video models
// (sd-model.conf mode=video), plus "frames"; returns PNG frames and fps.
void handle_video_generation(int fd, const std::string &body, InferenceState &state) {
    SdVideoRequest request;
    std::string model = "wan2.1-t2v-1.3b", size, init_image_b64;
    int seed = -1;
    bool want_stream = false;
    std::size_t at = 0;
    bool ok = property_string(body, "prompt", request.prompt) && !request.prompt.empty();
    if (ok && find_property(body, "model", 0, at)) ok = property_string(body, "model", model) && safe_model_path(model);
    if (ok && find_property(body, "negative_prompt", 0, at)) ok = property_string(body, "negative_prompt", request.negative_prompt);
    if (ok && find_property(body, "size", 0, at)) ok = property_string(body, "size", size) &&
        std::sscanf(size.c_str(), "%dx%d", &request.width, &request.height) == 2;
    if (ok && find_property(body, "init_image_b64", 0, at) && property_string(body, "init_image_b64", init_image_b64)) {
        bool decoded = false;
        request.init_image = base64_decode(init_image_b64, decoded);
        ok = decoded && !request.init_image.empty();
    }
    ok = ok && set_optional_int(body, "width", 128, 1280, request.width) &&
         set_optional_int(body, "height", 128, 1280, request.height) &&
         set_optional_int(body, "frames", 1, 121, request.frames) &&
         set_optional_int(body, "steps", 1, 100, request.steps) &&
         set_optional_float(body, "cfg_scale", 0.0f, 30.0f, request.cfg_scale) &&
         set_optional_int(body, "seed", -1, 0x7fffffff, seed) &&
         set_optional_bool(body, "stream", want_stream) &&
         request.width % 16 == 0 && request.height % 16 == 0 &&
         (request.frames == 0 || request.frames % 4 == 1);
    if (!ok) {
        send_response(fd, 422, "Unprocessable Entity", "application/json",
                      "{\"error\":{\"message\":\"prompt is required; optional model, size WxH (multiples of 16), frames (4k+1), steps, cfg_scale, seed, negative_prompt, init_image_b64 (PNG/JPEG first frame for image-to-video models), stream\",\"type\":\"invalid_request_error\"}}");
        status("BAD VIDEO REQUEST");
        return;
    }
    request.seed = seed;
    struct stat info{};
    if (stat(("/app0/" + model).c_str(), &info) != 0) {
        send_response(fd, 404, "Not Found", "application/json",
                      "{\"error\":{\"message\":\"video model not found in the app folder\",\"type\":\"invalid_request_error\"}}");
        return;
    }
    LoadStream stream;
    stream.fd = fd;
    stream.started = clock_seconds();
    if (want_stream) stream.open = begin_sse(fd);
    auto finish = [&](int code, const char *reason, const std::string &json) {
        if (!want_stream) {
            send_response(fd, code, reason, "application/json", json);
            return;
        }
        std::string last = json;
        last.insert(last.size() - 1, std::string(",\"http_status\":") + std::to_string(code));
        stream.event(last);
        stream.event("[DONE]");
        if (stream.open) end_sse(fd);
    };
    free_gpu_for("image", state, &stream);
    int last_value = -1;
    const SdProgress progress = [&](const char *phase, int value, int total) {
        if (value == last_value) return;
        last_value = value;
        if (std::strcmp(phase, "generating") == 0) statusf("VIDEO STEP %d/%d", value, total);
        char json[192];
        std::snprintf(json, sizeof(json), "{\"status\":\"%s\",\"step\":%d,\"steps\":%d,\"elapsed_s\":%.1f}",
                      phase, value, total, clock_seconds() - stream.started);
        stream.event(json);
    };
    std::string error;
    double load_s = 0.0;
    if (sd_service_loaded_model() != model) {
        statusf("LOADING VIDEO MODEL %s", model.c_str());
        if (!sd_service_load(model, "", error, progress, &load_s)) {
            finish(500, "Internal Server Error", "{\"error\":{\"message\":\"" + json_escape(error) + "\",\"type\":\"model_load_error\"}}");
            status("VIDEO MODEL LOAD FAILED");
            return;
        }
    }
    SdVideoResult result;
    if (!sd_service_generate_video(request, result, error, progress)) {
        finish(500, "Internal Server Error", "{\"error\":{\"message\":\"" + json_escape(error) + "\",\"type\":\"server_error\"}}");
        status("VIDEO GENERATION FAILED");
        return;
    }
    statusf("VIDEO DONE %dX%d %d FRAMES IN %.1F S", result.width, result.height, result.frames, result.generate_s);
    char head[384];
    std::snprintf(head, sizeof(head),
                  "{\"model\":\"%s\",\"seed\":%lld,\"width\":%d,\"height\":%d,\"frames\":%d,\"steps\":%d,\"fps\":%d,"
                  "\"timings\":{\"load_s\":%.2f,\"generate_s\":%.2f},\"data\":[",
                  json_escape(model).c_str(), static_cast<long long>(result.seed), result.width, result.height,
                  static_cast<int>(result.png_frames.size()), result.steps, result.fps, load_s, result.generate_s);
    std::string json = head;
    for (std::size_t i = 0; i < result.png_frames.size(); ++i)
        json += (i ? ",{\"b64_png\":\"" : "{\"b64_png\":\"") + base64_encode(result.png_frames[i]) + "\"}";
    finish(200, "OK", json + "]}");
}

// POST /v1/vla/act: one control step of an OpenVLA policy.
// {"instruction": "put the bowl on the plate", "rgb_b64": base64 of a 224x224
//  RGB image (150528 raw bytes, already cropped as the policy expects),
//  "model": folder name, default "openvla-libero-spatial"}.
// The model is /app0/<model>.gguf (the Llama-2 backbone) plus
// /app0/<model>/{dinov2,siglip,proj}.gguf. Returns the seven action tokens and
// each action dimension in [-1, 1]; the client scales them for its robot.
void handle_vla_act(int fd, const std::string &body, InferenceState &state) {
    const double started = clock_seconds();
    std::string model = "openvla-libero-spatial", instruction, encoded, error;
    std::size_t at = 0;
    bool ok = property_string(body, "instruction", instruction) && !instruction.empty() && instruction.size() <= 512 &&
              property_string(body, "rgb_b64", encoded);
    if (ok && find_property(body, "model", 0, at)) ok = property_string(body, "model", model) && safe_model_path(model);
    bool decoded = false;
    const std::string rgb = ok ? base64_decode(encoded, decoded) : std::string();
    if (!ok || !decoded || rgb.size() != static_cast<std::size_t>(kVlaImageSize) * kVlaImageSize * 3) {
        send_response(fd, 422, "Unprocessable Entity", "application/json",
                      "{\"error\":{\"message\":\"instruction (up to 512 characters) and rgb_b64 (base64 of 224x224x3 raw RGB bytes) are required; optional model\",\"type\":\"invalid_request_error\"}}");
        status("BAD VLA REQUEST");
        return;
    }
    const std::string llm = model + ".gguf";
    const std::string folder = "/app0/" + model;
    struct stat info{};
    if (stat(("/app0/" + llm).c_str(), &info) != 0 || stat((folder + "/dinov2.gguf").c_str(), &info) != 0 ||
        stat((folder + "/siglip.gguf").c_str(), &info) != 0 || stat((folder + "/proj.gguf").c_str(), &info) != 0) {
        send_response(fd, 404, "Not Found", "application/json",
                      "{\"error\":{\"message\":\"VLA model not found: it needs <model>.gguf and <model>/dinov2.gguf, siglip.gguf, proj.gguf in the app folder\",\"type\":\"invalid_request_error\"}}");
        return;
    }
    auto fail = [&](const std::string &message) {
        send_response(fd, 500, "Internal Server Error", "application/json",
                      "{\"error\":{\"message\":\"" + json_escape(message) + "\",\"type\":\"server_error\"}}");
        statusf("VLA STEP FAILED: %s", message.c_str());
    };
    double load_s = 0.0;
    if (state.model_file != llm || !state.model) {
        free_gpu_for("vla", state, nullptr);
        stt_unload();
        tts_unload();
        statusf("VLA: LOADING LLM %s", llm.c_str());
        ModelOptions options = options_from_state(state);
        options.context_tokens = 1024;
        options.max_output_tokens = 64;
        options.gpu_layers = 999;
        options.cache_type_k = GGML_TYPE_F16;
        options.cache_type_v = GGML_TYPE_F16;
        options.flash_attention = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        int context_tokens = options.context_tokens, output_tokens = options.max_output_tokens;
        llama_model *loaded = nullptr;
        llama_context *context = nullptr;
        if (!create_model_context("LOAD", llm, options, &loaded, &context, &context_tokens, &output_tokens)) {
            fail("the VLA language model failed to load");
            return;
        }
        state.model = loaded;
        state.context = context;
        state.model_file = llm;
        options.context_tokens = context_tokens;
        options.max_output_tokens = output_tokens;
        apply_options(state, options);
    }
    if (!vla_vision || vla_vision_model != model) {
        vla_release();
        vision_release();
        stt_unload();
        tts_unload();
        if (!sd_service_loaded_model().empty()) sd_service_unload();
        statusf("VLA: LOADING IMAGE ENCODER %s", model.c_str());
        static ggml_backend_t gpu = ggml_backend_vk_init(0);
        static ggml_backend_t cpu = ggml_backend_cpu_init();
        if (!gpu || !cpu) { fail("no backend for the VLA image encoder"); return; }
        ggml_backend_cpu_set_n_threads(cpu, 4);
        vla_vision = vla_vision_load(folder + "/dinov2.gguf", folder + "/siglip.gguf", folder + "/proj.gguf", gpu, cpu, error);
        if (!vla_vision) { fail(error); return; }
        vla_vision_model = model;
    }
    load_s = clock_seconds() - started;
    VlaStep step;
    if (!vla_step(vla_vision, state.context, reinterpret_cast<const uint8_t *>(rgb.data()), instruction,
                  state.batch_tokens, step, error)) {
        fail(error);
        return;
    }
    char json[640];
    std::snprintf(json, sizeof(json),
                  "{\"model\":\"%s\",\"tokens\":[%d,%d,%d,%d,%d,%d,%d],"
                  "\"normalized\":[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f],\"prompt_tokens\":%d,"
                  "\"timings\":{\"load_s\":%.2f,\"vision_s\":%.3f,\"prompt_s\":%.3f,\"decode_s\":%.3f,\"total_s\":%.3f}}",
                  json_escape(model).c_str(), step.tokens[0], step.tokens[1], step.tokens[2], step.tokens[3],
                  step.tokens[4], step.tokens[5], step.tokens[6], step.normalized[0], step.normalized[1],
                  step.normalized[2], step.normalized[3], step.normalized[4], step.normalized[5], step.normalized[6],
                  step.prompt_tokens, load_s, step.vision_s, step.prompt_s, step.decode_s, clock_seconds() - started);
    send_response(fd, 200, "OK", "application/json", json);
    statusf("VLA STEP %.2F S (VISION %.2F PROMPT %.2F DECODE %.2F)", clock_seconds() - started - load_s, step.vision_s,
            step.prompt_s, step.decode_s);
}

// POST /v1/voice/chat: one spoken turn of a voice agent, always streamed (SSE).
// {"audio_b64": WAV of the user's speech, "messages": optional history,
//  "system", "language", "speaker_b64" (reference voice), "max_tokens",
//  "llm", "stt_model", "tts_model"}. Events: {"type":"transcript"},
// {"type":"text","delta"}, {"type":"audio","audio_b64",...} one per spoken
// sentence (so playback can start before the reply is finished), {"type":"done"}.
// Whisper, a small LLM and the TTS model stay loaded together between turns.
void handle_voice_chat(int fd, const std::string &body, InferenceState &state) {
    const double started = clock_seconds();
    std::string audio_b64, language, speaker_b64;
    std::string system_prompt = "You are a helpful voice assistant. Answer in one to three short spoken "
                                "sentences. Plain words only: no lists, markdown, emoji or code.";
    std::string llm = "Qwen3-4B-Q4_K_M.gguf", stt_model = "whisper/ggml-large-v3-turbo-q5_0.bin",
                tts_model = "qwen3-tts-1.7b";
    int max_tokens = 200;
    std::size_t at = 0;
    // The first check that fails names the reason, for the 422 body and the log.
    std::string error, voice_reference;
    auto require = [&](bool condition, const char *reason) {
        if (error.empty() && !condition) error = reason;
        return error.empty();
    };
    require(property_string(body, "audio_b64", audio_b64) && !audio_b64.empty(), "audio_b64 (base64 WAV) is required");
    require(set_optional_int(body, "max_tokens", 1, 1024, max_tokens), "max_tokens must be 1 to 1024");
    for (auto field : {std::make_pair("system", &system_prompt), std::make_pair("language", &language),
                       std::make_pair("llm", &llm), std::make_pair("stt_model", &stt_model),
                       std::make_pair("tts_model", &tts_model), std::make_pair("speaker_b64", &speaker_b64)})
        if (error.empty() && find_property(body, field.first, 0, at) &&
            !property_string(body, field.first, *field.second))
            error = std::string(field.first) + " must be a string";
    require(safe_model_path(stt_model) && safe_model_path(tts_model) && valid_model_filename(llm),
            "llm, stt_model or tts_model is not a valid model name");
    std::vector<std::string> roles, contents;
    if (error.empty() && find_property(body, "messages", 0, at))
        require(parse_messages(body, roles, contents, true),
                "messages must be an array of {role: system|user|assistant, content: string}, at most 64");
    std::vector<float> samples;
    if (error.empty()) {
        bool decoded = false;
        const std::string wav = base64_decode(audio_b64, decoded);
        if (require(decoded, "audio_b64 is not valid base64") && !decode_wav(wav, 16000, samples, error) && error.empty())
            error = "audio_b64 is not a readable WAV file";
        if (error.empty() && !speaker_b64.empty()) {
            voice_reference = base64_decode(speaker_b64, decoded);
            require(decoded, "speaker_b64 is not valid base64");
        }
    }
    if (!error.empty()) {
        send_response(fd, 422, "Unprocessable Entity", "application/json",
                      "{\"error\":{\"message\":\"" + json_escape(error) + "\",\"type\":\"invalid_request_error\"}}");
        statusf("BAD VOICE REQUEST: %s", error.c_str());
        return;
    }
    LoadStream stream;
    stream.fd = fd;
    stream.started = started;
    stream.open = begin_sse(fd);
    auto fail = [&](const std::string &message) {
        stream.event("{\"type\":\"error\",\"message\":\"" + json_escape(message) + "\"}");
        stream.event("[DONE]");
        if (stream.open) end_sse(fd);
        statusf("VOICE TURN FAILED: %s", message.c_str());
    };

    // The voice stack: unload image/video models and any big LLM, then load
    // whatever is missing.
    free_gpu_for("voice", state, &stream);
    if (!state.model) {
        statusf("VOICE: LOADING LLM %s", llm.c_str());
        stream.phase("loading", llm);
        ModelOptions options = options_from_state(state);
        options.context_tokens = 4096;
        options.max_output_tokens = 1024;
        options.gpu_layers = 999;
        options.cache_type_k = GGML_TYPE_Q4_0;
        options.cache_type_v = GGML_TYPE_Q4_0;
        options.flash_attention = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        int context_tokens = options.context_tokens, output_tokens = options.max_output_tokens;
        llama_model *model = nullptr;
        llama_context *context = nullptr;
        if (!create_model_context("LOAD", llm, options, &model, &context, &context_tokens, &output_tokens)) {
            fail("voice LLM failed to load");
            return;
        }
        state.model = model;
        state.context = context;
        state.model_file = llm;
        options.context_tokens = context_tokens;
        options.max_output_tokens = output_tokens;
        apply_options(state, options);
    }
    if (stt_loaded_model() != stt_model) stream.phase("loading", stt_model);
    if (!stt_load(stt_model, error, nullptr)) { fail(error); return; }
    if (tts_loaded_model() != tts_model) stream.phase("loading", tts_model);
    if (!tts_load(tts_model, error, nullptr)) { fail(error); return; }
    const double ready_s = clock_seconds() - started;

    // 1. Speech to text.
    SttResult heard;
    if (!stt_transcribe(samples, language, false, heard, error)) { fail(error); return; }
    const double transcript_s = clock_seconds() - started;
    {
        char tail[96];
        std::snprintf(tail, sizeof(tail), "\",\"audio_s\":%.2f,\"at_s\":%.2f}", heard.audio_s, transcript_s);
        stream.event("{\"type\":\"transcript\",\"text\":\"" + json_escape(heard.text) + tail);
    }
    statusf("VOICE HEARD %.1F S IN %.2F S", heard.audio_s, heard.transcribe_s);

    // 2. Prompt for the LLM: system, history, then what was just said.
    std::string prompt;
    try {
        auto templates = common_chat_templates_init(state.model, "");
        common_chat_templates_inputs inputs;
        inputs.add_generation_prompt = true;
        inputs.use_jinja = true;
        inputs.enable_thinking = false;
        auto add = [&](const std::string &role, const std::string &content) {
            common_chat_msg message;
            message.role = role;
            message.content = content;
            inputs.messages.push_back(std::move(message));
        };
        add("system", system_prompt);
        for (std::size_t i = 0; i < roles.size(); ++i) add(roles[i], contents[i]);
        add("user", heard.text);
        prompt = common_chat_templates_apply(templates.get(), inputs).prompt;
    } catch (const std::exception &exception) {
        fail(std::string("chat template failed: ") + exception.what());
        return;
    }
    const llama_vocab *vocab = llama_model_get_vocab(state.model);
    std::vector<llama_token> tokens(prompt.size() + 16);
    const int n_prompt = llama_tokenize(vocab, prompt.data(), static_cast<int32_t>(prompt.size()), tokens.data(),
                                        static_cast<int>(tokens.size()), true, true);
    if (n_prompt <= 0 || n_prompt + max_tokens > state.context_tokens) { fail("conversation is too long for the context"); return; }
    tokens.resize(static_cast<std::size_t>(n_prompt));
    llama_memory_clear(llama_get_memory(state.context), true);
    int rc = 0;
    for (int from = 0; from < n_prompt && rc == 0; from += state.batch_tokens)
        rc = llama_decode(state.context, llama_batch_get_one(tokens.data() + from, std::min(state.batch_tokens, n_prompt - from)));
    if (rc != 0) { fail("prompt evaluation failed"); return; }
    llama_sampler *sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // 3. Generate, and speak each sentence as soon as it is complete.
    std::string reply, pending;
    int spoken = 0, generated = 0;
    double first_text_s = 0.0, first_audio_s = 0.0, tts_s = 0.0, speech_s = 0.0;
    bool speak_failed = false;
    auto speak = [&](std::string sentence) {
        std::string clean;
        bool has_word = false;
        for (unsigned char c : sentence) {
            if (c == '*' || c == '#' || c == '`' || c == '_') continue;
            clean += (c == '\n' || c == '\r') ? ' ' : static_cast<char>(c);
            has_word |= std::isalnum(c) != 0;
        }
        if (!has_word) return;
        TtsRequest request;
        request.text = clean;
        request.language = language;
        request.speaker_wav = voice_reference;
        request.max_frames = 600;
        request.seed = 7;
        TtsResult audio;
        std::string tts_error;
        if (!tts_generate(request, audio, tts_error, nullptr)) {
            speak_failed = true;
            statusf("VOICE TTS FAILED: %s", tts_error.c_str());
            return;
        }
        // Without a reference voice, clone the first sentence for the rest so
        // the voice stays the same through the reply.
        if (voice_reference.empty()) voice_reference = audio.wav;
        tts_s += audio.generate_s;
        speech_s += audio.audio_s;
        const double now = clock_seconds() - started;
        if (first_audio_s == 0.0) first_audio_s = now;
        char head[160];
        std::snprintf(head, sizeof(head), "{\"type\":\"audio\",\"index\":%d,\"audio_s\":%.2f,\"tts_s\":%.2f,\"at_s\":%.2f,\"text\":\"",
                      spoken++, audio.audio_s, audio.generate_s, now);
        const std::vector<unsigned char> wav(audio.wav.begin(), audio.wav.end());
        stream.event(head + json_escape(clean) + "\",\"audio_b64\":\"" + base64_encode(wav) + "\"}");
    };
    for (; generated < max_tokens && stream.open; ++generated) {
        llama_token token = llama_sampler_sample(sampler, state.context, -1);
        if (llama_vocab_is_eog(vocab, token)) break;
        char piece[512];
        const int n = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, false);
        if (n > 0) {
            if (first_text_s == 0.0) first_text_s = clock_seconds() - started;
            const std::string text(piece, static_cast<std::size_t>(n));
            reply += text;
            pending += text;
            stream.event("{\"type\":\"text\",\"delta\":\"" + json_escape(text) + "\"}");
            // A sentence ends at . ! ? followed by whitespace (so "3.5" stays whole).
            std::size_t cut = std::string::npos;
            for (std::size_t i = 0; i + 1 < pending.size(); ++i)
                if ((pending[i] == '.' || pending[i] == '!' || pending[i] == '?') &&
                    std::isspace(static_cast<unsigned char>(pending[i + 1])) && i >= 12)
                    cut = i + 1;
            if (cut != std::string::npos) {
                speak(pending.substr(0, cut));
                pending.erase(0, cut);
            }
        }
        if (llama_decode(state.context, llama_batch_get_one(&token, 1)) != 0) break;
    }
    llama_sampler_free(sampler);
    speak(pending);
    const double total_s = clock_seconds() - started;
    char done[512];
    std::snprintf(done, sizeof(done),
                  "\",\"tokens\":%d,\"sentences\":%d,\"timings\":{\"ready_s\":%.2f,\"transcript_s\":%.2f,\"first_text_s\":%.2f,"
                  "\"first_audio_s\":%.2f,\"tts_s\":%.2f,\"speech_s\":%.2f,\"total_s\":%.2f}}",
                  generated, spoken, ready_s, transcript_s, first_text_s, first_audio_s, tts_s, speech_s, total_s);
    stream.event("{\"type\":\"done\",\"transcript\":\"" + json_escape(heard.text) + "\",\"reply\":\"" + json_escape(reply) + done);
    stream.event("[DONE]");
    if (stream.open) end_sse(fd);
    statusf("VOICE TURN: FIRST AUDIO %.2F S, %.1F S SPEECH, TOTAL %.2F S%s", first_audio_s, speech_s, total_s,
            speak_failed ? " (TTS ERRORS)" : "");
}

void handle_client(int fd, InferenceState &state) {
    std::string method, path, body;
    std::string authorization;
    int error = 400;
    if (!read_request(fd, method, path, body, authorization, error)) {
        if (error == 0) return;
        send_response(fd, error, "Request Error", "application/json", "{\"error\":{\"message\":\"invalid or oversized HTTP request\",\"type\":\"invalid_request_error\"}}");
        status("HTTP REQUEST REJECTED");
        return;
    }
    if (!valid_api_key(authorization)) {
        send_response(fd, 401, "Unauthorized", "application/json",
                      "{\"error\":{\"message\":\"valid Bearer API key required\",\"type\":\"authentication_error\"}}");
        status("HTTP AUTHENTICATION REJECTED");
        return;
    }
    if (method == "GET" && path == "/health") {
        const bool serving = (state.model && state.context) || !sd_service_loaded_model().empty() ||
                             !stt_loaded_model().empty() || !tts_loaded_model().empty();
        const std::string health = "{\"status\":\"" +
            std::string(serving ? "ok" : "unavailable") +
            "\",\"model_loaded\":" + (state.model && state.context ? "true" : "false") +
            ",\"model_file\":\"" + json_escape(state.model_file) +
            "\",\"image_model\":\"" + json_escape(sd_service_loaded_model()) +
            "\",\"stt_model\":\"" + json_escape(stt_loaded_model()) +
            "\",\"tts_model\":\"" + json_escape(tts_loaded_model()) +
            "\",\"parameters\":" + options_json(options_from_state(state)) + "}";
        send_response(fd, serving ? 200 : 503, serving ? "OK" : "Service Unavailable",
                      "application/json", health);
        status(state.model && state.context ? "HEALTH CHECK OK" : "HEALTH CHECK - NO MODEL");
        return;
    }
    if (method == "GET" && path == "/v1/models") {
        bool ok = false;
        const std::string response = model_list_json(state, ok);
        if (!ok) {
            send_response(fd, 500, "Internal Server Error", "application/json",
                          "{\"error\":{\"message\":\"cannot enumerate app model directory\",\"type\":\"server_error\"}}");
            status("MODEL DIRECTORY LIST FAILED");
            return;
        }
        send_response(fd, 200, "OK", "application/json", response);
        status("MODEL LIST REQUEST");
        return;
    }
    if (method == "GET" && path == "/v1/stats") {
        const std::string response = gpu_stats_json(state);
        send_response(fd, 200, "OK", "application/json", response);
        status("GPU STATS REQUEST");
        return;
    }
    if (method == "POST" && path == "/v1/models/load") {
        handle_model_load(fd, body, state);
        return;
    }
    if (method == "POST" && path == "/v1/images/generations") {
        handle_image_generation(fd, body, state);
        return;
    }
    if (method == "POST" && path == "/v1/audio/transcriptions") {
        handle_transcription(fd, body, state);
        return;
    }
    if (method == "POST" && path == "/v1/audio/speech") {
        handle_speech(fd, body, state);
        return;
    }
    if (method == "POST" && path == "/v1/videos/generations") {
        handle_video_generation(fd, body, state);
        return;
    }
    if (method == "POST" && path == "/v1/vla/act") {
        handle_vla_act(fd, body, state);
        return;
    }
    if (method == "POST" && path == "/v1/voice/chat") {
        handle_voice_chat(fd, body, state);
        return;
    }
    if (method != "POST" || path != "/v1/chat/completions") {
        const int code = (method == "GET" || method == "POST") ? 404 : 405;
        send_response(fd, code, code == 404 ? "Not Found" : "Method Not Allowed", "application/json",
                      "{\"error\":{\"message\":\"endpoint not found\",\"type\":\"invalid_request_error\"}}");
        return;
    }
    if (!state.model || !state.context) {
        send_response(fd, 503, "Service Unavailable", "application/json",
                      "{\"error\":{\"message\":\"no model is loaded\",\"type\":\"server_error\"}}");
        return;
    }
    std::vector<std::string> roles, contents, images;
    if (!parse_messages(body, roles, contents, false, &images)) {
        send_response(fd, 422, "Unprocessable Entity", "application/json",
                      "{\"error\":{\"message\":\"messages must contain role/content objects; content is text or an array of text and image_url (base64 data URL, up to 4 images) parts\",\"type\":\"invalid_request_error\"}}");
        status("BAD CHAT REQUEST");
        return;
    }
    int max_tokens = 128;
    const bool has_max_tokens = property_int(body, "max_tokens", max_tokens);
    const bool has_max_completion_tokens = !has_max_tokens && property_int(body, "max_completion_tokens", max_tokens);
    if ((has_max_tokens || has_max_completion_tokens) && (max_tokens < 1 || max_tokens > state.max_output_tokens)) {
        char message[192];
        std::snprintf(message, sizeof(message),
                      "{\"error\":{\"message\":\"max_tokens must be between 1 and %d\",\"type\":\"invalid_request_error\"}}",
                      state.max_output_tokens);
        send_response(fd, 422, "Unprocessable Entity", "application/json", message);
        return;
    }
    bool stream = false;
    std::size_t stream_at = 0;
    if (find_property(body, "stream", 0, stream_at) && !property_bool(body, "stream", stream)) {
        send_response(fd, 422, "Unprocessable Entity", "application/json",
                      "{\"error\":{\"message\":\"stream must be a boolean\",\"type\":\"invalid_request_error\"}}");
        return;
    }
    float temperature = state.default_temperature;
    int top_k = state.top_k;
    float top_p = state.top_p, min_p = state.min_p;
    int repeat_last_n = state.repeat_last_n;
    float repeat_penalty = state.repeat_penalty;
    float frequency_penalty = state.frequency_penalty;
    float presence_penalty = state.presence_penalty;
    int requested_seed = 0;
    std::size_t requested_seed_at = 0;
    const bool has_requested_seed = find_property(body, "seed", 0, requested_seed_at);
    bool enable_thinking = true;
    if (!set_optional_float(body, "temperature", 0.0f, 2.0f, temperature) ||
        !set_optional_int(body, "top_k", 0, 1000, top_k) ||
        !set_optional_float(body, "top_p", 0.01f, 1.0f, top_p) ||
        !set_optional_float(body, "min_p", 0.0f, 1.0f, min_p) ||
        !set_optional_int(body, "repeat_last_n", 0, 4096, repeat_last_n) ||
        !set_optional_float(body, "repeat_penalty", 0.5f, 2.0f, repeat_penalty) ||
        !set_optional_float(body, "frequency_penalty", -2.0f, 2.0f, frequency_penalty) ||
        !set_optional_float(body, "presence_penalty", -2.0f, 2.0f, presence_penalty) ||
        !set_optional_bool(body, "enable_thinking", enable_thinking) ||
        (has_requested_seed && !set_optional_int(body, "seed", -1, 0x7fffffff, requested_seed))) {
        send_response(fd, 422, "Unprocessable Entity", "application/json",
                      "{\"error\":{\"message\":\"invalid sampling parameters; see GET /health for defaults and supported ranges\",\"type\":\"invalid_request_error\"}}");
        return;
    }
    std::vector<llama_chat_message> messages;
    messages.reserve(roles.size());
    for (std::size_t i = 0; i < roles.size(); ++i)
        messages.push_back({roles[i].c_str(), contents[i].c_str()});
    // Use llama.cpp's common Jinja/template engine. The legacy public formatter
    // does not support all model templates (notably Gemma 4 and Muse), and its
    // -1 return for an unsupported template was previously mistaken for a
    // buffer-size result.
    constexpr std::size_t kMaxFormattedPrompt = 1024 * 1024;
    std::string formatted_prompt;
    std::string template_error;
    try {
        auto templates = common_chat_templates_init(state.model, "");
        if (!templates) throw std::runtime_error("could not initialize model chat template");
        common_chat_templates_inputs template_inputs;
        template_inputs.add_generation_prompt = true;
        template_inputs.use_jinja = true;
        template_inputs.enable_thinking = enable_thinking;
        template_inputs.messages.reserve(roles.size());
        for (std::size_t i = 0; i < roles.size(); ++i) {
            common_chat_msg message;
            message.role = roles[i];
            message.content = contents[i];
            template_inputs.messages.push_back(std::move(message));
        }
        formatted_prompt = common_chat_templates_apply(templates.get(), template_inputs).prompt;
    } catch (const std::exception &error) {
        template_error = error.what();
    }
    // Retain the legacy route as a compatibility fallback for formats it
    // handles. A negative legacy return means unsupported; never interpret it
    // as a buffer size.
    if (formatted_prompt.empty()) {
        const char *legacy_template = llama_model_chat_template(state.model, nullptr);
        if (legacy_template) {
            std::vector<char> legacy_buffer(kMaxFormattedPrompt + 1);
            const int legacy_len = llama_chat_apply_template(legacy_template,
                messages.data(), messages.size(), true, legacy_buffer.data(),
                static_cast<int>(legacy_buffer.size()));
            if (legacy_len >= 0 && static_cast<std::size_t>(legacy_len) < legacy_buffer.size())
                formatted_prompt.assign(legacy_buffer.data(), static_cast<std::size_t>(legacy_len));
        }
    }
    if (formatted_prompt.empty()) {
        const std::string message = "{\"error\":{\"message\":\"chat template formatting failed" +
            (template_error.empty() ? std::string() : ": " + json_escape(template_error)) +
            "\",\"type\":\"server_error\"}}";
        send_response(fd, 422, "Unprocessable Entity", "application/json", message);
        status("CHAT TEMPLATE FORMATTING FAILED");
        return;
    }
    if (formatted_prompt.empty() || formatted_prompt.size() > kMaxFormattedPrompt) {
        send_response(fd, 413, "Payload Too Large", "application/json",
                      "{\"error\":{\"message\":\"formatted prompt exceeds 1 MiB limit\",\"type\":\"invalid_request_error\"}}");
        status("FORMATTED PROMPT OVER 1 MIB LIMIT");
        return;
    }
    const llama_vocab *vocab = llama_model_get_vocab(state.model);
    std::vector<llama_token> tokens(formatted_prompt.size() + 16);
    int n_prompt = llama_tokenize(vocab, formatted_prompt.data(), static_cast<int32_t>(formatted_prompt.size()), tokens.data(),
                                  static_cast<int>(tokens.size()), true, true);
    if (n_prompt < 0) {
        tokens.resize(static_cast<std::size_t>(-n_prompt));
        n_prompt = llama_tokenize(vocab, formatted_prompt.data(), static_cast<int32_t>(formatted_prompt.size()), tokens.data(),
                                  static_cast<int>(tokens.size()), true, true);
    }
    // With images, the prompt is text and image chunks from the vision projector.
    mtmd_context *vision = nullptr;
    mtmd::input_chunks_ptr vision_chunks;
    if (!images.empty()) {
        std::string vision_error;
        vision = vision_for(state, vision_error);
        std::vector<mtmd::bitmap_ptr> bitmaps;
        std::vector<const mtmd_bitmap *> bitmap_pointers;
        for (std::size_t i = 0; vision && i < images.size(); ++i) {
            auto wrapper = mtmd_helper_bitmap_init_from_buf(
                vision, reinterpret_cast<const unsigned char *>(images[i].data()), images[i].size(), false,
                mtmd_helper_init_opt_default());
            if (!wrapper.bitmap) { vision_error = "an image could not be decoded (PNG or JPEG expected)"; vision = nullptr; break; }
            bitmaps.emplace_back(wrapper.bitmap);
            bitmap_pointers.push_back(wrapper.bitmap);
        }
        if (vision) {
            vision_chunks.reset(mtmd_input_chunks_init());
            mtmd_input_text text{};
            text.text = formatted_prompt.c_str();
            text.text_len = formatted_prompt.size();
            text.add_special = true;
            text.parse_special = true;
            if (mtmd_tokenize(vision, vision_chunks.get(), &text, bitmap_pointers.data(), bitmap_pointers.size()) != 0) {
                vision_error = "the images could not be prepared for the model";
                vision = nullptr;
            }
        }
        if (!vision) {
            send_response(fd, 422, "Unprocessable Entity", "application/json",
                          "{\"error\":{\"message\":\"" + json_escape(vision_error) + "\",\"type\":\"invalid_request_error\"}}");
            statusf("BAD CHAT REQUEST: %s", vision_error.c_str());
            return;
        }
        n_prompt = static_cast<int>(mtmd_helper_get_n_tokens(vision_chunks.get()));
    }
    if (n_prompt <= 0 || n_prompt + max_tokens > state.context_tokens) {
        char message[192];
        std::snprintf(message, sizeof(message),
                      "{\"error\":{\"message\":\"prompt plus max_tokens exceeds the %d-token context limit\",\"type\":\"context_length_exceeded\"}}",
                      state.context_tokens);
        send_response(fd, 422, "Unprocessable Entity", "application/json", message);
        status("PROMPT TOO LONG");
        return;
    }
    tokens.resize(static_cast<std::size_t>(n_prompt));
    status("REQUEST ACCEPTED - TOKENIZING");
    llama_memory_clear(llama_get_memory(state.context), true);
    const double prefill_start = clock_seconds();
    int rc = 0;
    if (vision) {
        llama_pos n_past = 0;
        rc = mtmd_helper_eval_chunks(vision, state.context, vision_chunks.get(), 0, 0, state.batch_tokens, true, &n_past);
    } else {
        for (int at = 0; at < n_prompt && rc == 0; at += state.batch_tokens) {
            const int count = std::min(state.batch_tokens, n_prompt - at);
            rc = llama_decode(state.context, llama_batch_get_one(tokens.data() + at, count));
        }
    }
    const double prefill_s = std::max(0.000001, clock_seconds() - prefill_start);
    if (rc != 0) {
        send_response(fd, 500, "Internal Server Error", "application/json",
                      "{\"error\":{\"message\":\"prompt evaluation failed\",\"type\":\"server_error\"}}");
        status("PROMPT EVAL FAILED");
        return;
    }
    statusf("PROMPT EVALUATED %d TOKENS IN %.0F MS - GENERATING", n_prompt, prefill_s * 1000.0);
    llama_sampler *sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (!sampler) {
        send_response(fd, 500, "Internal Server Error", "application/json",
                      "{\"error\":{\"message\":\"sampler allocation failed\",\"type\":\"server_error\"}}");
        status("SAMPLER CREATE FAILED");
        return;
    }
    if (repeat_penalty != 1.0f || frequency_penalty != 0.0f || presence_penalty != 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_penalties(
            llama_vocab_n_tokens(vocab), repeat_last_n, repeat_penalty,
            frequency_penalty, presence_penalty));
    }
    if (top_k > 0) llama_sampler_chain_add(sampler, llama_sampler_init_top_k(top_k));
    if (top_p < 1.0f) llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
    if (min_p > 0.0f) llama_sampler_chain_add(sampler, llama_sampler_init_min_p(min_p, 1));
    if (temperature > 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(
            has_requested_seed ? (requested_seed < 0 ? LLAMA_DEFAULT_SEED : static_cast<uint32_t>(requested_seed)) : state.seed));
    } else {
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    }
    if (stream) {
        const std::string first_chunk =
            "{\"id\":\"chatcmpl-ps5\",\"object\":\"chat.completion.chunk\","
            "\"created\":0,\"model\":\"ps5-local\",\"choices\":[{\"index\":0,"
            "\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}";
        if (!begin_sse(fd) || !send_sse_data(fd, first_chunk)) {
            llama_sampler_free(sampler);
            status("STREAM CLIENT DISCONNECTED");
            return;
        }
    }
    std::string answer;
    std::string pending_utf8;
    int generated = 0;
    const double generation_start = clock_seconds();
    bool stopped = false;
    bool stream_failed = false;
    // Where each token's time goes, for the STREAM DONE line: building and
    // submitting the graph, waiting for the GPU, reading the logits (only with
    // PS5_PROBE_LOGITS, as it adds a read), CPU sampling, and the SSE send.
    double sample_s = 0.0, send_s = 0.0, decode_s = 0.0, wait_s = 0.0, logits_s = 0.0;
    static const bool probe_logits = std::getenv("PS5_PROBE_LOGITS") != nullptr;
    static std::vector<float> logits_copy;
    const int n_vocab = llama_vocab_n_tokens(vocab);
    for (; rc == 0 && generated < max_tokens; ++generated) {
        if (probe_logits) {
            const double read_start = clock_seconds();
            const float *logits = llama_get_logits_ith(state.context, -1);
            logits_copy.assign(logits, logits + n_vocab);
            logits_s += clock_seconds() - read_start;
        }
        const double sample_start = clock_seconds();
        llama_token token = llama_sampler_sample(sampler, state.context, -1);
        const double send_start = clock_seconds();
        sample_s += send_start - sample_start;
        if (llama_vocab_is_eog(vocab, token)) { stopped = true; break; }
        char piece[512];
        const int n = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, false);
        if (n > 0) {
            const std::string token_piece(piece, static_cast<std::size_t>(n));
            if (!stream) {
                answer += token_piece;
            } else {
                pending_utf8 += token_piece;
                const std::size_t ready = complete_utf8_prefix(pending_utf8);
                if (ready) {
                    const std::string delta = pending_utf8.substr(0, ready);
                    if (!send_sse_data(fd, sse_chunk(delta, nullptr))) {
                        stream_failed = true;
                        break;
                    }
                    pending_utf8.erase(0, ready);
                }
            }
        }
        if (stream_failed) break;
        const double decode_start = clock_seconds();
        send_s += decode_start - send_start;
        rc = llama_decode(state.context, llama_batch_get_one(&token, 1));
        const double wait_start = clock_seconds();
        decode_s += wait_start - decode_start;
        llama_synchronize(state.context);
        wait_s += clock_seconds() - wait_start;
        if (progress_every() && (generated + 1) % progress_every() == 0) {
            char progress[128];
            std::snprintf(progress, sizeof(progress), "GENERATING %d TOKENS %.1F TOK/S",
                          generated + 1, (generated + 1) / std::max(0.000001, clock_seconds() - generation_start));
            status(progress);
        }
    }
    const double generation_s = std::max(0.000001, clock_seconds() - generation_start);
    llama_sampler_free(sampler);
    if (stream_failed) {
        status("STREAM CLIENT DISCONNECTED");
        return;
    }
    if (rc != 0) {
        if (stream) {
            send_sse_data(fd, "{\"error\":{\"message\":\"token generation failed\",\"type\":\"server_error\"}}");
            send_sse_data(fd, "[DONE]");
            end_sse(fd);
        } else {
            send_response(fd, 500, "Internal Server Error", "application/json",
                          "{\"error\":{\"message\":\"token generation failed\",\"type\":\"server_error\"}}");
        }
        status("GENERATION FAILED");
        return;
    }
    if (stream) {
        if (!pending_utf8.empty()) {
            const std::size_t ready = complete_utf8_prefix(pending_utf8);
            if (ready) {
                const std::string delta = pending_utf8.substr(0, ready);
                if (!send_sse_data(fd, sse_chunk(delta, nullptr))) {
                    status("STREAM CLIENT DISCONNECTED");
                    return;
                }
            }
            if (ready < pending_utf8.size()) {
                const std::string replacement = "\xef\xbf\xbd";
                if (!send_sse_data(fd, sse_chunk(replacement, nullptr))) {
                    status("STREAM CLIENT DISCONNECTED");
                    return;
                }
            }
        }
        if (!send_sse_data(fd, sse_chunk("", stopped ? "stop" : "length")) ||
            !send_sse_data(fd, "[DONE]") || !end_sse(fd)) {
            status("STREAM CLIENT DISCONNECTED");
            return;
        }
        const double per_token_ms = 1000.0 / std::max(1, generated);
        statusf("STREAM DONE %d TOKENS %.1F TOK/S DECODE %.2F WAIT %.2F LOGITS %.2F SAMPLE %.2F SEND %.2F MS/TOK",
                generated, generated / generation_s, decode_s * per_token_ms, wait_s * per_token_ms,
                logits_s * per_token_ms, sample_s * per_token_ms, send_s * per_token_ms);
        return;
    }
    char response[65536];
    const int response_len = std::snprintf(response, sizeof(response),
        "{\"id\":\"chatcmpl-ps5\",\"object\":\"chat.completion\",\"created\":0,"
        "\"model\":\"ps5-local\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},"
        "\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d},"
        "\"timings\":{\"prompt_tokens_per_second\":%.3f,\"completion_tokens_per_second\":%.3f}}",
        json_escape(answer).c_str(), stopped ? "stop" : "length", n_prompt, generated,
        n_prompt + generated, n_prompt / prefill_s, generated / generation_s);
    if (response_len <= 0 || static_cast<std::size_t>(response_len) >= sizeof(response)) {
        send_response(fd, 500, "Internal Server Error", "application/json", "{}");
        status("RESPONSE TOO LARGE");
        return;
    }
    send_response(fd, 200, "OK", "application/json", std::string(response, static_cast<std::size_t>(response_len)));
    char done[160];
    std::snprintf(done, sizeof(done), "DONE %d TOKENS %.1F TOK/S", generated,
                  generated / generation_s);
    status(done);
}

bool post_smoke_test(int server, unsigned short port, llama_model *model, llama_context *context,
                     int context_tokens, int max_output_tokens) {
    static const char body[] =
        "{\"model\":\"ps5-local\",\"messages\":[{\"role\":\"user\","
        "\"content\":\"Reply with the single word READY.\"}],\"max_tokens\":8,"
        "\"temperature\":0}";
    char request[512];
    const int request_len = std::snprintf(request, sizeof(request),
        "POST /v1/chat/completions HTTP/1.1\r\nHost: ps5\r\nAuthorization: Bearer %s\r\n"
        "Content-Type: application/json\r\nContent-Length: %zu\r\n"
        "Connection: close\r\n\r\n%s", kApiKey, sizeof(body) - 1, body);
    if (request_len <= 0 || static_cast<std::size_t>(request_len) >= sizeof(request)) return false;

    const int client = socket(AF_INET, SOCK_STREAM, 0);
    if (client < 0) return false;
    struct timeval timeout{120, 0};
    setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    struct sockaddr_in address{};
    address.sin_len = sizeof(address);
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(client, reinterpret_cast<struct sockaddr *>(&address), sizeof(address)) < 0 ||
        !send_all(client, request, static_cast<std::size_t>(request_len))) {
        close(client);
        return false;
    }

    const int accepted = accept(server, nullptr, nullptr);
    if (accepted < 0) { close(client); return false; }
    InferenceState state = make_inference_state(model, context, context_tokens,
        max_output_tokens, 4, 999, 0.0f, "startup-self-test.gguf");
    handle_client(accepted, state);
    shutdown(accepted, SHUT_RDWR);
    close(accepted);

    char response[8192];
    std::size_t received = 0;
    while (received + 1 < sizeof(response)) {
        const ssize_t n = recv(client, response + received, sizeof(response) - received - 1, 0);
        if (n <= 0) break;
        received += static_cast<std::size_t>(n);
    }
    close(client);
    response[received] = '\0';
    return std::strstr(response, "HTTP/1.1 200 OK") &&
           std::strstr(response, "\"object\":\"chat.completion\"");
}
} // namespace

void set_startup_kv_options(int cache_type_k, int cache_type_v, int flash_attention) {
    startup_cache_type_k = static_cast<ggml_type>(cache_type_k);
    startup_cache_type_v = static_cast<ggml_type>(cache_type_v);
    startup_flash_attention = flash_attention;
}

void run_http_server(llama_model *model, llama_context *context, unsigned short port,
                     int context_tokens, int max_output_tokens, const char *model_file,
                     int threads, int gpu_layers, float default_temperature) {
    // A client may close an SSE stream at any point; treat that as a failed
    // write instead of letting the process-wide SIGPIPE terminate inference.
    std::signal(SIGPIPE, SIG_IGN);
    // Sampling runs on this thread. At low temperature most of the vocabulary's
    // exp() results are denormal, which cost ~24 ms per token over a 262K
    // vocabulary; flush them to zero (FTZ|DAZ). Those probabilities are < 1e-38.
    _mm_setcsr(_mm_getcsr() | 0x8040);
    auto open_listener = [port]() -> int {
        const int listener = socket(AF_INET, SOCK_STREAM, 0);
        if (listener < 0) return -1;
        int reuse = 1;
        setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
        struct sockaddr_in address{};
        address.sin_len = sizeof(address);
        address.sin_family = AF_INET;
        address.sin_port = htons(port);
        address.sin_addr.s_addr = htonl(INADDR_ANY);
        if (bind(listener, reinterpret_cast<struct sockaddr *>(&address), sizeof(address)) < 0 ||
            listen(listener, 2) < 0) {
            close(listener);
            return -1;
        }
        return listener;
    };
    int server = open_listener();
    if (server < 0) {
        status("HTTP BIND/LISTEN FAILED");
        return;
    }
    char ready[128];
    std::snprintf(ready, sizeof(ready), "HTTP READY 0.0.0.0:%u TAILSCALE + API KEY", unsigned(port));
    InferenceState state = make_inference_state(model, context, context_tokens,
        max_output_tokens, threads, gpu_layers, default_temperature, model_file);
    status_screen::headline("TESTING CHAT POST");
    status("RUNNING LOCAL POST SMOKE TEST");
    const bool post_ok = post_smoke_test(server, port, state.model, state.context,
                                         state.context_tokens, state.max_output_tokens);
    status_screen::headline(ready);
    status(post_ok ? "POST SELF-TEST PASSED" : "POST SELF-TEST FAILED");
    status(ready);
    while (true) {
        struct sockaddr_in peer{};
        socklen_t peer_len = sizeof(peer);
        const int client = accept(server, reinterpret_cast<struct sockaddr *>(&peer), &peer_len);
        if (client < 0) {
            if (errno == EINTR) continue;
            // A network change (cable moved, interface down) can leave the
            // listener failing every accept; back off and reopen it instead of
            // spinning.
            statusf("HTTP ACCEPT FAILED ERRNO %d - REOPENING LISTENER", errno);
            close(server);
            server = -1;
            while (server < 0) {
                sleep(2);
                server = open_listener();
            }
            status("HTTP LISTENER REOPENED");
            continue;
        }
        if (!allowed_tailnet_peer(peer)) {
            status("REJECTED NON-TAILNET CLIENT");
            shutdown(client, SHUT_RDWR);
            close(client);
            continue;
        }
        struct timeval timeout{30, 0};
        setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
        handle_client(client, state);
        shutdown(client, SHUT_RDWR);
        close(client);
    }
}
