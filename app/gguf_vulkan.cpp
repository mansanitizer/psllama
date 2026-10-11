// GGUF inference on the PS5 GPU through llama.cpp's Vulkan backend on
// PS5_Vulkan's RADV. Runs the model offloaded to Vulkan, then (optionally) on
// the CPU with the same prompt and greedy sampling, and reports speeds and
// whether the tokens agree. Output: kernel log (klogsrv) and the TV.
// SPDX-License-Identifier: GPL-3.0-or-later
#include "llama.h"
#include "ggml-backend.h"
#include "http_server.h"
#include "status_screen.h"
#include <cstdarg>
#include <chrono>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <streambuf>
#include <string>
#include <vector>

extern "C" {
int sceKernelDebugOutText(int, const char *);
int ps5_klog_capture_stderr(const char *prefix);
int sceKernelUsleep(unsigned);
}

static constexpr char default_model_file[] =
    "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf";
// One submit must stay well under RADV/PS5's 10 s no-progress watchdog.
constexpr int batch_size = 128;
constexpr int default_context_tokens = 1024;
constexpr int default_max_output_tokens = 256;
constexpr int max_supported_context_tokens = 32768;
constexpr int max_supported_output_tokens = 4096;

struct RuntimeSettings {
    int context_tokens = default_context_tokens;
    int max_output_tokens = default_max_output_tokens;
    int threads = 4;
    int gpu_layers = 999;
    float temperature = 0.0f;
    ggml_type cache_type_k = GGML_TYPE_F16;
    ggml_type cache_type_v = GGML_TYPE_F16;
    int flash_attention = LLAMA_FLASH_ATTN_TYPE_AUTO;
    char model_file[128] = {};
};

static void status(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void status(const char *fmt, ...) {
    char text[512];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(text, sizeof(text), fmt, args);
    va_end(args);
    std::string logged = std::string("[gguf-vk] ") + text + "\n";
    sceKernelDebugOutText(0, logged.c_str());
    status_screen::line(text);
}

// llama.cpp's DEBUG lines (hundreds per context setup) would bury klog.
static void log_callback(ggml_log_level level, const char *text, void *) {
    if (level != GGML_LOG_LEVEL_DEBUG) sceKernelDebugOutText(0, text);
}

static bool parse_cache_type(const char *name, ggml_type &type) {
    if (std::strcmp(name, "f16") == 0) type = GGML_TYPE_F16;
    else if (std::strcmp(name, "q8_0") == 0) type = GGML_TYPE_Q8_0;
    else if (std::strcmp(name, "q4_0") == 0) type = GGML_TYPE_Q4_0;
    else return false;
    return true;
}

static char *trim(char *text) {
    while (*text && std::isspace(static_cast<unsigned char>(*text))) ++text;
    char *end = text + std::strlen(text);
    while (end > text && std::isspace(static_cast<unsigned char>(end[-1]))) --end;
    *end = '\0';
    return text;
}

static bool valid_model_filename(const char *name) {
    const size_t length = std::strlen(name);
    if (length < 6 || length >= 128 ||
        std::strcmp(name + length - 5, ".gguf") != 0)
        return false;
    for (const unsigned char *p = reinterpret_cast<const unsigned char *>(name); *p; ++p) {
        if (!std::isalnum(*p) && *p != '-' && *p != '_' && *p != '.') return false;
    }
    return true;
}

static RuntimeSettings load_runtime_settings() {
    RuntimeSettings settings;
    std::snprintf(settings.model_file, sizeof(settings.model_file), "%s", default_model_file);
    FILE *file = std::fopen("/app0/inference.conf", "r");
    if (!file) {
        status("SETTINGS FILE MISSING - USING SAFE DEFAULTS");
        return settings;
    }
    char line[256];
    while (std::fgets(line, sizeof(line), file)) {
        char *entry = trim(line);
        if (!*entry || *entry == '#') continue;
        char *equals = std::strchr(entry, '=');
        if (!equals) continue;
        *equals = '\0';
        char *key = trim(entry);
        char *value_text = trim(equals + 1);
        if (std::strcmp(key, "model_file") == 0) {
            if (valid_model_filename(value_text))
                std::snprintf(settings.model_file, sizeof(settings.model_file), "%s", value_text);
            else
                status("IGNORING INVALID MODEL FILE NAME");
            continue;
        }
        if (std::strcmp(key, "cache_type_k") == 0 || std::strcmp(key, "cache_type_v") == 0) {
            ggml_type &type = key[11] == 'k' ? settings.cache_type_k : settings.cache_type_v;
            if (!parse_cache_type(value_text, type)) status("IGNORING INVALID CACHE TYPE");
            continue;
        }
        if (std::strcmp(key, "temperature") == 0) {
            char *float_end = nullptr;
            errno = 0;
            const float temperature = std::strtof(value_text, &float_end);
            while (float_end && *float_end && std::isspace(static_cast<unsigned char>(*float_end))) ++float_end;
            if (!errno && float_end != value_text && float_end && !*float_end &&
                std::isfinite(temperature) && temperature >= 0.0f && temperature <= 2.0f)
                settings.temperature = temperature;
            else
                status("TEMPERATURE SETTING OUT OF RANGE");
            continue;
        }
        errno = 0;
        char *end = nullptr;
        const long value = std::strtol(value_text, &end, 10);
        while (end && *end && std::isspace(static_cast<unsigned char>(*end))) ++end;
        if (errno || !end || end == value_text || *end) {
            status("IGNORING INVALID SETTINGS VALUE");
            continue;
        }
        if (std::strcmp(key, "context_tokens") == 0) {
            if (value >= 512 && value <= max_supported_context_tokens)
                settings.context_tokens = static_cast<int>(value);
            else
                status("CONTEXT SETTING OUT OF RANGE");
        } else if (std::strcmp(key, "max_output_tokens") == 0) {
            if (value >= 1 && value <= max_supported_output_tokens)
                settings.max_output_tokens = static_cast<int>(value);
            else
                status("OUTPUT LIMIT SETTING OUT OF RANGE");
        } else if (std::strcmp(key, "threads") == 0) {
            if (value >= 1 && value <= 4) settings.threads = static_cast<int>(value);
            else status("THREAD SETTING OUT OF RANGE");
        } else if (std::strcmp(key, "flash_attention") == 0) {
            if (value >= -1 && value <= 1) settings.flash_attention = static_cast<int>(value);
            else status("FLASH ATTENTION SETTING OUT OF RANGE");
        } else if (std::strcmp(key, "gpu_layers") == 0) {
            if (value >= 0 && value <= 999) settings.gpu_layers = static_cast<int>(value);
            else status("GPU LAYERS SETTING OUT OF RANGE");
        }
    }
    std::fclose(file);
    if (settings.max_output_tokens >= settings.context_tokens) {
        settings.max_output_tokens = settings.context_tokens - 1;
        status("OUTPUT LIMIT REDUCED TO FIT CONTEXT");
    }
    return settings;
}

// Optional driver and backend switches for experiments, set before Vulkan
// starts (the launcher passes no environment): KEY=VALUE lines in
// /app0/vk-env.conf, limited to the ggml-vulkan, Mesa and title namespaces.
static void load_environment_overrides() {
    FILE *file = std::fopen("/app0/vk-env.conf", "r");
    if (!file) return;
    static const char *const allowed_prefixes[] = {"GGML_VK_", "RADV_", "MESA_", "ACO_", "PS5_"};
    char line[256];
    while (std::fgets(line, sizeof(line), file)) {
        char *entry = trim(line);
        if (!*entry || *entry == '#') continue;
        char *equals = std::strchr(entry, '=');
        if (!equals) continue;
        *equals = '\0';
        char *key = trim(entry);
        char *value = trim(equals + 1);
        bool allowed = false;
        for (const char *prefix : allowed_prefixes)
            allowed |= std::strncmp(key, prefix, std::strlen(prefix)) == 0;
        if (!allowed) {
            status("IGNORING ENV %s", key);
            continue;
        }
        setenv(key, value, 1);
        status("ENV %s=%s", key, value);
    }
    std::fclose(file);
}

// libc++'s std::cerr keeps the stderr stream it started with, which
// ps5_klog_capture_stderr does not redirect, so ggml-vulkan's perf logger (and
// anything else on std::cerr) was lost. Send it to klog line by line.
class KlogStreambuf : public std::streambuf {
    std::string line_;
    void emit() {
        if (line_.empty()) return;
        if (line_.back() != '\n') line_.push_back('\n');
        sceKernelDebugOutText(0, ("[llama] " + line_).c_str());
        line_.clear();
    }
    int overflow(int ch) override {
        if (ch == traits_type::eof()) return 0;
        line_.push_back(static_cast<char>(ch));
        if (ch == '\n') emit();
        return ch;
    }
    // std::cerr flushes after every <<; keep the partial line until its newline.
    int sync() override { return 0; }
};

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static bool load_progress(float progress, void *) {
    static int last = -1;
    static double started = now();
    const int percent = int(progress * 100.0f);
    if (percent / 10 != last / 10 || percent == 100) {
        last = percent;
        status("LOADING MODEL %d%% T=%.0fS", percent, now() - started);
    }
    return true;
}

int main() {
    ps5_klog_capture_stderr("[llama] ");
    static KlogStreambuf klog_streambuf;
    std::cerr.rdbuf(&klog_streambuf);
    if (!status_screen::init("PS5 LOCAL INFERENCE SERVER"))
        sceKernelDebugOutText(0, "[status-screen] initialization failed\n");
    RuntimeSettings settings = load_runtime_settings();
    status("LOAD SETTINGS CTX=%d MAX_OUT=%d", settings.context_tokens, settings.max_output_tokens);
    status("LOAD SETTINGS GPU=%d THREADS=%d TEMP=%.2f", settings.gpu_layers,
           settings.threads, settings.temperature);
    status("MODEL FILE %s", settings.model_file);
    load_environment_overrides();
    llama_log_set(log_callback, nullptr);
    llama_backend_init();

    status("BACKEND DEVICES: %zu", ggml_backend_dev_count());
    bool have_gpu = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        size_t free_bytes = 0, total_bytes = 0;
        ggml_backend_dev_memory(device, &free_bytes, &total_bytes);
        const auto device_type = ggml_backend_dev_type(device);
        have_gpu |= device_type == GGML_BACKEND_DEVICE_TYPE_GPU ||
                    device_type == GGML_BACKEND_DEVICE_TYPE_IGPU;
        status("  %s - %s (%zu MIB FREE OF %zu)", ggml_backend_dev_name(device),
               ggml_backend_dev_description(device), free_bytes >> 20, total_bytes >> 20);
    }
    if (!have_gpu) status("NO VULKAN GPU DEVICE - CHECK KLOG");

    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = settings.gpu_layers;
    model_params.load_mode = LLAMA_LOAD_MODE_AUTO;
    model_params.lazy_mode = LLAMA_LAZY_MODE_AUTO;
    model_params.use_extra_bufts = false;
    model_params.check_tensors = false;
    model_params.progress_callback = load_progress;
    status("LOADING MODEL");
    char model_path[160];
    std::snprintf(model_path, sizeof(model_path), "/app0/%s", settings.model_file);
    llama_model *model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        status("MODEL LOAD FAILED");
        for (;;) sceKernelUsleep(1000000);
    }
    const uint32_t model_context = llama_model_n_ctx_train(model);
    if (model_context && settings.context_tokens > static_cast<int>(model_context)) {
        settings.context_tokens = static_cast<int>(model_context);
        if (settings.max_output_tokens >= settings.context_tokens)
            settings.max_output_tokens = settings.context_tokens - 1;
        status("CONTEXT CLAMPED TO MODEL LIMIT %u", model_context);
    }
    auto context_params = llama_context_default_params();
    context_params.n_ctx = static_cast<uint32_t>(settings.context_tokens);
    context_params.n_batch = batch_size;
    context_params.n_ubatch = batch_size;
    context_params.n_threads = settings.threads;
    context_params.n_threads_batch = settings.threads;
    context_params.offload_kqv = true;
    context_params.op_offload = true;
    context_params.flash_attn_type = static_cast<llama_flash_attn_type>(settings.flash_attention);
    context_params.type_k = settings.cache_type_k;
    context_params.type_v = settings.cache_type_v;
    context_params.rope_freq_base = 0.0f;
    context_params.rope_freq_scale = 0.0f;
    context_params.no_perf = false;
    llama_context *context = llama_init_from_model(model, context_params);
    if (!context) {
        status("CONTEXT CREATE FAILED");
        llama_model_free(model);
        for (;;) sceKernelUsleep(1000000);
    }
    static const char *const flash_attention_names[] = {"auto", "off", "on"};
    status("KV CACHE STARTUP %s K=%s V=%s CTX=%d FA=%s", settings.model_file,
           ggml_type_name(settings.cache_type_k), ggml_type_name(settings.cache_type_v),
           settings.context_tokens, flash_attention_names[settings.flash_attention + 1]);
    char model_desc[160] = {};
    llama_model_desc(model, model_desc, sizeof(model_desc));
    status("MODEL LOADED");
    status("%s", model_desc);
    set_startup_kv_options(settings.cache_type_k, settings.cache_type_v, settings.flash_attention);
    run_http_server(model, context, 8081, settings.context_tokens, settings.max_output_tokens,
                    settings.model_file, settings.threads, settings.gpu_layers,
                    settings.temperature);
    // Returning from main or calling exit crashes this launch context.
    for (;;) sceKernelUsleep(1000000);
}
