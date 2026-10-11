// Text-to-image for the PS5 title: stable-diffusion.cpp on the same ggml and
// Vulkan device as llama.cpp. One image model at a time; the caller unloads
// the LLM first (they do not both fit in the GPU heap).
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// Zero/negative fields mean "the model's default" (sd_service_defaults).
struct SdImageRequest {
    std::string prompt;
    std::string negative_prompt;
    int width = 0;
    int height = 0;
    int steps = 0;
    float cfg_scale = -1.0f;
    int64_t seed = -1;               // -1: random
    std::string sample_method;       // empty: the model's default
    // Editing (PNG/JPEG bytes). init_image: image-to-image, redrawn by
    // `strength` (0 keeps it, 1 ignores it). ref_images: reference-image
    // editing for models that support it (FLUX.2, Qwen-Image-Edit, Kontext).
    std::string init_image;
    float strength = -1.0f;          // negative: 0.75
    std::vector<std::string> ref_images;
};

// Per-model defaults: from the model folder's sd-model.conf, else SD-Turbo's.
struct SdModelDefaults {
    int width = 512;
    int height = 512;
    int steps = 1;
    float cfg_scale = 1.0f;
    std::string sample_method;
    bool video = false;       // sd-model.conf mode=video (e.g. Wan)
    int frames = 17;          // video: 4k+1 frames
    int fps = 16;
    float flow_shift = 0.0f;  // 0: the model's default
};

struct SdVideoRequest {
    std::string prompt;
    std::string negative_prompt;
    int width = 0, height = 0, frames = 0, steps = 0;
    float cfg_scale = -1.0f;
    int64_t seed = -1;
    std::string init_image;   // optional first frame (PNG/JPEG bytes) for image-to-video models
};

struct SdVideoResult {
    std::vector<std::vector<unsigned char>> png_frames;
    int64_t seed = 0;
    int width = 0, height = 0, frames = 0, steps = 0, fps = 0;
    double generate_s = 0.0;
};

struct SdImageResult {
    std::vector<unsigned char> png;
    int64_t seed = 0;
    double load_s = 0.0;             // model load, when this request loaded it
    double generate_s = 0.0;
    int width = 0, height = 0, steps = 0;  // as generated, after model defaults
    float cfg_scale = 0.0f;
};

// Progress: phase is "loading" (model load), then per request "preparing"
// (first-use weight loading), "generating" (sampling step of steps), "decoding" (VAE).
using SdProgress = std::function<void(const char *phase, int value, int total)>;

// Loads a model (a folder or file under /app0) unless it is already loaded. A
// folder may hold sd-model.conf (KEY=VALUE): component files diffusion_model,
// llm, vae, clip_l, clip_g, clip_vision, t5xxl, model (relative to the folder), options
// diffusion_flash_attn=1, prediction, params_backend (e.g.
// all=Vulkan0,te=disk,vae=disk), max_vram, and defaults width, height, steps,
// cfg_scale, sample_method. Without it the folder is loaded as one model
// (diffusers layout, e.g. SD-Turbo).
// prediction: "" (auto), "eps", "v", ... as stable-diffusion.cpp names them.
bool sd_service_load(const std::string &model, const std::string &prediction, std::string &error,
                     const SdProgress &progress, double *load_s);
void sd_service_unload();
const SdModelDefaults &sd_service_defaults();
const std::string &sd_service_loaded_model();
bool sd_service_generate(const SdImageRequest &request, SdImageResult &result, std::string &error,
                         const SdProgress &progress);
bool sd_service_generate_video(const SdVideoRequest &request, SdVideoResult &result, std::string &error,
                               const SdProgress &progress);
std::string base64_encode(const std::vector<unsigned char> &data);
