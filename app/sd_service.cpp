// Text-to-image with stable-diffusion.cpp on the PS5's Vulkan device.
// SPDX-License-Identifier: GPL-3.0-or-later
#include "sd_service.h"
#include "stable-diffusion.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#include "stb_image.h"

extern "C" int sceKernelDebugOutText(int, const char *);

namespace {
sd_ctx_t *context = nullptr;
std::string loaded_model;
SdModelDefaults defaults;
const SdProgress *active_progress = nullptr;
const char *active_phase = "loading";

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void on_log(sd_log_level_t level, const char *text, void *) {
    if (level < SD_LOG_INFO || !text) return;
    // stable-diffusion.cpp reports weight loading, sampling and VAE decoding
    // through one progress callback; its log lines say which stage follows.
    if (active_progress) {
        if (std::strstr(text, "generating image") || std::strstr(text, "generating video") ||
            std::strstr(text, "sampling using"))
            active_phase = "generating";
        else if (std::strstr(text, "decoding")) active_phase = "decoding";
    }
    char line[600];
    std::snprintf(line, sizeof(line), "[sd] %s%s", text,
                  text[0] && text[std::strlen(text) - 1] == '\n' ? "" : "\n");
    sceKernelDebugOutText(0, line);
}

void on_progress(int step, int steps, float, void *) {
    if (active_progress && *active_progress) (*active_progress)(active_phase, step, steps);
}

void png_write(void *context_out, void *data, int size) {
    auto *out = static_cast<std::vector<unsigned char> *>(context_out);
    const auto *bytes = static_cast<const unsigned char *>(data);
    out->insert(out->end(), bytes, bytes + size);
}
} // namespace

const std::string &sd_service_loaded_model() { return loaded_model; }
const SdModelDefaults &sd_service_defaults() { return defaults; }

namespace {
std::map<std::string, std::string> read_model_conf(const std::string &folder) {
    std::map<std::string, std::string> conf;
    FILE *file = std::fopen((folder + "/sd-model.conf").c_str(), "r");
    if (!file) return conf;
    char line[512];
    while (std::fgets(line, sizeof(line), file)) {
        std::string entry(line);
        while (!entry.empty() && (entry.back() == '\n' || entry.back() == '\r' || entry.back() == ' ')) entry.pop_back();
        const std::size_t equals = entry.find('=');
        if (entry.empty() || entry[0] == '#' || equals == std::string::npos) continue;
        conf[entry.substr(0, equals)] = entry.substr(equals + 1);
    }
    std::fclose(file);
    return conf;
}
} // namespace

void sd_service_unload() {
    if (context) free_sd_ctx(context);
    context = nullptr;
    loaded_model.clear();
}

bool sd_service_load(const std::string &model, const std::string &prediction, std::string &error,
                     const SdProgress &progress, double *load_s) {
    if (context && loaded_model == model) return true;
    sd_service_unload();
    sd_set_log_callback(on_log, nullptr);
    sd_set_progress_callback(on_progress, nullptr);
    const std::string path = "/app0/" + model;
    auto conf = read_model_conf(path);
    // Component paths must outlive new_sd_ctx; keep them in the map.
    for (const char *key : {"model", "diffusion_model", "llm", "vae", "clip_l", "clip_g", "clip_vision", "t5xxl"})
        if (conf.count(key)) conf[key] = path + "/" + conf[key];
    sd_ctx_params_t params;
    sd_ctx_params_init(&params);
    if (conf.empty()) {
        params.model_path = path.c_str();
    } else {
        auto component = [&](const char *key) { return conf.count(key) ? conf[key].c_str() : nullptr; };
        params.model_path = component("model");
        params.diffusion_model_path = component("diffusion_model");
        params.llm_path = component("llm");
        params.vae_path = component("vae");
        params.clip_l_path = component("clip_l");
        params.clip_g_path = component("clip_g");
        params.clip_vision_path = component("clip_vision");
        params.t5xxl_path = component("t5xxl");
        params.diffusion_flash_attn = conf["diffusion_flash_attn"] == "1";
    }
    std::string model_prediction = prediction.empty() ? conf["prediction"] : prediction;
    SdModelDefaults model_defaults;
    if (conf.count("width")) model_defaults.width = std::atoi(conf["width"].c_str());
    if (conf.count("height")) model_defaults.height = std::atoi(conf["height"].c_str());
    if (conf.count("steps")) model_defaults.steps = std::atoi(conf["steps"].c_str());
    if (conf.count("cfg_scale")) model_defaults.cfg_scale = static_cast<float>(std::atof(conf["cfg_scale"].c_str()));
    if (conf.count("sample_method")) model_defaults.sample_method = conf["sample_method"];
    model_defaults.video = conf["mode"] == "video";
    if (conf.count("frames")) model_defaults.frames = std::atoi(conf["frames"].c_str());
    if (conf.count("fps")) model_defaults.fps = std::atoi(conf["fps"].c_str());
    if (conf.count("flow_shift")) model_defaults.flow_shift = static_cast<float>(std::atof(conf["flow_shift"].c_str()));
    params.n_threads = 4;
    params.enable_mmap = false;
    params.backend = "Vulkan0";
    // Where weights live: on the GPU by default; sd-model.conf can move modules
    // to disk (loaded for their stage, then released), e.g.
    // params_backend=all=Vulkan0,te=disk,vae=disk for models larger than the heap.
    if (!conf.count("params_backend")) conf["params_backend"] = "Vulkan0";
    params.params_backend = conf["params_backend"].c_str();
    if (conf.count("max_vram")) params.max_vram = conf["max_vram"].c_str();
    if (!model_prediction.empty()) {
        params.prediction = str_to_prediction(model_prediction.c_str());
        if (params.prediction == PREDICTION_COUNT) {
            error = "unknown prediction type";
            return false;
        }
    }
    const double started = now_s();
    active_progress = &progress;
    active_phase = "loading";
    context = new_sd_ctx(&params);
    active_progress = nullptr;
    if (load_s) *load_s = now_s() - started;
    if (!context || !(sd_ctx_supports_image_generation(context) || sd_ctx_supports_video_generation(context))) {
        sd_service_unload();
        error = "image model failed to load (see klog)";
        return false;
    }
    loaded_model = model;
    defaults = model_defaults;
    return true;
}

bool sd_service_generate(const SdImageRequest &request, SdImageResult &result, std::string &error,
                         const SdProgress &progress) {
    if (!context) {
        error = "no image model is loaded";
        return false;
    }
    SdImageRequest filled = request;
    if (filled.width <= 0) filled.width = defaults.width;
    if (filled.height <= 0) filled.height = defaults.height;
    if (filled.steps <= 0) filled.steps = defaults.steps;
    if (filled.cfg_scale < 0.0f) filled.cfg_scale = defaults.cfg_scale;
    if (filled.sample_method.empty()) filled.sample_method = defaults.sample_method;
    const SdImageRequest &request_used = filled;
    sd_img_gen_params_t params;
    sd_img_gen_params_init(&params);
    params.prompt = request_used.prompt.c_str();
    params.negative_prompt = request_used.negative_prompt.c_str();
    params.width = request_used.width;
    params.height = request_used.height;
    params.sample_params.sample_steps = request_used.steps;
    params.sample_params.guidance.txt_cfg = request_used.cfg_scale;
    if (!request_used.sample_method.empty()) {
        params.sample_params.sample_method = str_to_sample_method(request_used.sample_method.c_str());
        if (params.sample_params.sample_method == SAMPLE_METHOD_COUNT) {
            error = "unknown sample_method";
            return false;
        }
    }
    // Decoded start and reference images; freed when this function returns.
    struct Pixels {
        std::vector<unsigned char *> owned;
        ~Pixels() { for (unsigned char *p : owned) stbi_image_free(p); }
        bool decode(const std::string &bytes, sd_image_t &image) {
            int width = 0, height = 0, channels = 0;
            unsigned char *pixels = stbi_load_from_memory(reinterpret_cast<const unsigned char *>(bytes.data()),
                                                          static_cast<int>(bytes.size()), &width, &height, &channels, 3);
            if (!pixels) return false;
            owned.push_back(pixels);
            image = sd_image_t{static_cast<uint32_t>(width), static_cast<uint32_t>(height), 3, pixels};
            return true;
        }
    } pixels;
    std::vector<sd_image_t> references(request_used.ref_images.size());
    if (!request_used.init_image.empty()) {
        if (!pixels.decode(request_used.init_image, params.init_image)) {
            error = "init image could not be decoded (PNG or JPEG expected)";
            return false;
        }
        params.strength = request_used.strength >= 0.0f ? request_used.strength : 0.75f;
    }
    for (std::size_t i = 0; i < references.size(); ++i) {
        if (!pixels.decode(request_used.ref_images[i], references[i])) {
            error = "a reference image could not be decoded (PNG or JPEG expected)";
            return false;
        }
    }
    if (!references.empty()) {
        params.ref_images = references.data();
        params.ref_images_count = static_cast<int>(references.size());
    }
    result.width = request_used.width;
    result.height = request_used.height;
    result.steps = request_used.steps;
    result.cfg_scale = request_used.cfg_scale;
    result.seed = request_used.seed >= 0 ? request_used.seed : static_cast<int64_t>(std::random_device{}() & 0x7fffffff);
    params.seed = result.seed;
    params.batch_count = 1;
    // Large VAE decodes as one dispatch could approach the 10 s GPU watchdog.
    params.vae_tiling_params.enabled = request_used.width * request_used.height > 768 * 768;

    const double started = now_s();
    active_progress = &progress;
    active_phase = "preparing";  // first-use weight loading and text encoding
    sd_image_t *images = nullptr;
    int count = 0;
    const bool ok = generate_image(context, &params, &images, &count);
    active_progress = nullptr;
    result.generate_s = now_s() - started;
    if (!ok || count < 1 || !images || !images[0].data) {
        free_sd_images(images, count);
        error = "image generation failed (see klog)";
        return false;
    }
    const sd_image_t &image = images[0];
    result.png.clear();
    stbi_write_png_to_func(png_write, &result.png, static_cast<int>(image.width), static_cast<int>(image.height),
                           static_cast<int>(image.channel), image.data,
                           static_cast<int>(image.width * image.channel));
    free_sd_images(images, count);
    if (result.png.empty()) {
        error = "PNG encoding failed";
        return false;
    }
    return true;
}

bool sd_service_generate_video(const SdVideoRequest &request, SdVideoResult &result, std::string &error,
                               const SdProgress &progress) {
    if (!context || !defaults.video) {
        error = context ? "the loaded model is not a video model" : "no video model is loaded";
        return false;
    }
    sd_vid_gen_params_t params;
    sd_vid_gen_params_init(&params);
    params.prompt = request.prompt.c_str();
    params.negative_prompt = request.negative_prompt.c_str();
    int image_width = 0, image_height = 0, image_channels = 0;
    unsigned char *image_pixels = nullptr;
    if (!request.init_image.empty()) {
        image_pixels = stbi_load_from_memory(reinterpret_cast<const unsigned char *>(request.init_image.data()),
                                             static_cast<int>(request.init_image.size()), &image_width,
                                             &image_height, &image_channels, 3);
        if (!image_pixels) {
            error = "init image could not be decoded (PNG or JPEG expected)";
            return false;
        }
        params.init_image = sd_image_t{static_cast<uint32_t>(image_width), static_cast<uint32_t>(image_height), 3,
                                       image_pixels};
    }
    // With a start image and no explicit size, follow the image (multiples of 16).
    const bool size_from_image = image_pixels && request.width <= 0;
    result.width = params.width = size_from_image ? image_width / 16 * 16
                                                  : request.width > 0 ? request.width : defaults.width;
    result.height = params.height = size_from_image ? image_height / 16 * 16
                                                    : request.height > 0 ? request.height : defaults.height;
    result.frames = params.video_frames = request.frames > 0 ? request.frames : defaults.frames;
    result.steps = params.sample_params.sample_steps = request.steps > 0 ? request.steps : defaults.steps;
    params.sample_params.guidance.txt_cfg = request.cfg_scale >= 0.0f ? request.cfg_scale : defaults.cfg_scale;
    if (!defaults.sample_method.empty())
        params.sample_params.sample_method = str_to_sample_method(defaults.sample_method.c_str());
    if (defaults.flow_shift > 0.0f) params.sample_params.flow_shift = defaults.flow_shift;
    params.fps = defaults.fps;
    result.seed = request.seed >= 0 ? request.seed : static_cast<int64_t>(std::random_device{}() & 0x7fffffff);
    params.seed = result.seed;
    // The Wan VAE decodes all frames at once; tile it to bound memory and
    // the length of any single GPU submission.
    params.vae_tiling_params.enabled = true;

    const double started = now_s();
    active_progress = &progress;
    active_phase = "preparing";
    sd_image_t *frames = nullptr;
    int count = 0;
    int fps = 0;
    const bool ok = generate_video(context, &params, &frames, &count, nullptr, &fps);
    active_progress = nullptr;
    stbi_image_free(image_pixels);
    result.generate_s = now_s() - started;
    if (!ok || count < 1 || !frames) {
        free_sd_images(frames, count);
        error = "video generation failed (see klog)";
        return false;
    }
    result.fps = fps > 0 ? fps : defaults.fps;
    result.png_frames.clear();
    for (int i = 0; i < count; ++i) {
        std::vector<unsigned char> png;
        stbi_write_png_to_func(png_write, &png, static_cast<int>(frames[i].width), static_cast<int>(frames[i].height),
                               static_cast<int>(frames[i].channel), frames[i].data,
                               static_cast<int>(frames[i].width * frames[i].channel));
        result.png_frames.push_back(std::move(png));
    }
    free_sd_images(frames, count);
    return true;
}

std::string base64_encode(const std::vector<unsigned char> &data) {
    static const char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((data.size() + 2) / 3 * 4);
    std::size_t i = 0;
    for (; i + 2 < data.size(); i += 3) {
        const unsigned v = data[i] << 16 | data[i + 1] << 8 | data[i + 2];
        out += table[v >> 18]; out += table[v >> 12 & 63]; out += table[v >> 6 & 63]; out += table[v & 63];
    }
    if (i < data.size()) {
        const unsigned v = data[i] << 16 | (i + 1 < data.size() ? data[i + 1] << 8 : 0);
        out += table[v >> 18]; out += table[v >> 12 & 63];
        out += i + 1 < data.size() ? table[v >> 6 & 63] : '=';
        out += '=';
    }
    return out;
}
