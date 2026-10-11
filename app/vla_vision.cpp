// OpenVLA image encoder (see vla_vision.h). The transformer blocks follow
// llama.cpp's tools/mtmd/clip.cpp so that only operations already proven on
// the PS5's Vulkan backend are used.
// SPDX-License-Identifier: GPL-3.0-or-later
#include "vla_vision.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "gguf.h"
#include <cmath>
#include <cstdio>
#include <map>

namespace {

struct Weights {
    ggml_context *ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::map<std::string, ggml_tensor *> tensors;
    int n_head = 16;
    int feature_layer = -1;            // index of the last block whose output is used
    float eps = 1e-6f;
    float mean[3] = {0.5f, 0.5f, 0.5f};
    float std[3] = {0.5f, 0.5f, 0.5f};

    ggml_tensor *get(const std::string &name) const {
        const auto it = tensors.find(name);
        return it == tensors.end() ? nullptr : it->second;
    }
    void free() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
        buffer = nullptr;
        ctx = nullptr;
    }
};

bool read_floats(const gguf_context *gguf, const char *key, float *out, size_t n) {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(gguf, id) != GGUF_TYPE_FLOAT32 ||
        gguf_get_arr_n(gguf, id) != n)
        return false;
    const float *data = static_cast<const float *>(gguf_get_arr_data(gguf, id));
    for (size_t i = 0; i < n; ++i) out[i] = data[i];
    return true;
}

// Loads every tensor of a GGUF file into a buffer of `backend`.
bool load_weights(const std::string &path, ggml_backend_t backend, bool vision_tower, Weights &w, std::string &error) {
    ggml_context *meta = nullptr;
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = &meta;
    gguf_context *gguf = gguf_init_from_file(path.c_str(), params);
    if (!gguf) {
        error = "could not read " + path;
        return false;
    }
    const int64_t n = gguf_get_n_tensors(gguf);
    ggml_init_params init{};
    init.mem_size = static_cast<size_t>(n + 1) * ggml_tensor_overhead();
    init.no_alloc = true;
    w.ctx = ggml_init(init);
    for (int64_t i = 0; i < n; ++i) {
        const char *name = gguf_get_tensor_name(gguf, i);
        ggml_tensor *copy = ggml_dup_tensor(w.ctx, ggml_get_tensor(meta, name));
        ggml_set_name(copy, name);
        w.tensors[name] = copy;
    }
    w.buffer = ggml_backend_alloc_ctx_tensors(w.ctx, backend);
    bool ok = w.buffer != nullptr;
    if (!ok) error = "not enough memory for " + path;
    FILE *file = ok ? std::fopen(path.c_str(), "rb") : nullptr;
    if (ok && !file) { ok = false; error = "could not open " + path; }
    std::vector<char> data;
    for (int64_t i = 0; ok && i < n; ++i) {
        ggml_tensor *tensor = w.tensors[gguf_get_tensor_name(gguf, i)];
        const size_t size = ggml_nbytes(tensor);
        data.resize(size);
        const long offset = static_cast<long>(gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, i));
        if (std::fseek(file, offset, SEEK_SET) != 0 || std::fread(data.data(), 1, size, file) != size) {
            ok = false;
            error = "short read in " + path;
            break;
        }
        ggml_backend_tensor_set(tensor, data.data(), 0, size);
    }
    if (file) std::fclose(file);
    if (ok && vision_tower) {
        int64_t id = gguf_find_key(gguf, "clip.vision.attention.head_count");
        if (id >= 0) w.n_head = static_cast<int>(gguf_get_val_u32(gguf, id));
        id = gguf_find_key(gguf, "clip.vision.attention.layer_norm_epsilon");
        if (id >= 0) w.eps = gguf_get_val_f32(gguf, id);
        id = gguf_find_key(gguf, "clip.vision.feature_layer");
        if (id >= 0 && gguf_get_kv_type(gguf, id) == GGUF_TYPE_ARRAY && gguf_get_arr_n(gguf, id) == 1) {
            const gguf_type type = gguf_get_arr_type(gguf, id);
            const void *value = gguf_get_arr_data(gguf, id);
            if (type == GGUF_TYPE_INT32) w.feature_layer = *static_cast<const int32_t *>(value);
            else if (type == GGUF_TYPE_UINT32) w.feature_layer = static_cast<int>(*static_cast<const uint32_t *>(value));
        }
        ok = read_floats(gguf, "clip.vision.image_mean", w.mean, 3) && read_floats(gguf, "clip.vision.image_std", w.std, 3) &&
             w.feature_layer >= 0 && w.get("v.patch_embd.weight") && w.get("v.position_embd.weight");
        if (!ok) error = path + " is not an OpenVLA vision tower (missing metadata or tensors)";
    }
    gguf_free(gguf);
    ggml_free(meta);
    if (!ok) w.free();
    return ok;
}

ggml_tensor *linear(ggml_context *ctx, ggml_tensor *x, ggml_tensor *weight, ggml_tensor *bias) {
    x = ggml_mul_mat(ctx, weight, x);
    return bias ? ggml_add(ctx, x, bias) : x;
}

ggml_tensor *layer_norm(ggml_context *ctx, ggml_tensor *x, ggml_tensor *weight, ggml_tensor *bias, float eps) {
    x = ggml_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, weight);
    return ggml_add(ctx, x, bias);
}

// One ViT up to and including block `feature_layer`, returning the patch
// tokens [n_embd, 256] with no final norm, as OpenVLA takes them.
ggml_tensor *tower(ggml_context *ctx, const Weights &w, ggml_tensor *image) {
    ggml_tensor *patch_weight = w.get("v.patch_embd.weight");
    const int n_embd = static_cast<int>(patch_weight->ne[3]);
    const int patch = static_cast<int>(patch_weight->ne[0]);
    ggml_tensor *x = ggml_conv_2d(ctx, patch_weight, image, patch, patch, 0, 0, 1, 1);
    x = ggml_reshape_2d(ctx, x, kVlaPatches, n_embd);
    x = ggml_cont(ctx, ggml_transpose(ctx, x));                       // [n_embd, n_patches]
    if (ggml_tensor *bias = w.get("v.patch_embd.bias")) x = ggml_add(ctx, x, bias);
    x = ggml_add(ctx, x, w.get("v.position_embd.weight"));           // positions cover the patches only
    // DINOv2 puts a class token and register tokens in front; they carry no position.
    int prefix = 0;
    if (ggml_tensor *cls = w.get("v.class_embd")) {
        ggml_tensor *front = ggml_reshape_2d(ctx, cls, n_embd, 1);
        if (ggml_tensor *registers = w.get("v.reg_embd")) front = ggml_concat(ctx, front, registers, 1);
        prefix = static_cast<int>(front->ne[1]);
        x = ggml_concat(ctx, front, x, 1);
    }
    const int n_pos = kVlaPatches + prefix;
    const int d_head = n_embd / w.n_head;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d_head));
    for (int il = 0; il <= w.feature_layer; ++il) {
        const std::string p = "v.blk." + std::to_string(il) + ".";
        ggml_tensor *cur = layer_norm(ctx, x, w.get(p + "ln1.weight"), w.get(p + "ln1.bias"), w.eps);
        ggml_tensor *q = linear(ctx, cur, w.get(p + "attn_q.weight"), w.get(p + "attn_q.bias"));
        ggml_tensor *k = linear(ctx, cur, w.get(p + "attn_k.weight"), w.get(p + "attn_k.bias"));
        ggml_tensor *v = linear(ctx, cur, w.get(p + "attn_v.weight"), w.get(p + "attn_v.bias"));
        q = ggml_permute(ctx, ggml_reshape_3d(ctx, q, d_head, w.n_head, n_pos), 0, 2, 1, 3);
        k = ggml_permute(ctx, ggml_reshape_3d(ctx, k, d_head, w.n_head, n_pos), 0, 2, 1, 3);
        v = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, v, d_head, w.n_head, n_pos), 1, 2, 0, 3));
        ggml_tensor *kq = ggml_soft_max_ext(ctx, ggml_mul_mat(ctx, k, q), nullptr, scale, 0.0f);
        cur = ggml_permute(ctx, ggml_mul_mat(ctx, v, kq), 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx, cur, n_embd, n_pos);
        cur = linear(ctx, cur, w.get(p + "attn_out.weight"), w.get(p + "attn_out.bias"));
        if (ggml_tensor *ls = w.get(p + "ls1.weight")) cur = ggml_mul(ctx, cur, ls);
        x = ggml_add(ctx, x, cur);

        cur = layer_norm(ctx, x, w.get(p + "ln2.weight"), w.get(p + "ln2.bias"), w.eps);
        cur = linear(ctx, cur, w.get(p + "ffn_up.weight"), w.get(p + "ffn_up.bias"));
        cur = ggml_gelu_erf(ctx, cur);
        cur = linear(ctx, cur, w.get(p + "ffn_down.weight"), w.get(p + "ffn_down.bias"));
        if (ggml_tensor *ls = w.get(p + "ls2.weight")) cur = ggml_mul(ctx, cur, ls);
        x = ggml_add(ctx, x, cur);
    }
    if (prefix) x = ggml_cont(ctx, ggml_view_2d(ctx, x, n_embd, kVlaPatches, x->nb[1], prefix * x->nb[1]));
    return x;
}

constexpr size_t kGraphNodes = 4096;

}  // namespace

struct VlaVision {
    Weights dinov2, siglip, proj;
    ggml_backend_t backends[2] = {nullptr, nullptr};
    int n_backends = 0;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> graph_memory;
};

VlaVision *vla_vision_load(const std::string &dinov2_path, const std::string &siglip_path,
                           const std::string &proj_path, ggml_backend_t backend, ggml_backend_t fallback,
                           std::string &error) {
    auto *vision = new VlaVision;
    if (!load_weights(dinov2_path, backend, true, vision->dinov2, error) ||
        !load_weights(siglip_path, backend, true, vision->siglip, error) ||
        !load_weights(proj_path, backend, false, vision->proj, error)) {
        vla_vision_free(vision);
        return nullptr;
    }
    if (!vision->proj.get("fc1.weight") || !vision->proj.get("fc2.weight") || !vision->proj.get("fc3.weight")) {
        error = proj_path + " is not an OpenVLA projector";
        vla_vision_free(vision);
        return nullptr;
    }
    vision->backends[vision->n_backends++] = backend;
    if (fallback && fallback != backend) vision->backends[vision->n_backends++] = fallback;
    vision->sched = ggml_backend_sched_new(vision->backends, nullptr, vision->n_backends, kGraphNodes, false, true);
    vision->graph_memory.resize(ggml_tensor_overhead() * kGraphNodes + ggml_graph_overhead_custom(kGraphNodes, false));
    return vision;
}

void vla_vision_free(VlaVision *vision) {
    if (!vision) return;
    if (vision->sched) ggml_backend_sched_free(vision->sched);
    vision->dinov2.free();
    vision->siglip.free();
    vision->proj.free();
    delete vision;
}

bool vla_vision_encode(VlaVision *vision, const uint8_t *rgb, std::vector<float> &embeddings, int *n_embd,
                       std::string &error) {
    ggml_init_params init{};
    init.mem_size = vision->graph_memory.size();
    init.mem_buffer = vision->graph_memory.data();
    init.no_alloc = true;
    ggml_context *ctx = ggml_init(init);
    ggml_cgraph *graph = ggml_new_graph_custom(ctx, kGraphNodes, false);

    // Each tower normalises the same pixels with its own mean and deviation.
    ggml_tensor *image_dinov2 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kVlaImageSize, kVlaImageSize, 3, 1);
    ggml_tensor *image_siglip = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kVlaImageSize, kVlaImageSize, 3, 1);
    ggml_set_input(image_dinov2);
    ggml_set_input(image_siglip);
    ggml_tensor *features = ggml_concat(ctx, tower(ctx, vision->dinov2, image_dinov2),
                                        tower(ctx, vision->siglip, image_siglip), 0);
    const Weights &proj = vision->proj;
    ggml_tensor *out = linear(ctx, features, proj.get("fc1.weight"), proj.get("fc1.bias"));
    out = ggml_gelu_erf(ctx, out);
    out = linear(ctx, out, proj.get("fc2.weight"), proj.get("fc2.bias"));
    out = ggml_gelu_erf(ctx, out);
    out = linear(ctx, out, proj.get("fc3.weight"), proj.get("fc3.bias"));
    ggml_set_output(out);
    ggml_build_forward_expand(graph, out);

    bool ok = ggml_backend_sched_alloc_graph(vision->sched, graph);
    if (!ok) error = "not enough memory to run the image encoder";
    if (ok) {
        const int plane = kVlaImageSize * kVlaImageSize;
        std::vector<float> pixels(static_cast<size_t>(plane) * 3);
        for (const auto &side : {std::make_pair(&vision->dinov2, image_dinov2), std::make_pair(&vision->siglip, image_siglip)}) {
            for (int c = 0; c < 3; ++c)
                for (int i = 0; i < plane; ++i)
                    pixels[static_cast<size_t>(c) * plane + i] = (rgb[i * 3 + c] / 255.0f - side.first->mean[c]) / side.first->std[c];
            ggml_backend_tensor_set(side.second, pixels.data(), 0, pixels.size() * sizeof(float));
        }
        ok = ggml_backend_sched_graph_compute(vision->sched, graph) == GGML_STATUS_SUCCESS;
        if (!ok) error = "the image encoder failed to run";
    }
    if (ok) {
        *n_embd = static_cast<int>(out->ne[0]);
        embeddings.resize(static_cast<size_t>(out->ne[0]) * out->ne[1]);
        ggml_backend_tensor_get(out, embeddings.data(), 0, embeddings.size() * sizeof(float));
    }
    ggml_backend_sched_reset(vision->sched);
    ggml_free(ctx);
    return ok;
}
