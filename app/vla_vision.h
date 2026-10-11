// OpenVLA's image encoder on ggml: a DINOv2 ViT-L/14 and a SigLIP So400m/14
// (both at 224 px) whose patch features are concatenated and projected by a
// three-layer MLP into 256 embeddings for the Llama-2 language model.
// Weights: the dinov2.gguf / siglip.gguf / proj.gguf split of an OpenVLA
// checkpoint (clip-style tensor names, F32).
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "ggml-backend.h"
#include <cstdint>
#include <string>
#include <vector>

struct VlaVision;

// `backend` runs the encoder; `fallback` (may be null) takes any operation the
// first one lacks. Returns null and sets `error` on failure.
VlaVision *vla_vision_load(const std::string &dinov2_path, const std::string &siglip_path,
                           const std::string &proj_path, ggml_backend_t backend, ggml_backend_t fallback,
                           std::string &error);
void vla_vision_free(VlaVision *vision);

constexpr int kVlaImageSize = 224;     // input is kVlaImageSize x kVlaImageSize RGB
constexpr int kVlaPatches = 256;       // embeddings produced per image

// rgb: 224*224*3 bytes, row-major, R G B per pixel. embeddings: kVlaPatches
// rows of `*n_embd` floats (4096 for Llama-2 7B).
bool vla_vision_encode(VlaVision *vision, const uint8_t *rgb, std::vector<float> &embeddings, int *n_embd,
                       std::string &error);
