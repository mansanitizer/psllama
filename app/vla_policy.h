// One OpenVLA control step on top of llama.cpp: image + instruction in, seven
// action tokens out. The caller owns the language model and its context.
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "llama.h"
#include "vla_vision.h"
#include <string>

constexpr int kVlaActionDim = 7;   // x, y, z, roll, pitch, yaw, gripper

struct VlaStep {
    llama_token tokens[kVlaActionDim] = {};
    // Each action dimension in [-1, 1] (the centre of its bin). The caller
    // scales it with the dataset's statistics (q01/q99) for the robot.
    float normalized[kVlaActionDim] = {};
    int prompt_tokens = 0;
    double vision_s = 0.0, prompt_s = 0.0, decode_s = 0.0;
};

// rgb: 224x224x3 bytes. `context` is cleared and reused; `n_batch` is the
// context's batch size.
bool vla_step(VlaVision *vision, llama_context *context, const uint8_t *rgb, const std::string &instruction,
              int n_batch, VlaStep &step, std::string &error);
