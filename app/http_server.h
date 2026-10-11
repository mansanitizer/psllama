// Minimal single-client HTTP inference service for the PS5 Vulkan title.
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

struct llama_model;
struct llama_context;

// KV cache types (ggml_type) and flash attention mode (-1 auto, 0 off, 1 on)
// the startup context was created with; call before run_http_server.
void set_startup_kv_options(int cache_type_k, int cache_type_v, int flash_attention);

void run_http_server(llama_model *model, llama_context *context, unsigned short port,
                     int context_tokens, int max_output_tokens, const char *model_file,
                     int threads, int gpu_layers, float default_temperature);
