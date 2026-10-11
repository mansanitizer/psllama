// Native end-to-end check of OpenVLA on the CPU backend:
//   policy_test MODEL_DIR IMAGE.rgb "instruction"
#include "vla_policy.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <fstream>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 4) { std::fprintf(stderr, "usage: %s MODEL_DIR IMAGE.rgb \"instruction\"\n", argv[0]); return 2; }
    const std::string dir = argv[1];
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model *model = llama_model_load_from_file((dir + "/llm_q4_k_m.gguf").c_str(), mp);
    if (!model) { std::fprintf(stderr, "language model failed to load\n"); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = 512;
    cp.n_threads = cp.n_threads_batch = 4;
    llama_context *context = llama_init_from_model(model, cp);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 4);
    std::string error;
    VlaVision *vision = vla_vision_load(dir + "/dinov2.gguf", dir + "/siglip.gguf", dir + "/proj.gguf", cpu, nullptr, error);
    if (!vision) { std::fprintf(stderr, "vision failed to load: %s\n", error.c_str()); return 1; }
    std::ifstream in(argv[2], std::ios::binary);
    std::vector<uint8_t> rgb(kVlaImageSize * kVlaImageSize * 3);
    if (!in.read(reinterpret_cast<char *>(rgb.data()), static_cast<std::streamsize>(rgb.size()))) { std::fprintf(stderr, "bad image file\n"); return 1; }
    VlaStep step;
    if (!vla_step(vision, context, rgb.data(), argv[3], 512, step, error)) { std::fprintf(stderr, "step failed: %s\n", error.c_str()); return 1; }
    std::printf("prompt tokens %d | vision %.1f s, prompt %.1f s, decode %.1f s\ntokens:", step.prompt_tokens, step.vision_s, step.prompt_s, step.decode_s);
    for (llama_token t : step.tokens) std::printf(" %d", t);
    std::printf("\nnormalized:");
    for (float v : step.normalized) std::printf(" %.4f", v);
    std::printf("\n");
    return 0;
}
