// Native check of the OpenVLA image encoder on the CPU backend:
//   vision_test MODEL_DIR IMAGE.rgb OUT.f32
// IMAGE.rgb is 224*224*3 raw bytes; OUT.f32 receives 256 x n_embd float32.
#include "vla_vision.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cstdio>
#include <fstream>

int main(int argc, char **argv) {
    if (argc != 4) { std::fprintf(stderr, "usage: %s MODEL_DIR IMAGE.rgb OUT.f32\n", argv[0]); return 2; }
    const std::string dir = argv[1];
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 4);
    std::string error;
    VlaVision *vision = vla_vision_load(dir + "/dinov2.gguf", dir + "/siglip.gguf", dir + "/proj.gguf", cpu, nullptr, error);
    if (!vision) { std::fprintf(stderr, "load failed: %s\n", error.c_str()); return 1; }
    std::ifstream in(argv[2], std::ios::binary);
    std::vector<uint8_t> rgb(kVlaImageSize * kVlaImageSize * 3);
    if (!in.read(reinterpret_cast<char *>(rgb.data()), static_cast<std::streamsize>(rgb.size()))) { std::fprintf(stderr, "bad image file\n"); return 1; }
    std::vector<float> embeddings;
    int n_embd = 0;
    const auto start = std::chrono::steady_clock::now();
    if (!vla_vision_encode(vision, rgb.data(), embeddings, &n_embd, error)) { std::fprintf(stderr, "encode failed: %s\n", error.c_str()); return 1; }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::ofstream(argv[3], std::ios::binary).write(reinterpret_cast<const char *>(embeddings.data()), static_cast<std::streamsize>(embeddings.size() * sizeof(float)));
    std::printf("encoded %zu x %d in %.2f s; first values %.5f %.5f %.5f\n", embeddings.size() / n_embd, n_embd, seconds, embeddings[0], embeddings[1], embeddings[2]);
    vla_vision_free(vision);
    ggml_backend_free(cpu);
    return 0;
}
