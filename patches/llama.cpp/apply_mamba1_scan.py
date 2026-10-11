#!/usr/bin/env python3
"""Adds a Vulkan SSM_SCAN path for Mamba-1 to llama.cpp's ggml-vulkan.
usage: apply_mamba1_scan.py LLAMA_CPP_DIR (run from the directory holding ssm_scan_mamba1.comp)"""
import shutil, sys, os
root = sys.argv[1]
vk = os.path.join(root, "ggml/src/ggml-vulkan")
shutil.copy("ssm_scan_mamba1.comp", os.path.join(vk, "vulkan-shaders/ssm_scan_mamba1.comp"))

def patch(path, pairs):
    s = open(path).read()
    for a, b in pairs:
        if b in s: continue                      # already applied
        assert s.count(a) == 1, (path, s.count(a), a[:70])
        s = s.replace(a, b)
    open(path, "w").write(s)

patch(os.path.join(vk, "vulkan-shaders/vulkan-shaders-gen.cpp"), [(
    '    string_to_spv("ssm_scan_f32",          "ssm_scan.comp", {{"A_TYPE", "float"}});',
    '    string_to_spv("ssm_scan_f32",          "ssm_scan.comp", {{"A_TYPE", "float"}});\n'
    '    string_to_spv("ssm_scan_mamba1_f32",   "ssm_scan_mamba1.comp", {{"A_TYPE", "float"}});')])

cpp = os.path.join(vk, "ggml-vulkan.cpp")
patch(os.path.join(vk, "ggml-vulkan-types.h"), [(
    "    vk_pipeline pipeline_ssm_scan_f32_d128;",
    "    vk_pipeline pipeline_ssm_scan_f32_d128;\n    vk_pipeline pipeline_ssm_scan_mamba1_f32;")])

patch(cpp, [
    # the pipeline: 64 channels per workgroup, d_state 16
    ('''    ggml_vk_create_pipeline(device, device->pipeline_ssm_conv_f32, ''',
     '''    ggml_vk_create_pipeline(device, device->pipeline_ssm_scan_mamba1_f32, "ssm_scan_mamba1_f32", ssm_scan_mamba1_f32_len, ssm_scan_mamba1_f32_data, "main", 8, sizeof(vk_op_ssm_scan_push_constants), {64, 1, 1}, {16}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_ssm_conv_f32, '''),
    # which pipeline
    ('''            const uint32_t d_state = src0->ne[0];
            if (d_state == 128) {
                return ctx->device->pipeline_ssm_scan_f32_d128;''',
     '''            const uint32_t d_state = src0->ne[0];
            if (dst->src[3] && dst->src[3]->nb[1] != sizeof(float)) {
                return d_state == 16 ? ctx->device->pipeline_ssm_scan_mamba1_f32 : nullptr;   // Mamba-1
            }
            if (d_state == 128) {
                return ctx->device->pipeline_ssm_scan_f32_d128;'''),
    # supports_op: Mamba-1 with one state row per channel, d_state 16, no snapshots
    ('''                bool is_mamba2 = (op->src[3] && op->src[3]->nb[1] == sizeof(float));
                if (!is_mamba2) {
                    return false;
                }''',
     '''                bool is_mamba2 = (op->src[3] && op->src[3]->nb[1] == sizeof(float));
                if (!is_mamba2) {
                    // Mamba-1 (ssm_scan_mamba1.comp): per-channel decay vector.
                    return op->src[3] && d_state == 16 && head_dim == 1 && op->src[4]->ne[1] == 1 &&
                           ggml_get_op_params_i32(op, 0) == 1;
                }'''),
    # dispatch
    ('''    bool is_mamba2 = (src3->nb[1] == sizeof(float));
    GGML_ASSERT(is_mamba2);
''', '''    const bool is_mamba2 = (src3->nb[1] == sizeof(float));
'''),
    # Mamba-2 sizes its workgroups from d_state / subgroup_size; with Mamba-1's
    # d_state of 16 that is 0, so the Mamba-1 case must not run that division.
    ('''    const uint32_t d_state = src0->ne[0];
    uint32_t num_subgroups = d_state / ctx->device->subgroup_size;
    const uint32_t num_workgroups_x = CEIL_DIV(n_head * head_dim, num_subgroups);
    const uint32_t num_workgroups_y = n_seq;
    elements = { num_workgroups_x, num_workgroups_y, 1 };''',
     '''    if (is_mamba2) {
        const uint32_t d_state = src0->ne[0];
        uint32_t num_subgroups = d_state / ctx->device->subgroup_size;
        const uint32_t num_workgroups_x = CEIL_DIV(n_head * head_dim, num_subgroups);
        const uint32_t num_workgroups_y = n_seq;
        elements = { num_workgroups_x, num_workgroups_y, 1 };
    } else {
        elements = { n_head, n_seq, 1 };   // one invocation per channel; the pipeline divides by its 64-wide groups
    }'''),
])
print("applied")
