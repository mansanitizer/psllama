#!/usr/bin/env python3
"""Reference for the OpenVLA image encoder: the same GGUF weights loaded into
timm's own DINOv2 and SigLIP implementations (PyTorch, CPU), then the
projector. Compares with the output of the ggml encoder (vision_test).

    reference_vision.py MODEL_DIR IMAGE.rgb [GGML_OUT.f32]"""
import sys
import numpy as np, torch, timm, gguf

def weights(path):
    reader = gguf.GGUFReader(path)
    fields = {k: f.contents() for k, f in reader.fields.items() if k.startswith("clip.")}
    return {t.name: torch.from_numpy(np.array(t.data)).float() for t in reader.tensors}, fields

def load_tower(name, path):
    w, kv = weights(path)
    model = timm.create_model(name, pretrained=False, num_classes=0, img_size=224)
    sd = {"patch_embed.proj.weight": w["v.patch_embd.weight"], "patch_embed.proj.bias": w["v.patch_embd.bias"],
          "pos_embed": w["v.position_embd.weight"].unsqueeze(0)}
    if "v.class_embd" in w:
        sd["cls_token"] = w["v.class_embd"].reshape(1, 1, -1)
        sd["reg_token"] = w["v.reg_embd"].unsqueeze(0)
    for i in range(len(model.blocks)):
        p, b = f"v.blk.{i}.", f"blocks.{i}."
        sd[b + "attn.qkv.weight"] = torch.cat([w[p + f"attn_{x}.weight"] for x in "qkv"])
        sd[b + "attn.qkv.bias"] = torch.cat([w[p + f"attn_{x}.bias"] for x in "qkv"])
        for src, dst in (("attn_out", "attn.proj"), ("ln1", "norm1"), ("ln2", "norm2"), ("ffn_up", "mlp.fc1"), ("ffn_down", "mlp.fc2")):
            sd[b + dst + ".weight"], sd[b + dst + ".bias"] = w[p + src + ".weight"], w[p + src + ".bias"]
        if p + "ls1.weight" in w:
            sd[b + "ls1.gamma"], sd[b + "ls2.gamma"] = w[p + "ls1.weight"], w[p + "ls2.weight"]
    for k, v in sd.items():
        assert model.state_dict()[k].shape == v.shape, (k, model.state_dict()[k].shape, v.shape)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(name, "| not provided by the GGUF:", sorted({m.split(".")[0] for m in missing}), "| unexpected:", unexpected)
    return model.eval(), kv

model_dir, image_path = sys.argv[1], sys.argv[2]
rgb = np.fromfile(image_path, dtype=np.uint8).reshape(224, 224, 3)
features = []
for name, file in (("vit_large_patch14_reg4_dinov2.lvd142m", "dinov2.gguf"), ("vit_so400m_patch14_siglip_224", "siglip.gguf")):
    model, kv = load_tower(name, f"{model_dir}/{file}")
    mean, std = np.array(kv["clip.vision.image_mean"], dtype=np.float32), np.array(kv["clip.vision.image_std"], dtype=np.float32)
    x = torch.from_numpy(((rgb / 255.0 - mean) / std).astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    layer = kv["clip.vision.feature_layer"][0]
    with torch.no_grad():
        out = model.get_intermediate_layers(x, n={layer})[0]     # patch tokens, no final norm
    print("  feature layer", layer, "of", len(model.blocks), "->", tuple(out.shape))
    features.append(out)
w, _ = weights(f"{model_dir}/proj.gguf")
x = torch.cat(features, dim=2)
with torch.no_grad():
    x = torch.nn.functional.gelu(x @ w["fc1.weight"].T + w["fc1.bias"])
    x = torch.nn.functional.gelu(x @ w["fc2.weight"].T + w["fc2.bias"])
    x = x @ w["fc3.weight"].T + w["fc3.bias"]
ref = x[0].numpy()
ref.tofile("reference.f32")
print("reference", ref.shape, "first values", ref[0, :3])
if len(sys.argv) > 3:
    got = np.fromfile(sys.argv[3], dtype=np.float32).reshape(ref.shape)
    cos = (got * ref).sum(1) / (np.linalg.norm(got, axis=1) * np.linalg.norm(ref, axis=1))
    print("ggml vs PyTorch: max abs difference %.5f, mean abs %.6f (values span %.2f..%.2f), cosine min %.6f" % (
        np.abs(got - ref).max(), np.abs(got - ref).mean(), ref.min(), ref.max(), cos.min()))
