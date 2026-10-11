#!/usr/bin/env python3
"""Summarise ggml-vulkan perf-logger blocks (GGML_VK_PERF_LOGGER) from a klogd file.
usage: perfparse.py KLOG START END [GRAPHS_PER_BLOCK=32]   (times HH:MM:SS)
Groups ops by kind; for matrix-vector products also by weight type, with the
achieved weight bandwidth (m*k*bits/8 per call)."""
import re, sys
BITS = {"q4_K": 4.5, "q5_K": 5.5, "q6_K": 6.5625, "q8_0": 8.5, "q4_0": 4.5, "f16": 16, "f32": 32,
        "iq2_xxs": 2.0625, "iq2_xs": 2.3125, "iq2_s": 2.5625, "iq3_xxs": 3.0625, "iq3_s": 3.4375,
        "iq4_xs": 4.25, "q2_K": 2.625, "q3_K": 3.4375, "mxfp4": 4.25}
path, start, end = sys.argv[1:4]
per = int(sys.argv[4]) if len(sys.argv) > 4 else 32
text = "".join(l.split("[llama] ", 1)[1].rstrip("\n") for l in open(path, errors="replace")
               if start <= l[:8] <= end and "[llama] " in l)
op_re = re.compile(r"([A-Z][A-Z_]*(?: [^:]*?)?): (\d+) x ([\d.]+) us = ([\d.]+) us")
for block in text.split("Vulkan Timings:")[2:4]:  # skip the first (prompt) block
    groups, total = {}, 0.0
    for name, n, avg, t in op_re.findall(block):
        n, t = int(n), float(t)
        total += t
        m = re.search(r"MUL_MAT(?:_ID)?_VEC (\w+) m=(\d+) n=\d+ k=(\d+)", name)
        if m and m.group(1) in BITS:
            key = f"matvec {m.group(1)}"
            g = groups.setdefault(key, [0, 0.0, 0.0])
            g[2] += n * int(m.group(2)) * int(m.group(3)) * BITS[m.group(1)] / 8
        else:
            key = re.sub(r"[ (].*", "", name.split(" MUL_MAT")[0]) if "VEC" not in name else name.split(" m=")[0]
            g = groups.setdefault(key, [0, 0.0, 0.0])
        g[0] += n; g[1] += t
    print(f"GPU busy {total / per / 1000:.2f} ms/token ({per} tokens per block)")
    for key, (n, t, b) in sorted(groups.items(), key=lambda kv: -kv[1][1]):
        bw = f" {b / (t * 1e3):6.0f} GB/s" if b else ""
        print(f"  {key:26s} {n / per:6.0f} calls/tok {t / per / 1000:6.2f} ms/tok {100 * t / total:5.1f}%"
              f"  avg {t / n:6.1f} us{bw}")
    break
