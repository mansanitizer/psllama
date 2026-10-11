#!/usr/bin/env python3
"""Same prompts and seed on several image models; saves PNGs and a timing table.
usage: compare_images.py OUTDIR SIZE MODEL..."""
import base64, json, os, sys, time, urllib.request
out, size, models = sys.argv[1], sys.argv[2], sys.argv[3:]
key = open(os.path.expanduser("~/ps5lab/.key")).read().strip()
prompts = {
    "ramen": 'A cozy neon-lit ramen shop on a rainy Tokyo street at night, a glowing sign above the door reads "PS5 RAMEN", reflections on wet pavement, cinematic photograph',
    "fox": "a red fox sitting in an autumn meadow, golden hour, photograph",
    "astronaut": "an astronaut playing chess with a robot on the surface of the moon, Earth rising in the background, detailed illustration",
}
os.makedirs(out, exist_ok=True)
rows = []
for model in models:
    for name, prompt in prompts.items():
        body = {"model": model, "prompt": prompt, "seed": 42}
        if not model.startswith("sd-turbo"): body["size"] = size
        req = urllib.request.Request("http://ps5.tailnet.example:8081/v1/images/generations", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        t = time.time()
        try:
            with urllib.request.urlopen(req, timeout=1800) as r: d = json.load(r)
        except urllib.error.HTTPError as e:
            print(model, name, "HTTP", e.code, e.read()[:200], flush=True); continue
        png = base64.b64decode(d["data"][0]["b64_json"])
        open(f"{out}/{model}-{name}.png", "wb").write(png)
        rows.append((model, name, d["width"], d["steps"], d["timings"]["load_s"], d["timings"]["generate_s"], time.time() - t))
        print(f"{model:20s} {name:10s} {d['width']}x{d['height']} {d['steps']:2d} steps load {d['timings']['load_s']:6.1f}s gen {d['timings']['generate_s']:6.1f}s", flush=True)
json.dump(rows, open(f"{out}/timings.json", "w"), indent=1)
