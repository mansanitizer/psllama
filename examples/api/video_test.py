"""Wan2.1 1.3B smoke test on the PS5: streams progress, saves PNG frames."""
import base64, json, os, sys, time, urllib.request
key = open(os.path.expanduser("~/ps5lab/.key")).read().strip()
out = os.path.expanduser("~/ps5lab/video-test"); os.makedirs(out, exist_ok=True)
body = {"model": "wan2.1-t2v-1.3b", "prompt": "a red fox running through a snowy forest, cinematic", "seed": 42, "stream": True}
body.update(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
req = urllib.request.Request("http://ps5.tailnet.example:8081/v1/videos/generations", data=json.dumps(body).encode(),
                             headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
last = {}
t = time.time()
with urllib.request.urlopen(req, timeout=7200) as r:
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]": continue
        d = json.loads(line[6:])
        if "data" in d:
            for i, frame in enumerate(d["data"]):
                open(f"{out}/frame-{i:03d}.png", "wb").write(base64.b64decode(frame["b64_png"]))
            d["data"] = f"<{len(d['data'])} frames>"; print("FINAL", d, flush=True)
        elif "error" in d: print("ERROR", d, flush=True)
        else:
            s = d.get("status")
            if s != last.get("s") or d.get("step") == d.get("steps"):
                print(f"{time.time()-t:7.1f}s", d, flush=True)
            last = {"s": s}
