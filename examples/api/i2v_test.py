"""Video test on the PS5 (text- or image-to-video); streams progress, saves frames.
usage: i2v_test.py OUTDIR '{"model":..., "prompt":..., ...}' [START_IMAGE]"""
import base64, json, os, sys, time, urllib.request
key = open(os.path.expanduser("~/ps5lab/.key")).read().strip()
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
body = {"seed": 42, "stream": True}
body.update(json.loads(sys.argv[2]))
if len(sys.argv) > 3:
    body["init_image_b64"] = base64.b64encode(open(sys.argv[3], "rb").read()).decode()
req = urllib.request.Request("http://ps5.tailnet.example:8081/v1/videos/generations", data=json.dumps(body).encode(),
                             headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
t, last = time.time(), None
with urllib.request.urlopen(req, timeout=10800) as r:
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]": continue
        d = json.loads(line[6:])
        if "data" in d:
            for i, frame in enumerate(d["data"]):
                open(f"{out}/frame-{i:03d}.png", "wb").write(base64.b64decode(frame["b64_png"]))
            d["data"] = f"<{len(d['data'])} frames>"; print("FINAL", d, flush=True)
        elif "error" in d: print("ERROR", d, flush=True)
        elif d.get("steps", 0) <= 60 and d.get("status") == "generating" or d.get("status") != last:
            print(f"{time.time()-t:7.1f}s {d.get('status')} {d.get('step','')}/{d.get('steps','')}", flush=True)
        last = d.get("status")
