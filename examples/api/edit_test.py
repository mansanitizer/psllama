import base64, json, urllib.request, urllib.error, time, sys
key = open(".key").read().strip()
def gen(name, body):
    r = urllib.request.Request("http://ps5.tailnet.example:8081/v1/images/generations", json.dumps(body).encode(),
        {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    t = time.time()
    try:
        d = json.load(urllib.request.urlopen(r, timeout=900))
    except urllib.error.HTTPError as e:
        print(name, "FAILED", e.code, e.read().decode()[:300]); return None
    png = base64.b64decode(d["data"][0]["b64_json"])
    open("edit-test/%s.png" % name, "wb").write(png)
    print(name, "ok", d["width"], d["height"], d["timings"], "wall %.0fs" % (time.time() - t), flush=True)
    return base64.b64encode(png).decode()
base = gen("1-base", {"model": "flux2-klein-4b", "prompt": "a red fox sitting in fresh snow, golden hour, detailed fur, photo", "width": 512, "height": 512, "seed": 7})
if base:
    gen("2-ref-edit", {"model": "flux2-klein-4b", "prompt": "the same fox wearing a blue knitted scarf, in a neon-lit city street at night", "width": 512, "height": 512, "seed": 7, "ref_images_b64": [base]})
    gen("3-img2img", {"model": "flux2-klein-4b", "prompt": "a watercolor painting of a red fox in snow", "width": 512, "height": 512, "seed": 7, "init_image_b64": base, "strength": 0.6})
