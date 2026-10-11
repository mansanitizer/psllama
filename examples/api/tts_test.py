"""TTS on the PS5 with and without a reference voice, then Whisper on the PS5
transcribes each result as a check."""
import base64, json, os, urllib.request
key = open(os.path.expanduser("~/ps5lab/.key")).read().strip()
def post(path, body, timeout=900):
    req = urllib.request.Request("http://ps5.tailnet.example:8081" + path, data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r: return json.load(r)
    except urllib.error.HTTPError as e: return {"error": e.read()[:400].decode(errors="replace")}
text = "Hello! I am speaking to you from a PlayStation 5. This voice was generated on the console's graphics chip."
speaker = base64.b64encode(open("speaker.wav", "rb").read()).decode()
outs = []
for name, extra in [("plain", {}), ("cloned", {"speaker_b64": speaker})]:
    d = post("/v1/audio/speech", {"input": text, "language": "en", "seed": 7, **extra})
    if "error" in d: print(name, "ERROR", d["error"]); continue
    open(f"tts-{name}.wav", "wb").write(base64.b64decode(d["audio_b64"]))
    print(name, "frames", d["frames"], "rate", d["sample_rate"], d["timings"]); outs.append(name)
for name in outs:
    d = post("/v1/audio/transcriptions", {"audio_b64": base64.b64encode(open(f"tts-{name}.wav", "rb").read()).decode(), "language": "en"})
    print(name, "heard:", d.get("text", d).strip() if isinstance(d.get("text"), str) else d)
