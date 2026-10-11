"""Voice agent test: spoken questions in, streamed reply audio out, with the
arrival time of every event measured at the client (like a frontend)."""
import base64, json, os, sys, time, urllib.request, wave
key = open(os.path.expanduser("~/ps5lab/.key")).read().strip()
out = os.path.expanduser("~/ps5lab/voice-test"); os.makedirs(out, exist_ok=True)
history = []
for turn, path in enumerate(sys.argv[1:], 1):
    body = {"audio_b64": base64.b64encode(open(path, "rb").read()).decode(), "language": "en", "messages": history}
    if not history: body.pop("messages")
    req = urllib.request.Request("http://ps5.tailnet.example:8081/v1/voice/chat", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    t = time.time(); first_text = None; chunks = []
    print(f"--- turn {turn}", flush=True)
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            d = json.loads(line[6:]); now = time.time() - t
            if d.get("type") == "transcript": print(f"  {now:6.2f}s heard: {d['text'].strip()}")
            elif d.get("type") == "text":
                if first_text is None: first_text = now; print(f"  {now:6.2f}s first text")
            elif d.get("type") == "audio":
                wav = base64.b64decode(d["audio_b64"]); chunks.append(wav)
                open(f"{out}/turn{turn}-{d['index']}.wav", "wb").write(wav)
                print(f"  {now:6.2f}s audio {d['index']}: {d['audio_s']:.1f}s of speech (made in {d['tts_s']:.2f}s): {d['text'].strip()}")
            elif d.get("type") == "done":
                print(f"  {now:6.2f}s done: {d['timings']}")
                history += [{"role": "user", "content": d["transcript"].strip()}, {"role": "assistant", "content": d["reply"].strip()}]
            elif d.get("type") == "error": print("  ERROR", d)
            elif d.get("status"): print(f"  {now:6.2f}s {d['status']} {d.get('model','')}")
    if chunks:  # join the sentence clips into one reply file
        with wave.open(f"{out}/turn{turn}-reply.wav", "wb") as w:
            for i, c in enumerate(chunks):
                import io
                with wave.open(io.BytesIO(c)) as part:
                    if i == 0: w.setparams(part.getparams())
                    w.writeframes(part.readframes(part.getnframes()))
