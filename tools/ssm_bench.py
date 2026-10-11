#!/usr/bin/env python3
"""Mamba on the PS5: GPU (gpu_layers=999) against CPU (gpu_layers=0) at several
prompt lengths. Pauses the LIBERO evaluation at its next episode boundary,
runs, and resumes it. Rates are read from the title's own klog lines."""
import glob, json, os, re, subprocess, time, urllib.request, urllib.error
LAB = os.path.expanduser("~/ps5lab"); VLA = os.path.expanduser("~/vla")
KEY = open(LAB + "/.key").read().strip()
import sys
MODELS = sys.argv[1:] or ["mamba-130m-hf.Q4_K_M.gguf"]
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def episodes(): return len(re.findall(r"^(OK|FAIL)", open(VLA + "/libero-runs.log").read(), re.M))
def klog_tail():
    path = sorted(glob.glob(LAB + "/klog/*.log"))[-1]
    with open(path, errors="replace") as f: return f.readlines()[-400:]
def chat(prompt, n):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": n, "temperature": 0}
    r = urllib.request.Request("http://ps5.tailnet.example:8081/v1/chat/completions", json.dumps(body).encode(), {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    t = time.time()
    try: d = json.load(urllib.request.urlopen(r, timeout=900))
    except urllib.error.HTTPError as e: return None, "HTTP %d %s" % (e.code, e.read().decode()[:160])
    except Exception as e: return None, "FAILED %r" % e
    return time.time() - t, d
HOLD = not os.environ.get("NO_PAUSE")     # NO_PAUSE=1: LIBERO is already paused (or absent); leave it alone
if HOLD:
    n = episodes(); log("waiting for the current LIBERO episode to end (done so far: %d)" % n)
    while episodes() == n and subprocess.run(["pgrep", "-f", "libero_eval.py --tasks 0-9"], capture_output=True).returncode == 0: time.sleep(0.2)
    subprocess.run(["pkill", "-STOP", "-f", "libero_eval.py --tasks 0-9"]); log("LIBERO paused after %d episodes" % episodes())
if os.environ.get("PRE_CMD"):     # e.g. deploy a new build while the evaluation is paused
    pre = subprocess.run(os.environ["PRE_CMD"], shell=True, capture_output=True, text=True)
    for line in (pre.stdout + pre.stderr).strip().splitlines()[-14:]: log("  pre:", line[:150])
sentence = "The quick brown fox jumps over the lazy dog. "
try:
    for MODEL in MODELS:
        log("=====", MODEL)
        for label, layers in (("GPU, gpu_layers=999", 999), ("CPU, gpu_layers=0", 0)):
            out = subprocess.run(["python3", LAB + "/ps5ctl.py", "load", MODEL, "gpu_layers=%d" % layers, "cache_type_k=f16", "cache_type_v=f16"], capture_output=True, text=True)
            log(label, "|", (out.stdout.strip().splitlines() or ["?"])[-1][:90])
            for repeat in (1, 24, 100, 330):
                wall, d = chat(sentence * repeat + "Continue the story.", 64)
                if wall is None: log("   prompt x%d:" % repeat, d); continue
                time.sleep(2.0)
                lines = klog_tail()
                pe = [m for m in (re.search(r"PROMPT EVALUATED (\d+) TOKENS IN (\d+) MS", l) for l in lines) if m][-1]
                dn = [m for m in (re.search(r"DONE (\d+) TOKENS ([\d.]+) TOK/S", l) for l in lines) if m][-1]
                ptok, pms = int(pe.group(1)), int(pe.group(2))
                log("   prompt %4d tokens: read at %6.0f tok/s (%5d ms) | wrote %s tokens at %5.1f tok/s | wall %.1fs" % (ptok, 1000.0 * ptok / max(pms, 1), pms, dn.group(1), float(dn.group(2)), wall))
            if True:   # the same questions on GPU and CPU: greedy answers should agree
                for q in ("What is the capital of France?", "Write two sentences about the ocean.", "List three fruits."):
                    wall, d = chat(q, 60)
                    log("   Q:", q, "| A:", repr(d["choices"][0]["message"]["content"][:170]) if wall is not None else d)
finally:
    if HOLD:
        subprocess.run(["pkill", "-CONT", "-f", "libero_eval.py --tasks 0-9"]); log("LIBERO resumed")
log("BENCH DONE")
