#!/usr/bin/env python3
"""ps5ctl: one command per routine step of PS5 inference work, run on Aisha.

  status                      lock, title PID, health, last notable klog lines
  lock take WHY | lock free   the shared "PS5 busy" lock (~/ps5lab/lock)
  deploy TAG                  upload out/TAG/eboot.bin, verify, swap (keeps rollback)
  restart                     kill PPSA99005 only, relaunch, wait for HTTP READY
  load MODEL [k=v ...]        load with a full explicit option set, verify it applied
  env set K=V ... | env clear write vk-env.conf (applies at the next restart)
  bench [--variants ...]      measure the loaded model; rates from STREAM DONE
  stats                       GPU heap summary from /v1/stats
  run EXPERIMENT.json         build/env/model matrix with a canary check first

klog is read from klogd's files (~/ps5lab/klog), never from the PS5 directly.
Commands that change the PS5 take the lock for $PS5LAB_USER (default: claude)
and refuse to run while someone else holds it."""
import datetime, ftplib, glob, json, os, re, socket, sys, tempfile, time, urllib.error, urllib.request

HOST = os.environ.get("PS5_HOST", "ps5.tailnet.example")
SLOT = "/data/homebrew/PPSA99005"
LAB = os.path.expanduser("~/ps5lab")
OUT = os.path.expanduser("~/ps5lm-inference/vk/out")
USER = os.environ.get("PS5LAB_USER", "claude")
DEFAULT_LOAD = {"context_tokens": 4096, "max_output_tokens": 1024, "threads": 4, "threads_batch": 4,
                "gpu_layers": 999, "batch_tokens": 128, "cache_type_k": "q4_0", "cache_type_v": "q4_0",
                "flash_attention": 1}
PROMPT = ("Implement an efficient thread-safe LRU Cache in Python with get and put methods, "
          "using collections.OrderedDict and threading.Lock. Include clean type hints.")
VARIANTS = {"greedy": {"temperature": 0.0}, "t0.1": {"temperature": 0.1},
            "t0.7": {"temperature": 0.7}, "t0.7k40": {"temperature": 0.7, "top_k": 40}}
DONE_RE = re.compile(r"STREAM DONE (\d+) TOKENS ([\d.]+) TOK/S(.*)")
SPLIT_RE = re.compile(r"(DECODE|WAIT|LOGITS|SAMPLE|SEND) ([\d.]+)")
PREFILL_RE = re.compile(r"PROMPT EVALUATED (\d+) TOKENS IN (\d+) MS")


def say(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


# ---- klog (from klogd's files) ----
def klog_path():
    return os.path.join(LAB, "klog", time.strftime("%Y-%m-%d") + ".log")


def klog_mark():
    path = klog_path()
    return path, os.path.getsize(path) if os.path.exists(path) else 0


def klog_since(mark):
    path, offset = mark
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        f.seek(offset)
        return f.read().decode("utf-8", "replace").splitlines()


def klog_wait(mark, pattern, timeout, show=None, fail=None):
    """Wait for a klog line matching pattern after mark; echo lines matching show."""
    deadline, seen = time.time() + timeout, 0
    while time.time() < deadline:
        lines = klog_since(mark)
        for line in lines[seen:]:
            if show and re.search(show, line):
                say("  ", line[13:].strip())
            if fail and re.search(fail, line):
                raise SystemExit(f"failed: {line.strip()}")
            m = re.search(pattern, line)
            if m:
                return m, line
        seen = len(lines)
        time.sleep(0.5)
    return None, None


def klogd_alive():
    path = klog_path()
    return os.path.exists(path) and time.time() - os.path.getmtime(path) < 120


# ---- PS5 access ----
def shsrv(*commands):
    s = socket.create_connection((HOST, 2323), timeout=10)
    s.settimeout(1.0)

    def drain(seconds):
        out, end = b"", time.time() + seconds
        while time.time() < end:
            try:
                d = s.recv(65536)
                if not d:
                    break
                out += d
            except socket.timeout:
                pass
        return out.decode("latin1")
    drain(1.5)
    output = ""
    for command in commands:
        s.sendall(command.encode() + b"\n")
        output += drain(3)
    s.sendall(b"exit\n")
    s.close()
    return output


def title_pids():
    return [line.split()[0] for line in shsrv("ps").splitlines()
            if len(line.split()) > 7 and line.split()[7] == "PPSA99005"]


def ftp():
    f = ftplib.FTP()
    f.connect(HOST, 2121, timeout=60)
    f.login()
    return f


def key():
    return open(os.path.join(LAB, ".key")).read().strip()


def http(method, path, body=None, timeout=30):
    req = urllib.request.Request(f"http://{HOST}:8081{path}", method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {key()}",
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


# ---- lock and state ----
def read_json(name, default):
    try:
        return json.load(open(os.path.join(LAB, name)))
    except (OSError, ValueError):
        return default


def write_json(name, value):
    path = os.path.join(LAB, name)
    with open(path + ".tmp", "w") as f:
        json.dump(value, f, indent=1)
    os.replace(path + ".tmp", path)


AUTO_LOCK = False  # taken by this command only; released when it exits


def take_lock(why, keep=False):
    global AUTO_LOCK
    lock = read_json("lock", None)
    if lock and lock["who"] != USER:
        raise SystemExit(f"PS5 is locked by {lock['who']} since {lock['since']}: {lock['why']}")
    if not lock:
        write_json("lock", {"who": USER, "why": why, "since": time.strftime("%Y-%m-%d %H:%M:%S")})
        AUTO_LOCK = AUTO_LOCK or not keep


# ---- commands ----
def cmd_status(_):
    lock = read_json("lock", None)
    say("lock:", f"{lock['who']} since {lock['since']}: {lock['why']}" if lock else "free")
    say("klogd:", "recording" if klogd_alive() else "NOT RECORDING (start klogd.py)")
    say("title PID:", ",".join(title_pids()) or "not running")
    try:
        code, health = http("GET", "/health", timeout=5)
        p = health.get("parameters", {})
        say("health:", code, health.get("model_file"), {k: p.get(k) for k in
            ("context_tokens", "cache_type_k", "cache_type_v", "flash_attention", "top_k")})
    except (OSError, ValueError):
        say("health: no answer (server busy, loading, or down)")
    state = read_json("state.json", {})
    say("deployed:", state.get("deployed", "?"), "| env:", state.get("env", {}))
    path = klog_path()
    if os.path.exists(path):
        notable = [l for l in open(path, errors="replace").read().splitlines()[-4000:]
                   if re.search(r"KV CACHE|HTTP READY|FAILED|STREAM DONE|SWITCHING MODEL|ROLLBACK", l)]
        for line in notable[-6:]:
            say("  ", line.strip())


def cmd_lock(args):
    if args[:1] == ["take"]:
        take_lock(" ".join(args[1:]) or "unspecified", keep=True)
    elif args[:1] == ["free"]:
        lock = read_json("lock", None)
        if lock and lock["who"] != USER and "--force" not in args:
            raise SystemExit(f"held by {lock['who']}; use --force to free it")
        try:
            os.remove(os.path.join(LAB, "lock"))
        except FileNotFoundError:
            pass
    cmd_status([])


def cmd_deploy(args):
    tag = args[0]
    take_lock(f"deploy {tag}")
    build = os.path.join(OUT, tag)
    eboot, elf = os.path.join(build, "eboot.bin"), os.path.join(build, "work", "eboot.elf")
    f = ftp()
    staged = f"{SLOT}/eboot.{tag}.bin"
    say("uploading", eboot)
    with open(eboot, "rb") as src:
        f.storbinary(f"STOR {staged}", src)
    # The FTP server decrypts SELFs on download: compare with the unsigned ELF.
    with tempfile.NamedTemporaryFile() as back:
        f.retrbinary(f"RETR {staged}", back.write)
        back.flush()
        diffs, first = 0, None
        with open(elf, "rb") as a, open(back.name, "rb") as b:
            pos = 0
            while True:
                x, y = a.read(1 << 20), b.read(1 << 20)
                if not x and not y:
                    break
                if x != y:
                    for i in range(max(len(x), len(y))):
                        if i >= len(x) or i >= len(y) or x[i] != y[i]:
                            diffs += 1
                            first = pos + i if first is None else first
                pos += max(len(x), len(y))
        size = os.path.getsize(elf)
    f.quit()
    if diffs > 1000 or (first is not None and first < size - 4096):
        raise SystemExit(f"readback mismatch: {diffs} bytes differ, first at {first} of {size}")
    say(f"verified: readback equals the ELF except {diffs} trailing bytes")
    rollback = f"eboot.before-{tag}-{time.strftime('%H%M%S')}.bin"
    shsrv(f"cd {SLOT}", f"mv eboot.bin {rollback}", f"mv eboot.{tag}.bin eboot.bin")
    state = read_json("state.json", {})
    state.update(deployed=tag, rollback=rollback)
    write_json("state.json", state)
    say(f"deployed {tag}; previous eboot kept as {rollback}; takes effect on restart")


def cmd_restart(_):
    take_lock("restart")
    if not klogd_alive():
        raise SystemExit("klogd is not recording; start it first")
    pids = title_pids()
    if len(pids) > 1:
        raise SystemExit(f"several PPSA99005 processes: {pids}")
    mark = klog_mark()
    t = time.time()
    if pids:
        say("stopping PPSA99005 pid", pids[0])
        shsrv(f"kill -s KILL {pids[0]}")
        time.sleep(3)
    shsrv("launch PPSA99005")
    m, line = klog_wait(mark, r"HTTP READY", 400, show=r"LOADING MODEL \d+%|KV CACHE|ENV |FAILED",
                        fail=r"LOAD FAILED|CREATE FAILED|BIND/LISTEN FAILED")
    if not m:
        raise SystemExit("no HTTP READY within 400 s")
    say(f"ready in {time.time() - t:.0f}s")


def parse_kv(args):
    out = {}
    for arg in args:
        k, v = arg.split("=", 1)
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def cmd_load(args):
    model, options = args[0], {**DEFAULT_LOAD, **parse_kv(args[1:])}
    take_lock(f"load {model}")
    mark, t = klog_mark(), time.time()
    say("loading", model, options)
    code, reply = http("POST", "/v1/models/load", {"model": model, **options}, timeout=900)
    klog_wait(mark, r"KV CACHE (LOAD|ROLLBACK)|FAILED", 5)  # klog can trail the HTTP reply
    kv = [l[13:].strip() for l in klog_since(mark) if "KV CACHE" in l or "FAILED" in l]
    for line in kv:
        say("  ", line)
    if code != 200:
        raise SystemExit(f"load failed HTTP {code}: {reply}")
    applied = reply.get("parameters", {})
    names = {"flash_attention": {"1": "enabled", "0": "disabled", "-1": "auto"}}
    wrong = {k: (v, applied.get(k)) for k, v in options.items() if k in applied and
             str(applied[k]).lower() not in (str(v).lower(), names.get(k, {}).get(str(v), "?"))}
    say(f"loaded in {time.time() - t:.0f}s" + (f"; NOT APPLIED: {wrong}" if wrong else "; all options applied"))


def cmd_env(args):
    """env set K=V (adds to the current set) | env replace K=V (only these) | env clear"""
    take_lock("env change")
    if args[:1] == ["clear"]:
        env = {}
    elif args[:1] == ["replace"]:
        env = parse_kv(args[1:])
    else:
        env = {**read_json("state.json", {}).get("env", {}), **parse_kv(args[1:])}
    text = "# written by ps5ctl; applies at startup\n" + "".join(f"{k}={v}\n" for k, v in env.items())
    f = ftp()
    with tempfile.TemporaryFile() as tmp:
        tmp.write(text.encode())
        tmp.seek(0)
        f.storbinary(f"STOR {SLOT}/vk-env.conf", tmp)
    f.quit()
    state = read_json("state.json", {})
    state["env"] = env
    write_json("state.json", state)
    say("vk-env.conf:", env or "(empty)", "- restart to apply")


def one_request(body):
    mark = klog_mark()
    t = time.time()
    req = urllib.request.Request(f"http://{HOST}:8081/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key()}", "Content-Type": "application/json"})
    text = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            if raw.strip() == b"data: [DONE]":
                break
            if raw.startswith(b"data: {"):
                try:
                    text.append(json.loads(raw[6:])["choices"][0]["delta"].get("content") or "")
                except (ValueError, KeyError, IndexError):
                    pass
    m, _ = klog_wait(mark, DONE_RE.pattern, 15)
    if not m:
        return None
    lines = klog_since(mark)
    prefill = next((PREFILL_RE.search(l) for l in lines if PREFILL_RE.search(l)), None)
    r = {"tokens": int(m.group(1)), "tok_s": float(m.group(2)), "wall_s": round(time.time() - t, 2),
         "text_sha": __import__("hashlib").sha256("".join(text).encode()).hexdigest()[:12]}
    r.update({f"{k.lower()}_ms": float(v) for k, v in SPLIT_RE.findall(m.group(3))})
    if prefill:
        r.update(prompt_tokens=int(prefill.group(1)), prefill_ms=int(prefill.group(2)))
    return r


def bench(variants, tokens, rounds, prompt, label=""):
    if not klogd_alive():
        raise SystemExit("klogd is not recording; start it first")
    code, health = http("GET", "/health", timeout=30)
    state = read_json("state.json", {})
    results = {}
    for rnd in range(rounds):
        for name in variants:
            body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": tokens, "stream": True,
                    "enable_thinking": False, "seed": 1234, **VARIANTS.get(name, {})}
            r = one_request(body)
            if r is None:
                raise SystemExit(f"{name}: no STREAM DONE line in klog (instrument check failed)")
            results.setdefault(name, []).append(r)
            ms = 1000 / r["tok_s"]
            split = "".join(f" {k[:-3]} {r[k]:.2f}" for k in
                            ("decode_ms", "wait_ms", "logits_ms", "sample_ms", "send_ms") if k in r)
            say(f"  r{rnd + 1} {name:8s} {r['tokens']:4d} tok {r['tok_s']:6.1f} tok/s {ms:6.2f} ms/tok{split} text {r['text_sha']}"
                + (f" | prefill {r['prompt_tokens']} tok {r['prefill_ms']} ms" if "prefill_ms" in r else ""))
            with open(os.path.join(LAB, "results.jsonl"), "a") as out:
                out.write(json.dumps({"time": datetime.datetime.now().isoformat(timespec="seconds"),
                                      "label": label, "variant": name, "deployed": state.get("deployed"),
                                      "env": state.get("env", {}), "model": health.get("model_file"),
                                      "load": health.get("parameters", {}), **r}) + "\n")
    return results


def cmd_bench(args):
    opts = {"variants": "greedy,t0.1,t0.7", "tokens": "128", "rounds": "1", "label": ""}
    for a in args:
        k, v = a.lstrip("-").split("=", 1)
        opts[k] = v
    take_lock("bench")
    bench(opts["variants"].split(","), int(opts["tokens"]), int(opts["rounds"]), PROMPT, opts["label"])


def cmd_stats(_):
    code, s = http("GET", "/v1/stats", timeout=30)
    v = s.get("vram", {})
    say(s.get("model", {}).get("file"), f"| driver usage {v.get('reported_usage_bytes', 0) / 1e9:.2f} GB"
        f" | budget {v.get('budget_bytes', 0) / 1e9:.2f} GB | free {v.get('free_bytes', 0) / 1e9:.2f} GB"
        f" | heap {v.get('total_bytes', 0) / 1e9:.2f} GB")


def cmd_run(args):
    """{"label", "deploy": TAG?, "env": {..}?, "models": [{"model", "options"}],
        "variants", "tokens", "rounds", "restore": {"model", "options"}?}"""
    exp = json.load(open(args[0]))
    take_lock(f"experiment {exp.get('label', args[0])}")
    state = read_json("state.json", {})
    restart = False
    if exp.get("deploy") and exp["deploy"] != state.get("deployed"):
        cmd_deploy([exp["deploy"]])
        restart = True
    if "env" in exp and exp["env"] != state.get("env", {}):
        cmd_env(["replace"] + [f"{k}={v}" for k, v in exp["env"].items()])
        restart = True
    if restart:
        cmd_restart([])
    for i, m in enumerate(exp.get("models", [])):
        if m["model"] != "startup":  # "startup": measure the model the title started with
            cmd_load([m["model"]] + [f"{k}={json.dumps(v)}" for k, v in m.get("options", {}).items()])
        if i == 0:  # canary: one short request must produce a parsed STREAM DONE line
            say("canary request")
            bench(["greedy"], 16, 1, "Say OK.", label=exp.get("label", "") + " canary")
        bench(exp.get("variants", ["greedy", "t0.1", "t0.7"]), exp.get("tokens", 128),
              exp.get("rounds", 1), PROMPT, exp.get("label", ""))
    if exp.get("restore"):
        r = exp["restore"]
        cmd_load([r["model"]] + [f"{k}={json.dumps(v)}" for k, v in r.get("options", {}).items()])


COMMANDS = {"status": cmd_status, "lock": cmd_lock, "deploy": cmd_deploy, "restart": cmd_restart,
            "load": cmd_load, "env": cmd_env, "bench": cmd_bench, "stats": cmd_stats, "run": cmd_run}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(__doc__)
    try:
        COMMANDS[sys.argv[1]](sys.argv[2:])
    finally:
        if AUTO_LOCK:
            try:
                os.remove(os.path.join(LAB, "lock"))
            except FileNotFoundError:
                pass
