#!/usr/bin/env python3
"""The PS5 drives a simulated robot arm. Each turn this script renders the
camera image and makes two calls to the vision model on the PS5: one to
describe the image, one to choose a command from that description and the
task. It then runs the command (see blocks_env.py for the commands).

    MUJOCO_GL=egl venv/bin/python vla_agent.py [--episodes 12] [--blind] [--out DIR]

--blind sends the same prompts without the image, as a control: a model that
cannot see has to guess where each colour is."""
import argparse, base64, io, json, os, re, time, urllib.error, urllib.request
import numpy as np
from PIL import Image
from blocks_env import BlocksEnv, COLORS

PS5 = "http://ps5.tailnet.example:8081/v1/chat/completions"
KEY = open(os.path.expanduser("~/ps5lab/.key")).read().strip()
LOOK = """This is a camera view of a robot arm working at a white table. At the back of the table are three grey square pads in a row: left, middle and right. Each pad either has one small coloured cube standing on it or is empty. Describe the scene in three short sentences:
1. For each pad (left, middle, right): the colour of the cube on it, or that the pad is empty.
2. Whether any cubes are sitting on the round yellow plate at the front, and their colours from bottom to top.
3. Whether the gripper of the robot, at the top of the image, is holding a cube, and its colour."""
DECIDE = """You control a robot arm at a table with three places in a row (left, middle, right) and a plate.
The commands are: pick left, pick middle, pick right, pick plate, place left, place middle, place right, place plate, done.
Commands name places, never colours. "pick X" grasps the block at place X; the gripper holds one block at a time. "place X" puts the held block down at place X; placing where a block already is stacks them. "done" ends the task; say it as soon as the scene shows the task is complete.

Task: {task}
What the camera shows now (the pads are the places): {scene}
What you have done so far, oldest first: {history}

Reply in exactly this form, two lines:
Left to do: <what the task still needs, given what the camera shows, or nothing>
Command: <one command>"""
COMMAND = re.compile(r"command:\s*((?:pick|place)\s+(?:left|middle|right|plate)|done)\b")


def make_task(rng):
    """Returns (instruction, check(env) -> bool). Each needs the image to find a colour.
    The checks are strict: the plate must hold exactly what was asked for."""
    a, b, c = (COLORS[i] for i in rng.permutation(3))
    kind = rng.integers(3)
    if kind == 0:
        return f"Put the {a} block on the plate.", lambda env: env.scene()["plate"] == a
    if kind == 1:
        return f"Put the block that is neither {b} nor {c} on the plate.", lambda env: env.scene()["plate"] == a
    return (f"Put the {a} block on the plate, then stack the {b} block on top of it.",
            lambda env: env.scene()["plate"] == f"{a},{b}" and env.stacked(b, a))


def chat(content, max_tokens):
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": max_tokens, "temperature": 0}
    request = urllib.request.Request(PS5, json.dumps(body).encode(), {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(request, timeout=300))["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as error:
        raise SystemExit(f"PS5 refused the request: {error.code} {error.read().decode()[:300]}")


def ask(image, task, history, blind):
    """One turn, two calls to the PS5: describe the image, then choose a command
    from that description. Returns (scene description, decision reply, seconds)."""
    started = time.time()
    if blind:
        scene = "(the camera is off; you have to guess where each colour is)"
    else:
        png = io.BytesIO()
        Image.fromarray(image).save(png, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(png.getvalue()).decode()
        said = chat([{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": LOOK}], 110)
        # Keep the three numbered sentences; drop any greeting around them.
        scene = " ".join(line.strip() for line in said.splitlines() if re.match(r"\s*[123][.)]", line)) or said
    steps = "; ".join(f"{i + 1}. {step}" for i, step in enumerate(history)) if history else "nothing yet"
    reply = chat([{"type": "text", "text": DECIDE.format(task=task, scene=scene, history=steps)}], 70)
    return scene, reply, time.time() - started


def episode(env, seed, blind, max_turns=7):
    rng = np.random.default_rng(1000 + seed)
    layout = env.reset(seed)
    task, check = make_task(rng)
    history, think_s, trace = [], [], []
    for _ in range(max_turns):
        truth = env.scene()
        scene, reply, seconds = ask(env.image(), task, history, blind)
        think_s.append(seconds)
        trace.append({"truth": truth, "scene": scene, "reply": reply})
        match = COMMAND.search(reply.lower())
        if not match:
            history.append("(reply had no valid command; commands name places, e.g. pick left)")
            continue
        command = match.group(1)
        if command == "done":
            history.append("done")
            break
        history.append(f"{command} -> {env.run(command)}")
    return {"seed": seed, "layout": layout, "task": task, "commands": history, "success": bool(check(env)),
            "declared_done": bool(history) and history[-1] == "done", "think_s": [round(s, 2) for s in think_s],
            "final_scene": env.scene(), "trace": trace}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--blind", action="store_true")
    parser.add_argument("--out", default="runs")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    env = BlocksEnv()
    results = []
    for seed in range(args.episodes):
        result = episode(env, seed, args.blind)
        results.append(result)
        print(("OK  " if result["success"] else "FAIL"), seed, result["task"], "|", "; ".join(result["commands"]),
              "| %.1fs/turn" % (sum(result["think_s"]) / len(result["think_s"])), flush=True)
        if not args.blind and env.frames:
            frames = [Image.fromarray(f) for f in env.frames[::2]]
            frames[0].save(os.path.join(args.out, f"episode-{seed:02d}.gif"), save_all=True, append_images=frames[1:], duration=120, loop=0)
    wins = sum(r["success"] for r in results)
    decisions = [s for r in results for s in r["think_s"]]
    print(f"{'blind' if args.blind else 'vision'}: {wins}/{len(results)} tasks done, {len(decisions)} turns, {sum(decisions) / len(decisions):.1f}s each")
    json.dump(results, open(os.path.join(args.out, "blind.json" if args.blind else "vision.json"), "w"), indent=1)
