#!/usr/bin/env python3
"""OpenVLA on the PS5 driving LIBERO (MuJoCo) on this machine.

Each control step renders the third-person camera, prepares the image as the
OpenVLA LIBERO evaluation does, posts it with the task's instruction to the
PS5 (/v1/vla/act), scales the seven returned values with the dataset
statistics and steps the simulator. Follows experiments/robot/libero in the
OpenVLA repository: 10 idle steps first, 220 steps at most for LIBERO-Spatial,
centre crop, gripper binarised and sign-flipped.

    PYTHONPATH=LIBERO MUJOCO_GL=egl python libero_eval.py --tasks 0-9 --trials 3
"""
import argparse, base64, json, os, time, urllib.error, urllib.request
import numpy as np, torch

_load = torch.load
torch.load = lambda *a, **k: _load(*a, **{**k, "weights_only": False})   # LIBERO init states are pickled arrays
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image
from vla_image import prepare

PS5 = os.environ.get("PS5_URL", "http://ps5.tailnet.example:8081") + "/v1/vla/act"
KEY = open(os.path.expanduser("~/ps5lab/.key")).read().strip()
IDLE = [0, 0, 0, 0, 0, 0, -1]


def act(rgb, instruction, model):
    body = {"model": model, "instruction": instruction, "rgb_b64": base64.b64encode(rgb.tobytes()).decode()}
    request = urllib.request.Request(PS5, json.dumps(body).encode(), {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(request, timeout=600))
    except urllib.error.HTTPError as error:
        raise SystemExit(f"PS5 refused the request: {error.code} {error.read().decode()[:300]}")


def to_robot(normalized, stats):
    """[-1, 1] per dimension -> the robot's units, then OpenVLA's gripper convention."""
    q01, q99, mask = np.array(stats["q01"]), np.array(stats["q99"]), np.array(stats["mask"])
    a = np.array(normalized, dtype=np.float64)
    a = np.where(mask, 0.5 * (a + 1) * (q99 - q01) + q01, a)
    a[-1] = np.sign(2 * a[-1] - 1)      # gripper: [0, 1] -> -1 or +1
    a[-1] = -a[-1]                      # the training data's sign is the reverse of the simulator's
    return a


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--model", default="openvla-libero-spatial")
    parser.add_argument("--stats", default=os.path.expanduser("~/models/openvla/dataset_statistics.json"))
    parser.add_argument("--tasks", default="0-9")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--out", default="libero-runs")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    stats = json.load(open(args.stats))[args.suite]["action"]
    first, _, last = args.tasks.partition("-")
    task_ids = range(int(first), int(last or first) + 1)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    results, step_s = [], []
    for task_id in task_ids:
        task = suite.get_task(task_id)
        env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
                                 camera_heights=256, camera_widths=256)
        env.seed(0)
        init_states = suite.get_task_init_states(task_id)
        for trial in range(args.trials):
            env.reset()
            obs = env.set_init_state(init_states[trial])
            for _ in range(10):
                obs, *_ = env.step(IDLE)
            frames, success, steps, started = [], False, 0, time.time()
            for steps in range(1, args.max_steps + 1):
                frame = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                frames.append(frame)
                t = time.time()
                reply = act(prepare(frame), task.language, args.model)
                step_s.append(time.time() - t)
                obs, _, done, _ = env.step(to_robot(reply["normalized"], stats).tolist())
                if done:
                    success = True
                    break
            results.append({"task": task_id, "instruction": task.language, "trial": trial, "success": success, "steps": steps,
                            "seconds": round(time.time() - started, 1)})
            print(("OK  " if success else "FAIL"), f"task {task_id} trial {trial}: {steps} steps in {results[-1]['seconds']:.0f}s |", task.language, flush=True)
            if trial == 0:
                clip = [Image.fromarray(f) for f in frames[::4]]
                clip[0].save(os.path.join(args.out, f"task{task_id}-trial0-{'ok' if success else 'fail'}.gif"), save_all=True,
                             append_images=clip[1:], duration=200, loop=0)
            json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=1)
        env.close()
    wins = sum(r["success"] for r in results)
    print(f"{args.suite}: {wins}/{len(results)} episodes succeeded ({100 * wins / len(results):.0f}%), "
          f"{np.mean(step_s):.2f}s per control step over {len(step_s)} steps")
