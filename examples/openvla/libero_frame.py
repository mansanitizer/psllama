import os, sys, numpy as np, torch
_load = torch.load
torch.load = lambda *a, **k: _load(*a, **{**k, "weights_only": False})   # LIBERO init states are pickled arrays
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(0)
print("tasks:", suite.n_tasks, "| task 0:", task.language)
env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file), camera_heights=256, camera_widths=256)
env.seed(0); env.reset()
obs = env.set_init_state(suite.get_task_init_states(0)[0])
for _ in range(10): obs, *_ = env.step([0, 0, 0, 0, 0, 0, -1])
img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
from PIL import Image
Image.fromarray(img).save("libero-frame.png")
print("frame", img.shape, img.dtype, "state keys", [k for k in obs if "eef" in k or "gripper" in k])
