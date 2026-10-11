#!/usr/bin/env python3
"""A tabletop for the PS5 to act in: a simulated Fetch arm, three coloured
blocks in a row (left, middle, right as the camera sees them) and a yellow
plate. The arm is driven by four high-level commands; the motion for each is
scripted here, so whoever issues the commands only has to decide what to do.

    pick left|middle|right|plate     grasp whatever is at that place
    place left|middle|right|plate    put down what is held there
    done                             declare the task finished

Run with MUJOCO_GL=egl (headless)."""
import os, shutil
import numpy as np
import mujoco
import gymnasium_robotics.envs.fetch.pick_and_place as fetch_pick

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(os.path.dirname(fetch_pick.__file__), "..", "assets", "fetch")
COLORS = ["red", "green", "blue"]                       # object0, object1, object2
PLACES = {"left": (1.22, 0.57), "middle": (1.22, 0.75), "right": (1.22, 0.93), "plate": (1.44, 0.75)}
TABLE_Z = 0.4
BLOCK_Z = TABLE_Z + 0.025


class BlocksEnv:
    def __init__(self, size=896):
        shutil.copy(os.path.join(HERE, "ps5_blocks.xml"), os.path.join(ASSETS, "ps5_blocks.xml"))
        fetch_pick.MODEL_XML_PATH = os.path.join("fetch", "ps5_blocks.xml")
        self.env = fetch_pick.MujocoFetchPickAndPlaceEnv()
        self.model, self.data = self.env.model, self.data_of(self.env)
        self.model.vis.global_.offwidth = self.model.vis.global_.offheight = size
        self.renderer = mujoco.Renderer(self.model, size, size)
        self.frames = []          # camera frames recorded while the arm moves
        self.record_every = 4
        self._tick = 0

    @staticmethod
    def data_of(env):
        return env.data

    def reset(self, seed):
        """Shuffles which colour sits at which place. Returns {place: colour}."""
        self.env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        order = rng.permutation(3)
        self.layout = {}
        for place, block in zip(["left", "middle", "right"], order):
            x, y = PLACES[place]
            qpos = np.array([x, y, BLOCK_Z, 1, 0, 0, 0], dtype=float)
            self.env._utils.set_joint_qpos(self.model, self.data, f"object{block}:joint", qpos)
            self.layout[place] = COLORS[block]
        mujoco.mj_forward(self.model, self.data)
        self.frames, self._tick = [], 0
        for _ in range(10):
            self._step(np.zeros(3), 1.0)
        self.park()
        return dict(self.layout)

    # ---- observation ----
    def image(self):
        self.renderer.update_scene(self.data, camera="front")
        return self.renderer.render().copy()

    def block_position(self, colour):
        return self.data.body(f"object{COLORS.index(colour)}").xpos.copy()

    def gripper(self):
        return self.env._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()

    def on_plate(self, colour):
        p = self.block_position(colour)
        return bool(np.linalg.norm(p[:2] - np.array(PLACES["plate"])) < 0.06 and p[2] < TABLE_Z + 0.06)

    def scene(self):
        """Ground truth in the words the model is asked to use."""
        truth = {}
        for place in ("left", "middle", "right", "plate"):
            here = [c for c in COLORS if np.linalg.norm(self.block_position(c)[:2] - PLACES[place]) < 0.06
                    and self.block_position(c)[2] < TABLE_Z + 0.1]
            truth[place] = ",".join(sorted(here, key=lambda c: self.block_position(c)[2])) or "empty"
        truth["gripper"] = self.holding() or "empty"
        return truth

    def holding(self):
        """The colour of the block lifted off the table, if any."""
        for colour in COLORS:
            if self.block_position(colour)[2] > TABLE_Z + 0.08:
                return colour
        return None

    # ---- scripted motion ----
    def _step(self, delta, grip):
        action = np.concatenate([np.clip(np.asarray(delta) / 0.05, -1, 1), [grip]]).astype(np.float32)
        self.env.step(action)
        self._tick += 1
        if self._tick % self.record_every == 0:
            self.frames.append(self.image()[::2, ::2])   # half size is plenty for the replay

    def _move(self, target, grip, steps=60, tolerance=0.006):
        target = np.asarray(target, dtype=float)
        for _ in range(steps):
            error = target - self.gripper()
            if np.linalg.norm(error) < tolerance:
                break
            self._step(error, grip)

    def _hold(self, grip, steps=12):
        for _ in range(steps):
            self._step(np.zeros(3), grip)

    def pick(self, place):
        x, y = PLACES[place]
        # Grasp the block actually resting near that place, if there is one.
        near = [c for c in COLORS if np.linalg.norm(self.block_position(c)[:2] - (x, y)) < 0.06]
        if near:
            x, y, z = self.block_position(near[0])
        else:
            z = BLOCK_Z
        self._move((x, y, TABLE_Z + 0.2), 1.0)
        self._move((x, y, z + 0.003), 1.0)
        self._hold(-1.0)
        self._move((x, y, TABLE_Z + 0.22), -1.0)

    def place(self, place):
        x, y = PLACES[place]
        occupied = any(np.linalg.norm(self.block_position(c)[:2] - (x, y)) < 0.05 and self.block_position(c)[2] < TABLE_Z + 0.06 for c in COLORS)
        z = BLOCK_Z + (0.05 if occupied else 0.0)
        self._move((x, y, TABLE_Z + 0.22), -1.0)
        self._move((x, y, z + 0.006), -1.0)
        self._hold(1.0)
        self._move((x, y, TABLE_Z + 0.22), 1.0)

    def park(self):
        """Moves the gripper up and back: clear of the blocks, but with anything it holds in plain view."""
        self._move((1.08, 0.75, TABLE_Z + 0.27), -1.0 if self.holding() else 1.0, steps=80, tolerance=0.01)

    def stacked(self, top, bottom):
        t, b = self.block_position(top), self.block_position(bottom)
        return bool(np.linalg.norm(t[:2] - b[:2]) < 0.03 and 0.035 < t[2] - b[2] < 0.065)

    def run(self, command):
        """Runs 'pick X' or 'place X'. Returns a short factual result."""
        verb, _, place = command.partition(" ")
        if verb not in ("pick", "place") or place not in PLACES:
            return "unknown command"
        if verb == "pick":
            if self.holding():
                return "already holding a block"
            self.pick(place)
            self.park()
            return "holding a block" if self.holding() else "the gripper closed on nothing"
        if not self.holding():
            return "not holding anything"
        self.place(place)
        self.park()
        return "released"


if __name__ == "__main__":
    # Self-test: the scripted motion alone, told exactly what to do.
    import sys
    from PIL import Image
    env = BlocksEnv()
    wins = 0
    for seed in range(6):
        layout = env.reset(seed)
        colour = COLORS[seed % 3]
        place = next(p for p, c in layout.items() if c == colour)
        if seed == 0:
            Image.fromarray(env.image()).save(os.path.join(HERE, "scene.png"))
        r1 = env.run(f"pick {place}")
        if seed == 0:
            Image.fromarray(env.image()).save(os.path.join(HERE, "scene-holding.png"))
        r2 = env.run("place plate")
        other = next(p for p, c in layout.items() if c != colour)
        env.run(f"pick {other}")
        env.run("place plate")
        ok = env.on_plate(colour) and env.stacked(layout[other], colour)
        wins += ok
        print(seed, layout, colour, place, r1, "|", r2, "| on plate and stacked:", ok, flush=True)
    Image.fromarray(env.image()).save(os.path.join(HERE, "scene-after.png"))
    print("scripted success", wins, "/ 6")
