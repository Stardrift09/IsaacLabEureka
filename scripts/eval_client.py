# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""IsaacLab-side evaluation driver for the PickItUp replay env.

This is the client half of LeRobot's ``eval_policy_http_server``. It is the
IsaacLab repo's ``scripts/eval_policy.py`` with the local policy swapped for a
**remote** one: instead of ``action = policy(obs)`` it POSTs the observation to
the policy server and applies the action it returns.

It drives ``ReplayPickItUpOneEnv``, whose eval interface already produces obs in
the LeRobot dataset layout:
  - ``observation.state``      : [8]  = 7 arm joint pos + 1 gripper (finger_joint1)
  - ``observation.images.image``  : agentview RGB  HxWx3 uint8
  - ``observation.images.image2`` : wrist eye-in-hand RGB HxWx3 uint8
and applies an 8-dim action (7 arm joint-position targets + 1 gripper) through
physics (joint PD, gains 3000/260), exactly matching how the dataset was made.

IMPORTANT — environment separation:
    This file imports **no** ``lerobot``. IsaacLab and LeRobot don't co-resolve
    their deps (IsaacLab pins numpy<2, LeRobot pulls numpy>=2), so they run in
    separate Python envs. Only ``isaaclab`` / ``gymnasium`` / ``torch`` / ``numpy``
    / stdlib are used here. The wire format is JSON+base64 (no pickle, numpy-
    version-agnostic).

This file lives in two places (kept in sync): the IsaacLabEureka repo at
``scripts/eval_client.py`` and the LeRobot tree at
``~/lerobot/examples/isaaclab/eval_client.py``. Run it from the eureka env either way.

Run in the IsaacLab env (conda ``eureka``):
```shell
# self-test (no server needed): replays the demo's own actions, ~48/50 success
~/miniconda3/envs/eureka/bin/python -u scripts/eval_client.py \
    --episodes 5 --self_test

# evaluate a policy (randomized inits):
~/miniconda3/envs/eureka/bin/python -u scripts/eval_client.py \
    --task EvalPickItUpOneEnv --episodes 20 --server_url http://<host>:8080 \
    --policy_type act --pretrained_path <checkpoint_dir> --policy_device cuda

# collect a dataset FROM the policy (randomized) -> successful/ + unsuccessful/:
~/miniconda3/envs/eureka/bin/python -u scripts/eval_client.py \
    --task EvalPickItUpOneEnv --episodes 50 --server_url http://<host>:8080 \
    --policy_type act --pretrained_path <checkpoint_dir> \
    --record_dir logs/collect_policy
```
"""

from __future__ import annotations

import argparse
import base64
import json
import urllib.request

from isaaclab.app import AppLauncher

# ----------------------------------------------------------------------------
# 1. CLI + launch IsaacSim (before importing isaaclab task modules)
# ----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="IsaacLab PickItUp eval driver against a remote LeRobot policy.")
parser.add_argument("--task", type=str, default="EvalPickItUpOneEnv")
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--controller", type=str, default="joint_pd", choices=["joint_pd", "osc"])
parser.add_argument("--kp", type=float, default=3000.0, help="joint_pd arm stiffness (sweep winner 3000).")
parser.add_argument("--kd", type=float, default=260.0, help="joint_pd arm damping (sweep winner 260).")
parser.add_argument("--task_text", type=str, default="pick up the alphabet_soup and put it in the basket")
parser.add_argument("--json_out", type=str, default=None)
# Data collection: record a LeRobot dataset (videos + state/action) while rolling
# out the policy, split into successful/ vs unsuccessful/ by the terminated flag.
# Use --task EvalPickItUpOneEnv for randomized inits (recommended for collection).
parser.add_argument("--record_dir", type=str, default=None,
                    help="If set, record a LeRobot-convertible dataset during the rollout.")
parser.add_argument("--horizon", type=int, default=None,
                    help="Max steps per episode for recording (default: longest demo).")
# Remote policy server
parser.add_argument("--server_url", type=str, default="http://127.0.0.1:8080")
parser.add_argument("--request_timeout", type=float, default=120.0)
parser.add_argument("--policy_type", type=str, default=None)
parser.add_argument("--pretrained_path", type=str, default=None)
parser.add_argument("--policy_device", type=str, default="cuda")
parser.add_argument(
    "--rename_map", type=str, default=None, help='JSON, e.g. \'{"observation.images.image":"..."}\''
)
# Self-test: ignore the server, replay the demo's own actions (sanity check).
parser.add_argument("--self_test", action="store_true", help="Use demo actions instead of the remote policy.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Visual policy -> cameras are always needed.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ----------------------------------------------------------------------------
# 2. IsaacLab imports (after the app is live)
# ----------------------------------------------------------------------------
import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402  (registers tasks, incl. ReplayPickItUpOneEnv)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


# ----------------------------------------------------------------------------
# JSON + base64 wire codec (numpy-version-agnostic; matches the server)
# ----------------------------------------------------------------------------
def _encode(obj):
    if isinstance(obj, np.ndarray):
        return {
            "__nd__": True,
            "b": base64.b64encode(obj.tobytes()).decode("ascii"),
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
        }
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(v) for v in obj]
    return obj


def _decode(obj):
    if isinstance(obj, dict):
        if obj.get("__nd__"):
            raw = base64.b64decode(obj["b"])
            return np.frombuffer(raw, dtype=obj["dtype"]).reshape(obj["shape"]).copy()
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    return obj


class RemotePolicy:
    """HTTP client for LeRobot's ``eval_policy_http_server``."""

    def __init__(self, server_url: str, timeout: float = 120.0):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(_encode(payload)).encode("utf-8")
        req = urllib.request.Request(  # nosec B310
            f"{self.server_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # nosec B310
            result = _decode(json.loads(resp.read().decode("utf-8")))
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(f"Policy server error on {path}: {result['error']}")
        return result

    def health(self) -> dict:
        req = urllib.request.Request(f"{self.server_url}/health", method="GET")  # nosec B310
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # nosec B310
            return _decode(json.loads(resp.read().decode("utf-8")))

    def load(self, policy_type, pretrained_path, device, rename_map=None):
        self._post(
            "/load",
            {
                "policy_type": policy_type,
                "pretrained_path": pretrained_path,
                "device": device,
                "rename_map": rename_map or {},
            },
        )

    def reset(self):
        self._post("/reset", {})

    def act(self, raw_observation: dict) -> np.ndarray:
        return self._post("/predict", raw_observation)["action"]


def obs_to_payload(obs: dict, task_text: str) -> dict:
    """Map ReplayPickItUpOneEnv's eval obs to the server's raw-obs contract.

    The env already uses LeRobot keys; we send the proprio state as ``agent_pos``
    and the two images as a ``pixels`` dict so the server's
    ``preprocess_observation`` produces ``observation.state`` +
    ``observation.images.image`` / ``.image2``. A leading batch dim is added.
    """
    state = np.asarray(obs["observation.state"], dtype=np.float32)[None]  # [1, 8]
    payload = {"agent_pos": state, "task": [task_text]}
    pixels = {}
    for key in ("observation.images.image", "observation.images.image2"):
        if key in obs:
            short = key.rsplit(".", 1)[-1]  # "image" / "image2"
            pixels[short] = np.asarray(obs[key])[None]  # [1, H, W, 3] uint8
    if pixels:
        payload["pixels"] = pixels
    return payload


def main():
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=1)
    env_cfg.sim.device = device
    env_cfg.record_cameras = True  # produce observation.images.image / image2
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.REPLAY_CONTROLLER = args_cli.controller
    if args_cli.controller == "joint_pd":
        env.PD_KP, env.PD_KD = args_cli.kp, args_cli.kd
    env.reset()

    remote = None
    if not args_cli.self_test:
        remote = RemotePolicy(args_cli.server_url, timeout=args_cli.request_timeout)
        if args_cli.policy_type and args_cli.pretrained_path:
            rename_map = json.loads(args_cli.rename_map) if args_cli.rename_map else None
            print(f"[client] requesting server load: {args_cli.policy_type} <- {args_cli.pretrained_path}")
            remote.load(args_cli.policy_type, args_cli.pretrained_path, args_cli.policy_device, rename_map)
        if not remote.health().get("loaded"):
            raise RuntimeError("Policy server has no policy loaded. Pass --policy_type/--pretrained_path.")

    # ------------------------------------------------------------------
    # Collection mode: roll out + record a LeRobot dataset. Reuses the env's
    # run_policy_rollout_all (which owns the record/split/manifest logic); the
    # policy is the SAME remote/self-test callable, reset between episodes at t==0.
    # ------------------------------------------------------------------
    if args_cli.record_dir:
        def policy_fn(obs, ep, t):
            if args_cli.self_test:
                return env.demo_action_sequence(ep)[t]
            if t == 0:
                remote.reset()  # clear the server's action queue per episode (ACT etc.)
            return remote.act(obs_to_payload(obs, args_cli.task_text))[0]  # [8]

        results = env.run_policy_rollout_all(
            policy_fn, log_dir=args_cli.record_dir, num_episodes=args_cli.episodes,
            record_dir=args_cli.record_dir, horizon=args_cli.horizon,
        )
        n_success = sum(bool(r["success"]) for r in results)
        rate = 100.0 * n_success / max(args_cli.episodes, 1)
        print(f"[client] DONE (recorded -> {args_cli.record_dir}) | "
              f"success {n_success}/{args_cli.episodes} ({rate:.1f}%)", flush=True)
        if args_cli.json_out:
            with open(args_cli.json_out, "w") as f:
                json.dump(
                    {"n_episodes": args_cli.episodes, "n_success": n_success,
                     "success_rate": rate / 100.0,
                     "episodes": [{"episode": r["episode"], "success": bool(r["success"]),
                                   "grasped": bool(r.get("grasped", False)),
                                   "object_z": r.get("object_z", float("nan"))} for r in results]},
                    f, indent=2,
                )
            print(f"[client] wrote summary to {args_cli.json_out}", flush=True)
        env.close()
        torch.cuda.empty_cache()
        env.sim.stop()
        env.sim.clear()
        return

    def get_action(obs, demo_seq, t):
        if args_cli.self_test:
            return demo_seq[t]
        return remote.act(obs_to_payload(obs, args_cli.task_text))[0]  # [8]

    n_success = 0
    results = []
    for ep in range(args_cli.episodes):
        demo_seq = env.demo_action_sequence(ep)  # [T, 8]
        horizon = demo_seq.shape[0]
        obs = env.eval_reset(ep)
        if remote is not None:
            remote.reset()
        success = False
        # info["grasped"] is the *instantaneous* grasp flag; by the time success
        # fires the object is in the basket and the gripper has released, so it
        # reads False. Track whether a grasp happened at any point this episode.
        grasped_ever = False
        info = {"success": False, "grasped": False, "object_z": float("nan")}
        for t in range(horizon):
            action = get_action(obs, demo_seq, t)
            obs, terminated, truncated, info = env.eval_step(action)
            success = success or info["success"]
            grasped_ever = grasped_ever or bool(info["grasped"])
            if info["success"]:
                break
        n_success += int(success)
        results.append(
            {"episode": ep, "success": bool(success), "grasped": grasped_ever, "object_z": info["object_z"]}
        )
        print(
            f"[client] ep {ep:>3}: success={success}  grasped(ever)={grasped_ever}  "
            f"obj_z={info['object_z']:.3f}",
            flush=True,
        )

    rate = 100.0 * n_success / max(args_cli.episodes, 1)
    print(f"[client] DONE | success {n_success}/{args_cli.episodes} ({rate:.1f}%)", flush=True)
    if args_cli.json_out:
        with open(args_cli.json_out, "w") as f:
            json.dump(
                {"n_episodes": args_cli.episodes, "n_success": n_success, "success_rate": rate / 100.0,
                 "episodes": results},
                f,
                indent=2,
            )
        print(f"[client] wrote summary to {args_cli.json_out}", flush=True)

    env.close()
    torch.cuda.empty_cache()
    env.sim.stop()
    env.sim.clear()


if __name__ == "__main__":
    main()
    simulation_app.close()
