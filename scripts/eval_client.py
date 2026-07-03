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
physics (joint PD), exactly matching how the dataset was made. For a policy this
uses the env's NATIVE actuator gains (the rsl_rl->BC dataset is collected with
them); --self_test uses 3000/260; explicit --kp/--kd override either.

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
parser.add_argument("--task", type=str, default="EvalPickItUpOneEnv",
                    help="Registered env id. Use OodEvalPickItUpOneEnv (or pass "
                         "--object_pos_noise) to test out-of-distribution target-object inits.")
parser.add_argument("--episodes", type=int, default=5)
# Init-state randomization overrides (applied to the env cfg before make; None = keep
# the cfg's value). --object_pos_noise is the OOD knob: Gaussian std (m) of XY jitter
# on the TARGET OBJECT at reset. arm/basket provided for completeness.
parser.add_argument("--object_pos_noise", type=float, default=None,
                    help="Override target-object XY reset noise (m). On normal tasks this is the "
                         "Gaussian std (train/eval=0.03). On OodEvalPickItUpOneEnv it is the OUTER "
                         "radius of the uniform OOD annulus (default 0.06). Higher = more OOD.")
parser.add_argument("--object_pos_floor", type=float, default=None,
                    help="OOD only (OodEvalPickItUpOneEnv): inner radius / min target-object "
                         "displacement (m) of the uniform annulus. Guarantees no episode lands in "
                         "the in-distribution core. Maps to cfg.object_pos_noise_floor; ignored elsewhere.")
parser.add_argument("--init_ring_outer_diam", type=float, default=None,
                    help="Clean envs only (PickItUpClean family): override the init ring OUTER "
                         "diameter (m). Object/basket reset uniformly in the disk of radius "
                         "diam/2 around nominal. Collection used 0.24 (radius 0.12); set smaller "
                         "(e.g. 0.06) to eval on a narrower in-distribution band. Ignored if absent.")
parser.add_argument("--arm_joint_noise", type=float, default=None,
                    help="Override arm-joint reset noise std (rad).")
parser.add_argument("--basket_pos_noise", type=float, default=None,
                    help="Override basket XY reset noise std (m).")
parser.add_argument("--sim_dt", type=float, default=None,
                    help="Override physics sim.dt to match the env the dataset was recorded in "
                         "(MimicGen pick-basket = 0.01; EvalPickItUpClean default = 1/180).")
parser.add_argument("--max_steps", type=int, default=None,
                    help="Episode truncation budget in CONTROL steps (sets episode_length_s). The "
                         "default (519, from the LIBERO demos) clips the longer MimicGen trajectories "
                         "(up to ~545). Use with a matching --horizon. e.g. --max_steps 600 --horizon 600.")
parser.add_argument("--controller", type=str, default="joint_pd", choices=["joint_pd", "osc"])
parser.add_argument("--kp", type=float, default=None,
                    help="joint_pd arm stiffness. Default: env-native gains for a policy (matches how "
                         "the rsl_rl->BC dataset was collected); 3000 for --self_test.")
parser.add_argument("--kd", type=float, default=None,
                    help="joint_pd arm damping. Default: env-native gains for a policy; 260 for --self_test.")
parser.add_argument("--eval_fps", type=int, default=60,
                    help="Control rate for eval_step (action hold = round(120/eval_fps) physics "
                         "steps). Must MATCH the dataset fps. Default: 60 for a policy "
                         "(60 fps rsl_rl->BC dataset), 20 for --self_test (LIBERO demos are 20 fps).")
parser.add_argument("--task_text", type=str, default="pick up the alphabet_soup and put it in the basket")
parser.add_argument("--json_out", type=str, default=None)
parser.add_argument("--save_init_states", type=str, default=None,
                    help="If set, write ONE JSON file with each episode's initial state "
                         "(object/basket/robot pose; for OodEvalPickItUpOneEnv also the OOD "
                         "displacement vector/radius) AND its success flag. For checking real "
                         "OOD performance vs init distance. Only the non-record eval loop.")
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
parser.add_argument("--policy_type", type=str, default="act")
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
args_cli.headless = False

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
    # Physics dt override: match the env the dataset was RECORDED in. The MimicGen pick-basket
    # data is recorded at sim.dt=0.01; EvalPickItUpClean defaults to 1/180, so the same joint
    # targets integrate to slightly different motion. Set --sim_dt 0.01 to remove that mismatch.
    # (With --eval_fps 20 the control period stays 0.05s: round((1/20)/0.01)=5 substeps.)
    if args_cli.sim_dt is not None:
        env_cfg.sim.dt = args_cli.sim_dt
        print(f"[client] sim.dt overridden -> {env_cfg.sim.dt}")
    # Episode truncation budget: max_episode_length = episode_length_s / (sim.dt * decimation).
    # Set it from --max_steps (in control steps) so the longer MimicGen rollouts aren't cut short.
    # Computed AFTER the sim_dt override so it uses the final dt.
    if args_cli.max_steps is not None:
        env_cfg.episode_length_s = args_cli.max_steps * env_cfg.sim.dt * env_cfg.decimation
        print(f"[client] max_steps={args_cli.max_steps} -> episode_length_s={env_cfg.episode_length_s:.3f}")
    # Init-state randomization overrides (OOD testing): None -> keep the cfg value.
    for field in ("object_pos_noise", "arm_joint_noise", "basket_pos_noise"):
        val = getattr(args_cli, field)
        if val is not None:
            setattr(env_cfg, field, val)
    # Clean-env init ring width (PickItUpClean family). Lets you sweep eval init spread to
    # separate "policy-limited" from "distribution-width-limited" success.
    if args_cli.init_ring_outer_diam is not None:
        if hasattr(env_cfg, "init_ring_outer_diam"):
            env_cfg.init_ring_outer_diam = args_cli.init_ring_outer_diam
        else:
            print(f"[client] WARNING: --init_ring_outer_diam ignored: task {args_cli.task} "
                  f"has no init_ring_outer_diam (use a PickItUpClean-family task).")
    # OOD min-displacement floor: only OodEvalPickItUpOneEnvCfg has this field.
    if args_cli.object_pos_floor is not None:
        if hasattr(env_cfg, "object_pos_noise_floor"):
            env_cfg.object_pos_noise_floor = args_cli.object_pos_floor
        else:
            print(f"[client] WARNING: --object_pos_floor ignored: task {args_cli.task} "
                  f"has no object_pos_noise_floor (use --task OodEvalPickItUpOneEnv).")
    floor = getattr(env_cfg, "object_pos_noise_floor", None)
    print(f"[client] init noise: object_pos_noise={env_cfg.object_pos_noise} "
          f"arm_joint_noise={env_cfg.arm_joint_noise} basket_pos_noise={env_cfg.basket_pos_noise} "
          f"object_pos_floor={floor} (randomize_init={env_cfg.randomize_init})")
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.REPLAY_CONTROLLER = args_cli.controller
    if args_cli.controller == "joint_pd":
        # Gains: leave PD_KP/PD_KD = None to keep the env's NATIVE ImplicitActuator gains
        # (shoulder 1500/200, forearm 1200/180) -- this is what collect_rsl_rl.py used to
        # make the dataset, so a policy MUST eval at the same gains to reproduce the actions.
        # self_test (LIBERO demo replay) keeps the 3000/260 sweep winner. Explicit --kp/--kd
        # always override.
        kp = args_cli.kp if args_cli.kp is not None else (3000.0 if args_cli.self_test else None)
        kd = args_cli.kd if args_cli.kd is not None else (260.0 if args_cli.self_test else None)
        if kp is not None:
            env.PD_KP = kp
        if kd is not None:
            env.PD_KD = kd
        print(f"[client] joint_pd gains: "
              f"{'env-native (1500/200, 1200/180)' if kp is None and kd is None else f'kp={kp} kd={kd}'}")
    # eval control rate: must match the dataset fps. 60 fps for a policy (the
    # rsl_rl->BC dataset is collected at 60 fps via collect_rsl_rl --record_every 1);
    # 20 fps for --self_test (LIBERO demo actions are 20 fps). steps = round(120/fps).
    eval_fps = args_cli.eval_fps if args_cli.eval_fps is not None else (20 if args_cli.self_test else 60)
    steps = max(1, round((1.0 / eval_fps) / env.cfg.sim.dt))
    env.EVAL_STEPS_PER_FRAME = steps
    print(f"[client] eval_fps={eval_fps} -> EVAL_STEPS_PER_FRAME={steps} (sim.dt={env.cfg.sim.dt})")
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
        if args_cli.save_init_states:
            print("[client] WARNING: --save_init_states is only collected in the non-record "
                  "eval loop; ignored together with --record_dir.", flush=True)

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

    # Horizon/reset source: for a POLICY with randomized/OOD inits the episode index
    # is NOT a demo index (there are only len(env.episodes) demos), so DON'T tie the
    # horizon or the reset to a per-episode demo -- that IndexErrors once --episodes
    # exceeds the demo count. Use --horizon, else the longest demo (same rule as the
    # collection path run_policy_rollout_all). --self_test replays a real demo, so it
    # still indexes demos, wrapping with modulo if --episodes exceeds the count.
    n_demos = len(env.episodes)
    policy_horizon = (args_cli.horizon if args_cli.horizon is not None
                      else max(env.demo_action_sequence(i).shape[0] for i in range(n_demos)))

    n_success = 0
    results = []
    for ep in range(args_cli.episodes):
        if args_cli.self_test:
            demo_seq = env.demo_action_sequence(ep % n_demos)  # [T, 8] demo actions
            horizon = demo_seq.shape[0]
            reset_idx = ep % n_demos
        else:
            demo_seq = None                 # policy path never reads demo_seq
            horizon = policy_horizon
            reset_idx = ep                  # randomized/OOD reset ignores the index
        obs = env.eval_reset(reset_idx)
        # init-state snapshot (right after reset, before any action) for OOD analysis
        init_state = (env.eval_init_state()
                      if args_cli.save_init_states and hasattr(env, "eval_init_state") else None)
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

            # # gripper command from policy (last dim; >0 = open) vs actual finger pos
            # grip_cmd = float(np.asarray(action).reshape(-1)[-1])
            # grip_act = float(np.asarray(obs["observation.state"]).reshape(-1)[7])
            # print(
            #     f"[client] ep {ep:>3} t {t:>3}: grip_cmd={grip_cmd:+.3f}  "
            #     f"grip_act(finger_joint1)={grip_act:.4f}",
            #     flush=True,
            # )
            obs, terminated, truncated, info = env.eval_step(action)
            success = success or info["success"]
            grasped_ever = grasped_ever or bool(info["grasped"])
            if info["success"]:
                break
        n_success += int(success)
        rec = {"episode": ep, "success": bool(success), "grasped": grasped_ever, "object_z": info["object_z"]}
        if init_state is not None:
            rec["init_state"] = init_state
        results.append(rec)
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

    if args_cli.save_init_states:
        with open(args_cli.save_init_states, "w") as f:
            json.dump(
                {"task": args_cli.task, "n_episodes": args_cli.episodes,
                 "n_success": n_success, "success_rate": rate / 100.0,
                 "object_pos_noise": getattr(env.cfg, "object_pos_noise", None),
                 "object_pos_noise_floor": getattr(env.cfg, "object_pos_noise_floor", None),
                 "episodes": results},   # each entry carries init_state + success
                f, indent=2,
            )
        print(f"[client] wrote init states + success to {args_cli.save_init_states}", flush=True)

    env.close()
    torch.cuda.empty_cache()
    env.sim.stop()
    env.sim.clear()


if __name__ == "__main__":
    main()
    simulation_app.close()
