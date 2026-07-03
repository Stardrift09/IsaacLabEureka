"""Collect GRASP-MOMENT initial states (robot + object + basket) from an rsl_rl policy.

Derived from ``scripts/collect_rsl_rl_parallel.py`` -- same skeleton: launch IsaacSim,
build a MANY-env task, load a trained rsl_rl checkpoint, and roll the policy out across all
envs in lockstep. The difference is what we do with the rollout:

  * It records NO dataset -- no cameras, no mp4s, no ProcessPool encoding, no manifest.
    The rsl_rl policy is STATE-based (proprio + object/basket pose), so grasp collection
    needs no images at all -> ``enable_cameras=False`` and the lean training env (default
    ``PickItUpClean``) instead of the camera-laden ``BatchEvalPickItUpClean``.
  * It watches the per-env ``grasped`` signal. The FIRST step an env's episode reports a
    grasp, it snapshots that env's robot state (joint qpos + base pose), object pose and
    basket pose (env-local frame). One snapshot per episode; the flag clears on reset.
  * It repeats across episodes/envs until ``--num_states`` (default 1000) snapshots are
    collected. Envs reset (and re-randomize their init) automatically inside ``env.step``.

The result -- object/basket/robot poses at the instant of grasp -- is exactly the
distribution a "start already grasped in the air" task needs to reset into. Saved as an
``.npz`` (stacked arrays) plus a sidecar ``.json`` (joint names, task, grasp rate).

Init diversity: by default the env's ring randomization (``randomize_init``) gives a
different object/basket placement each reset. Pass ``--grid_file`` (make_init_grid.py) for
deterministic even coverage instead. Run in conda ``eureka``:

    ~/miniconda3/envs/eureka/bin/python -u scripts/collect_init_states_grasped.py \
        --checkpoint <.../model_*.pt> --num_envs 64 --num_states 1000 \
        --out logs/init_states_grasped.npz --headless
"""

import argparse
import os

from isaaclab_eureka.utils import get_freest_gpu

# Per-snapshot fields captured at the grasp moment (env-local frame; quats are world-frame).
_STATE_KEYS = ("robot_qpos", "robot_base_pos", "robot_base_quat",
               "object_pos", "object_quat", "basket_pos", "basket_quat")


def capture_states(u, env_ids):
    """Snapshot robot/object/basket state for the given env ids (a torch index tensor).

    Returns a dict of numpy arrays [m, dim]. World->env-local is a translation by the env
    origin (orientations are frame-invariant)."""
    import numpy as np  # local: keep module import-light before the app launches

    origin = u.scene.env_origins[env_ids]

    def _np(x):
        return x.detach().cpu().numpy().astype(np.float32)

    return {
        "robot_qpos": _np(u._robot.data.joint_pos[env_ids]),
        "robot_base_pos": _np(u._robot.data.root_pos_w[env_ids] - origin),
        "robot_base_quat": _np(u._robot.data.root_quat_w[env_ids]),
        "object_pos": _np(u.target_object.data.root_pos_w[env_ids] - origin),
        "object_quat": _np(u.target_object.data.root_quat_w[env_ids]),
        "basket_pos": _np(u.target_site.data.root_pos_w[env_ids] - origin),
        "basket_quat": _np(u.target_site.data.root_quat_w[env_ids]),
    }


def save_states(out_path, chunks, n_keep, meta):
    """Concatenate per-step capture chunks, trim to n_keep, write <out>.npz + <out>.json."""
    import json

    import numpy as np

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    if chunks:
        arrays = {k: np.concatenate([c[k] for c in chunks], axis=0)[:n_keep] for k in _STATE_KEYS}
    else:
        arrays = {}
    np.savez(out_path, **arrays)
    with open(os.path.splitext(out_path)[0] + ".json", "w") as f:
        json.dump({**meta, "n_states": int(next(iter(arrays.values())).shape[0]) if arrays else 0,
                   "arrays": {k: list(v.shape) for k, v in arrays.items()}}, f, indent=2)


def main(args_cli):
    from isaaclab.app import AppLauncher

    device = args_cli.device
    if device == "cuda":
        device = f"cuda:{get_freest_gpu()}"
    # No cameras: the rsl_rl policy + grasp signal are state-based -> big speedup vs the parent.
    app_launcher = AppLauncher(headless=args_cli.headless, enable_cameras=False, device=device)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import numpy as np
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunner

    task = args_cli.task
    env_cfg = parse_env_cfg(task, device=device, num_envs=args_cli.num_envs)
    env_cfg.sim.device = device
    # Full init spread from step 0 (no reverse-curriculum ramp) so the collected grasp states
    # cover the whole object/basket distribution, not just the easy center.
    if hasattr(env_cfg, "curriculum_enabled"):
        env_cfg.curriculum_enabled = False
    if hasattr(env_cfg, "randomize_init"):
        env_cfg.randomize_init = True
    # Per-object reset range. The ring fields are DIAMETERS; the reset displaces the object
    # AND the basket each within a disk of radius outer_diam/2 of nominal. So a 0.03 m range
    # per object == outer diameter 0.06. (The cfg default is 0.16 -> radius 0.08, much wider;
    # the object_pos_noise/basket_pos_noise=0.03 fields are NOT used by this ring path.)
    if hasattr(env_cfg, "init_ring_outer_diam"):
        env_cfg.init_ring_outer_diam = 2.0 * args_cli.init_range
        env_cfg.init_ring_inner_diam = 0.0
    # Deterministic even coverage if a grid file is given; else the env's ring randomization.
    if args_cli.grid_file and hasattr(env_cfg, "init_grid_file"):
        env_cfg.init_grid_file = args_cli.grid_file
    # Shorten episodes so envs cycle (and re-randomize) faster once they have grasped. Set
    # BEFORE the env is built. 0 -> keep the env default length.
    if args_cli.max_frames and args_cli.max_frames > 0:
        control_dt = env_cfg.sim.dt * env_cfg.decimation
        env_cfg.episode_length_s = args_cli.max_frames * control_dt

    env = gym.make(task, cfg=env_cfg)
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = device
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(args_cli.checkpoint, map_location=device)
    policy = runner.get_inference_policy(device=device)
    u = env.unwrapped
    n_env = u.num_envs

    print(f"[collect_init] task={task} num_envs={n_env} target={args_cli.num_states} "
          f"grid={'yes' if (args_cli.grid_file and hasattr(env_cfg, 'init_grid_file')) else 'no (ring)'} "
          f"-> {args_cli.out}", flush=True)

    obs, _ = env.reset()

    chunks = []                                   # list of capture dicts (each [m, dim])
    n_collected = 0
    n_episodes_done = 0
    captured_this_ep = np.zeros(n_env, dtype=bool)  # one snapshot per env per episode
    last_ckpt = 0

    while n_collected < args_cli.num_states:
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

        grasped = u.grasped.detach().cpu().numpy().astype(bool)      # [n_env] post-step
        dones_np = dones.detach().cpu().numpy().astype(bool)

        # Envs that just grasped (first time this episode) -> snapshot the grasp moment.
        newly = np.nonzero(grasped & ~captured_this_ep)[0]
        if newly.size:
            idx = torch.as_tensor(newly, dtype=torch.long, device=u.device)
            chunks.append(capture_states(u, idx))
            captured_this_ep[newly] = True
            n_collected += int(newly.size)

        # Finished envs auto-reset inside env.step (fresh randomized init) -> allow a new
        # snapshot next episode. Count them for the grasp-rate stat.
        if dones_np.any():
            captured_this_ep[dones_np] = False
            n_episodes_done += int(dones_np.sum())

        if newly.size:
            print(f"[collect_init] +{newly.size} (total {n_collected}/{args_cli.num_states}) "
                  f"| episodes_done={n_episodes_done}", flush=True)
        if args_cli.save_every > 0 and n_collected - last_ckpt >= args_cli.save_every:
            save_states(args_cli.out, chunks, args_cli.num_states, _meta(u, n_episodes_done))
            last_ckpt = n_collected
            print(f"[collect_init] checkpoint saved ({min(n_collected, args_cli.num_states)} states) "
                  f"-> {args_cli.out}", flush=True)

    save_states(args_cli.out, chunks, args_cli.num_states, _meta(u, n_episodes_done))
    rate = 100.0 * args_cli.num_states / max(n_episodes_done, 1)
    print(f"[collect_init] DONE | {args_cli.num_states} grasp states from ~{n_episodes_done} "
          f"finished episodes (grasp/episode ~{rate:.0f}%) -> {args_cli.out}", flush=True)

    env.close()
    torch.cuda.empty_cache()
    simulation_app.close()


def _meta(u, n_episodes_done):
    return {
        "task": args_cli.task,
        "target_object": getattr(u, "target_object_name", None),
        "target_site": getattr(u, "target_site_name", None),
        "joint_names": list(getattr(u._robot, "joint_names", []) or []),
        "frame": "env-local (world minus env origin); quats are world-frame",
        "checkpoint": args_cli.checkpoint,
        "n_episodes_done": int(n_episodes_done),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect grasp-moment initial states (robot/object/basket) from an rsl_rl policy."
    )
    parser.add_argument("--task", type=str, default="PickItUpClean",
                        help="Many-env training env with a grasp signal (PickItUpClean family). The "
                             "checkpoint's policy obs must match.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Absolute path to model_*.pt.")
    parser.add_argument("--num_states", type=int, default=1000, help="Grasp-moment snapshots to collect.")
    parser.add_argument("--num_envs", type=int, default=64, help="Parallel envs.")
    parser.add_argument("--init_range", type=float, default=0.03,
                        help="Per-object reset RADIUS (m) of the init disk, applied to BOTH the object "
                             "and the basket (set as ring outer diameter = 2*init_range). Default 0.03 "
                             "(matches the in-distribution band). Ignored when --grid_file is set.")
    parser.add_argument("--grid_file", type=str, default=None,
                        help="Optional init grid npz (make_init_grid.py) for deterministic even-coverage "
                             "inits. Omit -> the env's ring randomization.")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Truncate each episode after N control steps so envs cycle faster once "
                             "grasped (0 = keep the env's default episode length). Keep > the policy's "
                             "typical grasp time, else episodes truncate before grasping.")
    parser.add_argument("--out", type=str, default="logs/init_states_grasped.npz",
                        help="Output .npz path (a sidecar .json with metadata is written next to it).")
    parser.add_argument("--save_every", type=int, default=200,
                        help="Checkpoint the collected states to --out every N new snapshots (0=off).")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--headless", action="store_true", default=True)
    args_cli = parser.parse_args()
    main(args_cli)
