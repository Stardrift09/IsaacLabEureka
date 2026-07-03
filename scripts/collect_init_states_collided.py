"""Collect TASK-ACCOMPLISHED states (robot + object + basket + can, with velocities).

Analogous to ``scripts/collect_init_states_grasped.py``, but for the *collide* task: it rolls
out the trained COLLIDE policy on ``PickItUpCollideClean`` (object starts already grasped in
the air, from the grasp-moment init file) and snapshots each env the FIRST step the collide
task is accomplished -- i.e. the can has been displaced past ``--collide_threshold`` AND the
object is still grasped. These snapshots are the init distribution the NEXT phase (placing the
object in the basket) should reset into, which is why directly switching policies after the
collision doesn't work: the place policy must START from the post-collision *dynamic* state.

So, unlike the grasped collector, this one saves NOT ONLY positions but also VELOCITIES
(robot joint velocities + every body's linear & angular velocity), because right after the
knock the arm, the held object and the can all carry momentum.

To capture the success state BEFORE the env auto-resets on it, the env's collision
termination is disabled (``collision_displacement_threshold`` set huge) and success is
detected here instead. Run in conda ``eureka``:

    ~/miniconda3/envs/eureka/bin/python -u scripts/collect_init_states_collided.py \
        --checkpoint <.../collide model_*.pt> --num_envs 64 --num_states 1000 \
        --out logs/init_states_collided.npz --headless
"""

import argparse
import os

from isaaclab_eureka.utils import get_freest_gpu

# Per-snapshot fields captured at the task-accomplished moment. Positions are env-local
# (world minus env origin); quaternions and ALL velocities are world-frame.
_STATE_KEYS = (
    "robot_qpos", "robot_qvel",
    "robot_base_pos", "robot_base_quat", "robot_base_lin_vel", "robot_base_ang_vel",
    "object_pos", "object_quat", "object_lin_vel", "object_ang_vel",
    "basket_pos", "basket_quat", "basket_lin_vel", "basket_ang_vel",
    "collision_pos", "collision_quat", "collision_lin_vel", "collision_ang_vel",
)


def capture_states(u, env_ids):
    """Snapshot full state (pose + velocity) of robot/object/basket/can for the given env ids.

    Returns a dict of numpy arrays [m, dim]. Positions are env-local; quats + velocities are
    world-frame (a pure env-origin translation leaves orientations and velocities unchanged)."""
    import numpy as np  # local: keep module import-light before the app launches

    origin = u.scene.env_origins[env_ids]
    rd = u._robot.data
    od = u.target_object.data
    bd = u.target_site.data
    cd = u.collision_object.data

    def _np(x):
        return x.detach().cpu().numpy().astype(np.float32)

    return {
        "robot_qpos": _np(rd.joint_pos[env_ids]),
        "robot_qvel": _np(rd.joint_vel[env_ids]),
        "robot_base_pos": _np(rd.root_pos_w[env_ids] - origin),
        "robot_base_quat": _np(rd.root_quat_w[env_ids]),
        "robot_base_lin_vel": _np(rd.root_lin_vel_w[env_ids]),
        "robot_base_ang_vel": _np(rd.root_ang_vel_w[env_ids]),
        "object_pos": _np(od.root_pos_w[env_ids] - origin),
        "object_quat": _np(od.root_quat_w[env_ids]),
        "object_lin_vel": _np(od.root_lin_vel_w[env_ids]),
        "object_ang_vel": _np(od.root_ang_vel_w[env_ids]),
        "basket_pos": _np(bd.root_pos_w[env_ids] - origin),
        "basket_quat": _np(bd.root_quat_w[env_ids]),
        "basket_lin_vel": _np(bd.root_lin_vel_w[env_ids]),
        "basket_ang_vel": _np(bd.root_ang_vel_w[env_ids]),
        "collision_pos": _np(cd.root_pos_w[env_ids] - origin),
        "collision_quat": _np(cd.root_quat_w[env_ids]),
        "collision_lin_vel": _np(cd.root_lin_vel_w[env_ids]),
        "collision_ang_vel": _np(cd.root_ang_vel_w[env_ids]),
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
    # No cameras: the rsl_rl policy + collision/grasp signals are state-based.
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
    # Keep the grasp-moment file init (object starts grasped in the air -- how the collide
    # policy was trained). Optional override.
    if args_cli.init_states_file is not None and hasattr(env_cfg, "init_states_file"):
        env_cfg.init_states_file = args_cli.init_states_file
    # Do NOT let the env auto-terminate on the collision -- otherwise the success env resets
    # before we can snapshot it. Disable env-side termination; detect success here instead.
    if hasattr(env_cfg, "collision_displacement_threshold"):
        env_cfg.collision_displacement_threshold = 1.0e9
    # Shorten episodes so envs cycle faster after they succeed (0 = env default length). Keep
    # this above the collide policy's typical time-to-collision or episodes truncate too early.
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

    print(f"[collect_collided] task={task} num_envs={n_env} target={args_cli.num_states} "
          f"collide_threshold={args_cli.collide_threshold} -> {args_cli.out}", flush=True)

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

        # Task accomplished = can displaced past the threshold AND object still grasped.
        can_disp = torch.norm(u.collision_obj_pos - u.collision_object_init_pos, dim=-1)
        accomplished = ((can_disp > args_cli.collide_threshold) & u.grasped).detach().cpu().numpy()
        dones_np = dones.detach().cpu().numpy().astype(bool)

        newly = np.nonzero(accomplished & ~captured_this_ep)[0]
        if newly.size:
            idx = torch.as_tensor(newly, dtype=torch.long, device=u.device)
            chunks.append(capture_states(u, idx))
            captured_this_ep[newly] = True
            n_collected += int(newly.size)

        # Finished envs auto-reset inside env.step (fresh grasped-air init) -> allow a new
        # snapshot next episode.
        if dones_np.any():
            captured_this_ep[dones_np] = False
            n_episodes_done += int(dones_np.sum())

        if newly.size:
            print(f"[collect_collided] +{newly.size} (total {n_collected}/{args_cli.num_states}) "
                  f"| episodes_done={n_episodes_done}", flush=True)
        if args_cli.save_every > 0 and n_collected - last_ckpt >= args_cli.save_every:
            save_states(args_cli.out, chunks, args_cli.num_states, _meta(u, n_episodes_done))
            last_ckpt = n_collected
            print(f"[collect_collided] checkpoint saved ({min(n_collected, args_cli.num_states)} states) "
                  f"-> {args_cli.out}", flush=True)

    save_states(args_cli.out, chunks, args_cli.num_states, _meta(u, n_episodes_done))
    rate = 100.0 * args_cli.num_states / max(n_episodes_done, 1)
    print(f"[collect_collided] DONE | {args_cli.num_states} accomplished states from ~{n_episodes_done} "
          f"finished episodes (success/episode ~{rate:.0f}%) -> {args_cli.out}", flush=True)

    env.close()
    torch.cuda.empty_cache()
    simulation_app.close()


def _meta(u, n_episodes_done):
    return {
        "task": args_cli.task,
        "target_object": getattr(u, "target_object_name", None),
        "target_site": getattr(u, "target_site_name", None),
        "collision_object": getattr(u, "collision_object_name", None),
        "joint_names": list(getattr(u._robot, "joint_names", []) or []),
        "frame": "positions env-local (world minus env origin); quats + velocities world-frame",
        "collide_threshold": args_cli.collide_threshold,
        "checkpoint": args_cli.checkpoint,
        "n_episodes_done": int(n_episodes_done),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect task-accomplished states (pose + velocity) from the collide policy."
    )
    parser.add_argument("--task", type=str, default="PickItUpCollideClean",
                        help="Collide env exposing grasped + collision signals. Checkpoint obs must match.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Absolute path to the COLLIDE policy model_*.pt.")
    parser.add_argument("--num_states", type=int, default=1000, help="Accomplished-moment snapshots to collect.")
    parser.add_argument("--num_envs", type=int, default=64, help="Parallel envs.")
    parser.add_argument("--collide_threshold", type=float, default=0.1,
                        help="Can displacement (m) from spawn that counts as the task accomplished.")
    parser.add_argument("--init_states_file", type=str, default=None,
                        help="Override the grasp-moment init file (default: the env cfg's value).")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Truncate each episode after N control steps so envs cycle faster once they "
                             "succeed (0 = env default length). Keep > the policy's time-to-collision.")
    parser.add_argument("--out", type=str, default="logs/init_states_collided.npz",
                        help="Output .npz path (a sidecar .json with metadata is written next to it).")
    parser.add_argument("--save_every", type=int, default=200,
                        help="Checkpoint the collected states to --out every N new snapshots (0=off).")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--headless", action="store_true", default=True)
    args_cli = parser.parse_args()
    main(args_cli)
