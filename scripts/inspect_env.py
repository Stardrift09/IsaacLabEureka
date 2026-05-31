"""Inspect an Isaac Lab env with zero actions.

Prints per-step state: target object pos, drawer interior pos, EEF pos,
stage, grasped, inside_drawer, reward. No action applied.

Usage:
    python scripts/inspect_env.py [--task TestPlaceBasketInDrawer] [--num_envs 4] [--steps 300] [--headless]
"""

import argparse

from isaaclab_eureka.utils import get_freest_gpu


def main(args_cli):
    from isaaclab.app import AppLauncher

    device = args_cli.device
    if device == "cuda":
        device_id = get_freest_gpu()
        device = f"cuda:{device_id}"

    app_launcher = AppLauncher(headless=args_cli.headless, device=device)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch
    import isaaclab_tasks  # noqa: F401
    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab_tasks.utils import parse_env_cfg

    task = args_cli.task
    env_cfg: DirectRLEnvCfg = parse_env_cfg(task)
    env_cfg.sim.device = device
    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make(task, cfg=env_cfg)
    env_unwrapped = env.unwrapped

    obs, _ = env.reset()
    zero_action = torch.zeros(
        (args_cli.num_envs, env_unwrapped.cfg.action_space),
        device=device,
    )

    print(f"\n{'='*70}")
    print(f"Task: {task}  |  envs: {args_cli.num_envs}  |  steps: {args_cli.steps}")
    print(f"obs shape: {obs['policy'].shape}  |  action dim: {env_unwrapped.cfg.action_space}")
    print(f"{'='*70}\n")

    for step in range(args_cli.steps):
        with torch.inference_mode():
            obs, reward, terminated, truncated, info = env.step(zero_action)

        if step % args_cli.print_every == 0:
            e = env_unwrapped

            cc_pos   = e.target_object.data.root_pos_w[0]   # cream_cheese world pos
            drw_pos  = e.drawer_interior_pos_w[0]            # drawer interior world pos
            eef_pos  = e.robot_grasp_pos[0]
            stage    = e.stage[0].int().tolist()
            grasped  = e.grasped[0].item()
            inside   = e.inside_site[0].item()
            rew0     = reward[0].item()

            cc_rel   = cc_pos  - e.scene.env_origins[0]     # env-local coords
            drw_rel  = drw_pos - e.scene.env_origins[0]
            eef_rel  = eef_pos - e.scene.env_origins[0]

            dist_eef_cc  = (eef_pos - cc_pos).norm().item()
            dist_cc_drw  = (cc_pos  - drw_pos).norm().item()

            print(
                f"step {step:4d} | "
                f"cc_local=({cc_rel[0]:.3f},{cc_rel[1]:.3f},{cc_rel[2]:.3f}) | "
                f"drw_local=({drw_rel[0]:.3f},{drw_rel[1]:.3f},{drw_rel[2]:.3f}) | "
                f"eef_local=({eef_rel[0]:.3f},{eef_rel[1]:.3f},{eef_rel[2]:.3f}) | "
                f"d(eef→cc)={dist_eef_cc:.3f} d(cc→drw)={dist_cc_drw:.3f} | "
                f"stage={stage} grasped={int(grasped)} inside={int(inside)} | "
                f"rew={rew0:+.4f}"
            )

            if args_cli.verbose:
                # drawer joint pos (check it stays open)
                drw_joint = e._cabinet.data.joint_pos[0, e._cabinet.find_joints("drawer_top_joint")[0][0]].item()
                # cream_cheese rotation
                cc_quat = e.target_object.data.root_quat_w[0].tolist()
                print(
                    f"         drawer_joint={drw_joint:.4f} (target {e.cfg.drawer_open_pos:.3f}) | "
                    f"cc_quat=({cc_quat[0]:.3f},{cc_quat[1]:.3f},{cc_quat[2]:.3f},{cc_quat[3]:.3f})"
                )

        done = terminated | truncated
        if done.any():
            n_term = terminated.sum().item()
            n_trunc = truncated.sum().item()
            print(f"  -> reset: {n_term} terminated (success), {n_trunc} truncated")

        if not simulation_app.is_running():
            break

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect an Isaac Lab env with zero actions.")
    parser.add_argument("--task",         type=str,   default="TestPlaceCreamCheeseInDrawer", help="Gym task name.")
    parser.add_argument("--num_envs",    type=int,   default=4)
    parser.add_argument("--steps",       type=int,   default=300,  help="Total sim steps.")
    parser.add_argument("--print_every", type=int,   default=10,   help="Print interval (steps).")
    parser.add_argument("--device",      type=str,   default="cuda")
    parser.add_argument("--headless",    action="store_true", default=False)
    parser.add_argument("--verbose",     action="store_true", default=False,
                        help="Also print drawer joint pos and cream_cheese quaternion.")
    args_cli = parser.parse_args()
    main(args_cli)
