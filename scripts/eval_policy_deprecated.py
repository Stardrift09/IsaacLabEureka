"""Evaluation rollout for the PickItUp replay env.

Feeds 8-dim actions (7 arm joint targets + 1 gripper) into the env one control
step at a time and runs physics — the harness for evaluating a policy trained on
the recorded LeRobot dataset. The gripper is copied onto both finger joints
inside the env before applying.

By default this runs a SELF-TEST: the "policy" is the demo's own action sequence
(`env.demo_action_sequence`), so the success rate should match the replay
(~48/50 for joint_pd kp3000). Swap `policy_action` for a real policy to evaluate
it. Pass --cameras to also get the two image observations each step.

Usage:
    python scripts/eval_policy.py --episodes 5
    python scripts/eval_policy.py --episodes 50 --kp 3000 --kd 260
"""

import os
import torch

from isaaclab_eureka.utils import eureka_root_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--controller", type=str, default="joint_pd", choices=["joint_pd", "osc"])
    parser.add_argument("--kp", type=float, default=3000.0)
    parser.add_argument("--kd", type=float, default=260.0)
    parser.add_argument("--cameras", action="store_true", help="also return image obs (needs render)")
    args = parser.parse_args()

    task = "ReplayPickItUpOneEnv"
    root = eureka_root_dir()

    from isaaclab.app import AppLauncher
    from isaaclab_eureka.utils import get_freest_gpu

    device = f"cuda:{get_freest_gpu()}"
    app_launcher = AppLauncher(headless=True, enable_cameras=args.cameras, device=device)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg: DirectRLEnvCfg = parse_env_cfg(task)
    env_cfg.sim.device = device
    env_cfg.scene.num_envs = 1
    env_cfg.record_cameras = args.cameras
    env = gym.make(task, cfg=env_cfg)
    env = env.unwrapped
    env.REPLAY_CONTROLLER = args.controller
    if args.controller == "joint_pd":
        env.PD_KP, env.PD_KD = args.kp, args.kd
    env.reset()

    def policy_action(obs, demo_seq, t):
        """Self-test policy: replay the demo's ground-truth action. Replace this
        with your trained policy (obs -> 8-dim action)."""
        return demo_seq[t]

    n_success = 0
    for ep in range(args.episodes):
        demo_seq = env.demo_action_sequence(ep)  # [T, 8]
        horizon = demo_seq.shape[0]
        obs = env.eval_reset(ep)
        success = False
        for t in range(horizon):
            action = policy_action(obs, demo_seq, t)
            obs, terminated, truncated, info = env.eval_step(action)
            success = success or info["success"]
            if info["success"]:
                break
        n_success += int(success)
        print(f"  ep {ep:>3}: success={success}  grasped={info['grasped']}  obj_z={info['object_z']:.3f}",
              flush=True)

    print(f"[eval] success {n_success}/{args.episodes} "
          f"({100.0 * n_success / max(args.episodes, 1):.1f}%)", flush=True)

    env.close()
    torch.cuda.empty_cache()
    env.sim.stop()
    env.sim.clear()
    simulation_app.close()
