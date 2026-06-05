"""Physics-faithful single-env replay of a LIBERO demo for the PickItUp task.

Drives ONE env with a controller (joint-PD or OSC, selected by the hard-coded
``REPLAY_CONTROLLER`` flag inside ReplayPickItUpOneEnv) while the object obeys
physics, then prints a report on whether the demo terminates / grasps / lifts.

Usage:
    python scripts/replay_one_env.py --episode_idx 0
    python scripts/replay_one_env.py --episode_idx 3 --headless
"""

import os
import torch

from isaaclab_eureka.utils import eureka_root_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_idx", type=int, default=0)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--headless", action="store_true", default=False)
    args = parser.parse_args()

    task = "ReplayPickItUpOneEnv"
    root = eureka_root_dir()
    log_dir = args.log_dir if args.log_dir else os.path.join(root, "logs", f"replay_{task}")

    from isaaclab.app import AppLauncher
    from isaaclab_eureka.utils import get_freest_gpu

    device = "cuda"
    if device == "cuda":
        device = f"cuda:{get_freest_gpu()}"
    app_launcher = AppLauncher(headless=args.headless, device=device)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg: DirectRLEnvCfg = parse_env_cfg(task)
    env_cfg.sim.device = device
    env_cfg.scene.num_envs = 1  # single-env replay
    env = gym.make(task, cfg=env_cfg)
    env = env.unwrapped
    env.reset()
    env.run_replay_one_env(log_dir, episode_idx=args.episode_idx, render=not args.headless)
    print("finished replaying, exiting")
    env.close()

    torch.cuda.empty_cache()
    env.sim.stop()
    env.sim.clear()
    simulation_app.close()
