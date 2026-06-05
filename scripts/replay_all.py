"""Batch physics-faithful replay of ALL LIBERO demos for the PickItUp task.

Replays every episode in a single env, headless, no rendering/recording, then
writes an aggregate summary (success/grasp/lift rates) + per-episode CSV to the
log dir. Controller (joint_pd / osc) is the hard-coded ``REPLAY_CONTROLLER`` flag
inside ReplayPickItUpOneEnv.

Usage:
    python scripts/replay_all.py
    python scripts/replay_all.py --log_dir logs/replay_all_osc
"""

import os
import torch

from isaaclab_eureka.utils import eureka_root_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default=None)
    args = parser.parse_args()

    task = "ReplayPickItUpOneEnv"
    root = eureka_root_dir()
    log_dir = args.log_dir if args.log_dir else os.path.join(root, "logs", f"replay_all_{task}")

    from isaaclab.app import AppLauncher
    from isaaclab_eureka.utils import get_freest_gpu

    device = f"cuda:{get_freest_gpu()}"
    app_launcher = AppLauncher(headless=True, device=device)  # batch is always headless
    simulation_app = app_launcher.app

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg: DirectRLEnvCfg = parse_env_cfg(task)
    env_cfg.sim.device = device
    env_cfg.scene.num_envs = 1  # single-env sequential replay
    env = gym.make(task, cfg=env_cfg)
    env = env.unwrapped
    env.reset()
    env.run_replay_all(log_dir)
    print("finished replaying all episodes, exiting")
    env.close()

    torch.cuda.empty_cache()
    env.sim.stop()
    env.sim.clear()
    simulation_app.close()
