"""Record two LIBERO-style cameras (agentview + wrist) during physics replay.

Replays one controller/gain config over the demos in a single env, capturing
RGB from an external 'agentview' camera and a wrist 'eye_in_hand' camera at
20 fps. Successful and failed episodes (by the `terminated` flag) are written to
separate folders, with state/action arrays + a manifest, ready to convert to a
LeRobot v3 dataset later (scripts/convert_to_lerobot.py, in a separate env).

The config defaults to the gain-sweep winner (joint_pd, kp=3000 / kd=260).

Usage:
    python scripts/replay_record.py --episodes 50
    python scripts/replay_record.py --episodes 4 --controller osc --osc_kp 800
    python scripts/replay_record.py --controller joint_pd --kp 2000 --kd 200
"""

import os
import torch

from isaaclab_eureka.utils import eureka_root_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=None, help="num episodes (default: all 50)")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--controller", type=str, default="joint_pd", choices=["joint_pd", "osc"])
    parser.add_argument("--kp", type=float, default=3000.0, help="joint_pd arm stiffness")
    parser.add_argument("--kd", type=float, default=260.0, help="joint_pd arm damping")
    parser.add_argument("--osc_kp", type=float, default=150.0)
    parser.add_argument("--osc_dr", type=float, default=1.0)
    args = parser.parse_args()

    task = "ReplayPickItUpOneEnv"
    root = eureka_root_dir()
    log_dir = args.log_dir if args.log_dir else os.path.join(root, "logs", "replay_record")

    from isaaclab.app import AppLauncher
    from isaaclab_eureka.utils import get_freest_gpu

    device = f"cuda:{get_freest_gpu()}"
    # enable_cameras is REQUIRED for the RGB sensors to render (even headless).
    app_launcher = AppLauncher(headless=True, enable_cameras=True, device=device)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg: DirectRLEnvCfg = parse_env_cfg(task)
    env_cfg.sim.device = device
    env_cfg.scene.num_envs = 1
    env_cfg.record_cameras = True
    env = gym.make(task, cfg=env_cfg)
    env = env.unwrapped

    # pick the controller + gains to record
    env.REPLAY_CONTROLLER = args.controller
    if args.controller == "joint_pd":
        env.PD_KP, env.PD_KD = args.kp, args.kd
    else:
        env.OSC_KP, env.OSC_DAMPING_RATIO = args.osc_kp, args.osc_dr

    env.reset()
    env.run_replay_all(log_dir, num_episodes=args.episodes, record_dir=log_dir)
    print("finished recording, exiting")
    env.close()

    torch.cuda.empty_cache()
    env.sim.stop()
    env.sim.clear()
    simulation_app.close()
