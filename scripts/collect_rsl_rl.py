"""Collect a LeRobot-format dataset by rolling out a trained **rsl_rl** policy.

This is the RL->BC data path. An rsl_rl (PPO) policy drives the env through its
NATIVE action interface (8-dim *delta* actions, `_pre_physics_step`, 60 Hz control =
decimation 2 @ 120 Hz) -- so it CANNOT go through `eval_client.py` / `eval_step`
(those apply *absolute* targets at 20 Hz and talk to the LeRobot BC server). Instead
we run it like `scripts/play.py` and record on `env_0`:

  - `observation.state`  [8]  : 7 arm joint pos + 1 gripper (`_state8`)
  - `action`             [8]  : the ABSOLUTE joint targets the policy commanded
                                (`_target8`, = `robot_dof_targets`), so the dataset
                                is action-compatible with the replay/BC dataset and
                                `eval_step`.
  - `observation.images.image` / `image2` : agentview + wrist RGB, 256^2

Output is the SAME layout as `scripts/replay_record.py` (reuses the env's
`_write_episode_recording` / `_write_dataset_manifest`), split into
`successful/` vs `unsuccessful/` by the task-success (`terminated`) flag, so the
existing converter turns it into the `pickitup_replay` schema (codebase v3.0,
fps 20) unchanged.

Control is 60 Hz; the dataset is 20 fps -> we record every `--record_every` (=3)
control step.

Run in the IsaacLab env (conda `eureka`); NO policy server needed:
    ~/miniconda3/envs/eureka/bin/python -u scripts/collect_rsl_rl.py \
        --checkpoint <.../model_1499.pt> --num_episodes 50 \
        --record_dir logs/collect_policy/test_pick_it_up/policy_1 --headless
"""

import argparse

from isaaclab_eureka.utils import get_freest_gpu


def main(args_cli):
    from isaaclab.app import AppLauncher

    device = args_cli.device
    if device == "cuda":
        device = f"cuda:{get_freest_gpu()}"

    # enable_cameras REQUIRED for the RGB sensors to render (even headless).
    app_launcher = AppLauncher(headless=args_cli.headless, enable_cameras=True, device=device)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401  (registers EvalPickItUpOneEnv)
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunner

    task = args_cli.task
    env_cfg = parse_env_cfg(task, device=device, num_envs=1)
    env_cfg.sim.device = device
    env_cfg.record_cameras = True  # -> agentview + wrist cameras (env_0)
    env = gym.make(task, cfg=env_cfg)

    # ---- load the rsl_rl policy (mirrors scripts/play.py) -------------------
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = device
    env = RslRlVecEnvWrapper(env)
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    ppo_runner.load(args_cli.checkpoint, map_location=device)
    # mirror scripts/play.py: sampled actions (training behavior) vs the mean.
    if args_cli.stochastic:
        policy = lambda obs: ppo_runner.alg.policy.act(obs)  # noqa: E731
    else:
        policy = ppo_runner.get_inference_policy(device=device)
    u = env.unwrapped
    u.REPLAY_CONTROLLER = "rsl_rl"  # tags the manifest/meta (gains are the env's native actuators)

    # ---- rollout + record ---------------------------------------------------
    import os
    os.makedirs(args_cli.record_dir, exist_ok=True)
    max_steps = int(u.max_episode_length) + 5  # truncation safety net

    obs = env.get_observations()
    u._aim_agentview()  # static agentview points at env_0 workspace (one-time)

    def fresh():
        return ({name: [] for name in u._record_cams}, [], [])

    results = []
    ep = 0
    cam_frames, states_log, actions_log = fresh()
    step_in_ep, grasped_ever = 0, False
    print(f"[collect_rsl_rl] task={task} episodes={args_cli.num_episodes} "
          f"record_every={args_cli.record_every} -> {args_cli.record_dir}", flush=True)

    while ep < args_cli.num_episodes:
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

        # env.step() auto-resets terminated/truncated envs INSIDE the call, so once
        # `done` the obs / _capture_frame / _state8 / _target8 already reflect the
        # NEXT episode's reset state. Detect done first and skip recording that step
        # (finalize the episode from the frames recorded on prior, in-episode steps).
        done = bool(dones[0].item()) or step_in_ep >= max_steps

        if not done:
            # subsample 60 Hz control -> 20 fps dataset; record POST-step (state/frame
            # paired with the absolute target that produced them).
            if step_in_ep % args_cli.record_every == 0:
                fr = u._capture_frame()
                for name in cam_frames:
                    cam_frames[name].append(fr[name])
                states_log.append(u._state8().detach().cpu().numpy().copy())
                actions_log.append(u._target8().detach().cpu().numpy().copy())
            grasped_ever = grasped_ever or bool(u.grasped[0].item())
            step_in_ep += 1
            continue

        # ---- episode boundary: env already reset env_0 inside env.step() ----
        # `reset_terminated` is set by _get_dones and NOT cleared by _reset_idx, so
        # it still reflects THIS episode's termination (task success vs timeout).
        success = bool(u.reset_terminated[0].item())
        if len(states_log) == 0:
            print(f"  ep {ep:>3}: empty (no recorded frames), skipping", flush=True)
        else:
            stats = {
                "episode": ep, "controller": u.REPLAY_CONTROLLER,
                "frames": len(states_log), "terminated_step": -1,
                "success": success, "grasped": grasped_ever,
            }
            stats.update(u._write_episode_recording(
                args_cli.record_dir, ep, success, cam_frames, states_log, actions_log))
            results.append(stats)
            print(f"  ep {ep:>3}: success={success}  grasped(ever)={grasped_ever}  "
                  f"frames={len(states_log)}  {'OK ' if success else 'FAIL'}", flush=True)
        ep += 1
        cam_frames, states_log, actions_log = fresh()
        step_in_ep, grasped_ever = 0, False
        # No manual flag-clearing needed: the grasped_and_lifted carry-over bug is
        # fixed at the root in TestPickItUp._reset_idx (assignment, not no-op fill_).

    if results:
        u._write_dataset_manifest(args_cli.record_dir, results, len(results))
    n_succ = sum(bool(r["success"]) for r in results)
    print(f"[collect_rsl_rl] DONE | success {n_succ}/{len(results)} "
          f"({100.0 * n_succ / max(len(results), 1):.1f}%) -> {args_cli.record_dir}", flush=True)

    env.close()
    torch.cuda.empty_cache()
    simulation_app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect a LeRobot dataset from an rsl_rl policy.")
    parser.add_argument("--task", type=str, default="EvalPickItUpOneEnv",
                        help="Randomized + camera env sharing TestPickItUp obs/action.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Absolute path to model_*.pt.")
    parser.add_argument("--num_episodes", type=int, default=500, help="Number of episodes to collect.")
    parser.add_argument("--record_dir", type=str,
                        default="logs/collect_rsl_rl", help="Output dir (LeRobot-convertible).")
    parser.add_argument("--record_every", type=int, default=3,
                        help="Record every Nth control step (60 Hz / 3 = 20 fps).")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--stochastic", action="store_true", default=False,
                        help="Sample actions (training behavior) instead of the policy mean.")
    parser.add_argument("--headless", action="store_true", default=False)
    args_cli = parser.parse_args()
    main(args_cli)
