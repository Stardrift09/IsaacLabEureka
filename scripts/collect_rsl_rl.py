"""Collect a LeRobot-format dataset by rolling out a trained **rsl_rl** policy.

This is the RL->BC data path. An rsl_rl (PPO) policy drives the env through its
NATIVE action interface (8-dim *delta* actions, `_pre_physics_step`, 60 Hz control =
decimation 2 @ 120 Hz) -- so it CANNOT go through `eval_client.py` / `eval_step`
(those apply *absolute* joint targets and talk to the LeRobot BC server). Instead
we run it like `scripts/play.py` and record on `env_0`:

  - `observation.state`  [8]  : 7 arm joint pos + 1 gripper (`_state8`), ACHIEVED qpos.
  - `action`             [8]  : the COMMANDED absolute joint target (`_target8` =
        `robot_dof_targets`) -- the clamped PD target the policy actually drove the arm
        with. This is the ONLY representation that REPLAYS: feeding it back through
        `eval_step` reproduces the rollout (8-9/10 success). The arm actuator is soft
        (stiffness ~1.5k, effort cap 80 Nm), so the bang-bang policy holds the target at
        the joint LIMITS to generate torque via large position error; the achieved state
        is far from the target (corr ~0, 52-99% saturated) BY DESIGN.
        NOTE: do NOT record the achieved next-state (state[t+1]) as the action -- it has
        ~0 position error once reached -> ~0 holding/lifting torque -> open-loop replay
        gives 0/10 (arm cannot lift). Verified by scripts/replay_recorded_actions.py.
        Caveat: this action is near-bang-bang, so it is harder for BC to imitate; the
        clean long-term fix is to retrain RL with a smooth position-control action space
        (then achieved ~= commanded and next-state would be both faithful AND learnable).
  - `observation.images.image` / `image2` : agentview + wrist RGB, 256^2

Output is the SAME layout as `scripts/replay_record.py` (reuses the env's
`_write_episode_recording` / `_write_dataset_manifest`), split into
`successful/` vs `unsuccessful/` by the task-success (`terminated`) flag, so the
existing converter turns it into the `pickitup_replay` schema unchanged.

Control is 60 Hz; we capture every `--record_every` control step, and the dataset
fps is computed from the sim cfg (`_recording_fps`) = 60 / record_every. So the
default `--record_every 1` records at 60 fps; pass `--record_every 3` for 20 fps.

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

    # Cap episodes at args_cli.max_frames control steps. max_episode_length is a
    # property derived from cfg.episode_length_s, and _get_dones truncates at
    # episode_length_buf >= max_episode_length - 1, after which env.step() auto-resets
    # env_0 -- so the existing done/record loop needs no other change. Set this BEFORE
    # max_steps below so the safety net tracks the new (shorter) cap.
    if args_cli.max_frames and args_cli.max_frames > 0:
        control_dt = u.cfg.sim.dt * u.cfg.decimation
        u.cfg.episode_length_s = args_cli.max_frames * control_dt
        print(f"[collect_rsl_rl] episode cap = {args_cli.max_frames} frames "
              f"(episode_length_s={u.cfg.episode_length_s:.4f}, "
              f"max_episode_length={u.max_episode_length})", flush=True)

    max_steps = int(u.max_episode_length) + 5  # truncation safety net

    obs = env.get_observations()
    u._aim_agentview()  # static agentview points at env_0 workspace (one-time)

    def fresh():
        return ({name: [] for name in u._record_cams}, [], [])

    results = []
    ep = 0
    cam_frames, states_log, actions_log = fresh()
    step_in_ep, grasped_ever = 0, False
    # fps derived from the sim cfg (no hard-coding): we capture every
    # `record_every` control steps, and each control step is `decimation` physics
    # steps -> fps = 1 / (record_every * decimation * sim.dt) = 60 / record_every.
    rec_fps = u._recording_fps(args_cli.record_every * u.cfg.decimation)
    print(f"[collect_rsl_rl] task={task} episodes={args_cli.num_episodes} "
          f"record_every={args_cli.record_every} fps={rec_fps} -> {args_cli.record_dir}", flush=True)

    while ep < args_cli.num_episodes:
        # PAIRING (critical for BC): a dataset row must be (image_t, state_t, T_t) -- the
        # observation BEFORE acting paired with the target applied at it. So capture the
        # frame+state at the TOP (= obs_t, the current sim configuration) and read the
        # applied target T_t AFTER the step (robot_dof_targets, set by _pre_physics_step
        # from `actions`). Capturing POST-step instead stores (image_{t+1}, state_{t+1},
        # T_t): the label lags one control step, so the policy learns to re-command the
        # target that produced the CURRENT state and stalls. The near-bang-bang target
        # swings ~0.12 rad/joint/step, so that lag is a ~7-degree systematic error per
        # step. (Subsample 60 Hz control -> dataset fps = 60 / record_every.)
        record_now = (step_in_ep % args_cli.record_every == 0)
        if record_now:
            fr = u._capture_frame()                              # image_t
            state_t = u._state8().detach().cpu().numpy().copy()  # state_t
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

        # env.step() auto-resets terminated/truncated envs INSIDE the call, so once `done`
        # the env already reflects the NEXT episode's reset state. Detect done and skip
        # recording that step (the final obs_t/T_t pair is dropped; finalize from prior
        # in-episode steps).
        done = bool(dones[0].item()) or step_in_ep >= max_steps

        if not done:
            if record_now:
                for name in cam_frames:
                    cam_frames[name].append(fr[name])
                states_log.append(state_t)
                # action = the COMMANDED clamped joint target (`_target8` = robot_dof_targets),
                # all 8 dims. This is the actual PD target the policy drove the arm with, and
                # the ONLY action that replays through eval_step (next-state gives 0/10 -- no
                # lifting torque). See the module docstring.
                actions_log.append(u._target8().detach().cpu().numpy().copy())  # T_t
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
            # actions_log already holds the per-step commanded `_target8` (no post-processing).
            stats = {
                "episode": ep, "controller": u.REPLAY_CONTROLLER,
                "frames": len(states_log), "terminated_step": -1,
                "success": success, "grasped": grasped_ever,
            }
            stats.update(u._write_episode_recording(
                args_cli.record_dir, ep, success, cam_frames, states_log, actions_log, rec_fps))
            results.append(stats)
            print(f"  ep {ep:>3}: success={success}  grasped(ever)={grasped_ever}  "
                  f"frames={len(states_log)}  {'OK ' if success else 'FAIL'}", flush=True)
        ep += 1
        cam_frames, states_log, actions_log = fresh()
        step_in_ep, grasped_ever = 0, False
        # No manual flag-clearing needed: the grasped_and_lifted carry-over bug is
        # fixed at the root in TestPickItUp._reset_idx (assignment, not no-op fill_).

    if results:
        u._write_dataset_manifest(args_cli.record_dir, results, len(results), rec_fps)
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
    parser.add_argument("--num_episodes", type=int, default=10000, help="Number of episodes to collect.")
    parser.add_argument("--max_frames", type=int, default=100,
                        help="Truncate (and reset) each episode after this many control steps. "
                             "Caps long non-success rollouts; successful episodes end earlier. "
                             "Pass 0 to disable and use the env's default episode length.")
    parser.add_argument("--record_dir", type=str,
                        default="logs/collect_rsl_rl_v15", help="Output dir (LeRobot-convertible).")
    parser.add_argument("--record_every", type=int, default=1,
                        help="Record every Nth control step. Dataset fps = 60 / N "
                             "(N=1 -> 60 fps, N=3 -> 20 fps).")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--stochastic", action="store_true", default=False,
                        help="Sample actions (training behavior) instead of the policy mean.")
    parser.add_argument("--headless", action="store_true", default=True)
    args_cli = parser.parse_args()
    main(args_cli)
