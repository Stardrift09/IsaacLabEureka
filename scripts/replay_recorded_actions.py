"""Sanity-check the RL->BC dataset actions by REPLAYING them through the eval interface.

Question this answers: do the actions we record in ``collect_rsl_rl.py`` (arm = achieved
next joint pos, gripper = COMMANDED target) actually reproduce the task when fed back as
ABSOLUTE position targets via ``eval_step`` -- i.e. is the recorded dataset action-faithful,
and does the gripper fix actually grasp?

The recorded ``arrays.npz`` does NOT store the randomized object/basket poses, so we cannot
replay from disk faithfully. Instead, per episode we:
  1. reset (randomized init) and roll out the rsl_rl policy via the NATIVE step interface,
     recording the SAME hybrid actions as collect_rsl_rl.py;
  2. SNAPSHOT the full sim state (robot joints + every rigid object) at frame 0;
  3. RESTORE that snapshot and replay the recorded actions through ``eval_step`` (absolute
     position targets, the BC/eval path);
  4. compare RL-rollout success vs replay success.

If replay success ~= RL success, the recorded actions are faithful and the eval interface
reproduces them. If replay grasps (object lifted) where the old next-state gripper didn't,
the gripper fix is confirmed end-to-end.

Run (conda eureka; NO policy server, NO cameras needed):
    ~/miniconda3/envs/eureka/bin/python -u scripts/replay_recorded_actions.py \
        --checkpoint <.../model_1499.pt> --num_episodes 10 --headless
"""

import argparse

from isaaclab_eureka.utils import get_freest_gpu


def main(args_cli):
    from isaaclab.app import AppLauncher

    device = args_cli.device
    if device == "cuda":
        device = f"cuda:{get_freest_gpu()}"

    # No cameras: the rsl_rl policy is proprioceptive and we only check success, not video.
    app_launcher = AppLauncher(headless=args_cli.headless, device=device)
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
    env_cfg.record_cameras = False  # success-only test; no rendering needed
    env = gym.make(task, cfg=env_cfg)

    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = device
    env = RslRlVecEnvWrapper(env)
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    ppo_runner.load(args_cli.checkpoint, map_location=device)
    policy = ppo_runner.get_inference_policy(device=device)
    u = env.unwrapped
    GD = u.GRIPPER_DIM
    max_steps = int(u.max_episode_length) + 5

    # ---- full sim-state snapshot / restore (robot joints + every rigid object) ----
    def capture():
        return {
            "jp": u._robot.data.joint_pos.clone(),
            "jv": u._robot.data.joint_vel.clone(),
            "objs": {k: o.data.root_state_w.clone() for k, o in u.rigid_objects.items()},
        }

    @torch.inference_mode()  # write_*_to_sim does inplace updates on the env's inference tensors
    def restore(snap):
        u._robot.write_joint_state_to_sim(snap["jp"], snap["jv"])
        for k, o in u.rigid_objects.items():
            s = snap["objs"][k]
            o.write_root_pose_to_sim(s[:, :7])
            o.write_root_velocity_to_sim(s[:, 7:13])
        u.scene.write_data_to_sim()
        u.scene.update(dt=u.physics_dt)
        # clear accumulating task flags + clock so replay success/truncation are fresh
        u.episode_length_buf[:] = 0
        u.grasped_and_lifted[:] = False
        u.grasped[:] = False
        u._get_dones()
        u._get_observations()

    print(f"[replay_actions] task={task} episodes={args_cli.num_episodes}", flush=True)
    n_rl_succ = n_rep_succ = n_rep_grasp = n_match = 0

    for ep in range(args_cli.num_episodes):
        # ---- 1. reset, SNAPSHOT the init (pre-first-step) state, then native rollout ----
        with torch.inference_mode():  # _reset_idx does inplace writes on the env's inference tensors
            obs, _ = env.reset()
        snap = capture()              # episode init: exactly the state collect's actions start from
        tgts = []
        grasped_rl = False
        done = False
        t = 0
        while not done:
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
            done = bool(dones[0].item()) or t >= max_steps
            if not done:
                # action = COMMANDED clamped target (`_target8`), EXACTLY as collect_rsl_rl.py records
                tgts.append(u._target8().detach().cpu().numpy().copy())
                grasped_rl = grasped_rl or bool(u.grasped[0].item())
                t += 1
        success_rl = bool(u.reset_terminated[0].item())
        if len(tgts) < 2:
            print(f"  ep {ep:>3}: too short ({len(tgts)} frames), skipping", flush=True)
            continue

        # ---- 2. restore the init and replay the recorded actions via eval_step ----
        # action[i] = tgts[i] (UNSHIFTED, the collect convention), applied from the reset state
        # -> same init, same commands, same 2-step cadence as the native rollout.
        restore(snap)
        success_rep = grasped_rep = False
        z0 = u.target_object.data.root_pos_w[0, 2].item()
        max_z = z0
        with torch.inference_mode():
            for a in tgts:
                _, term, trunc, info = u.eval_step(torch.as_tensor(a, device=device))
                max_z = max(max_z, info["object_z"])
                grasped_rep = grasped_rep or bool(info["grasped"])
                success_rep = success_rep or bool(info["success"])
                if bool(term[0].item()) or bool(trunc[0].item()):
                    break

        n_rl_succ += success_rl
        n_rep_succ += success_rep
        n_rep_grasp += grasped_rep
        n_match += (success_rl == success_rep)
        print(f"  ep {ep:>3}: RL[succ={int(success_rl)} grasp={int(grasped_rl)}]  "
              f"REPLAY[succ={int(success_rep)} grasp={int(grasped_rep)} lift={max_z - z0:+.3f}]  "
              f"{'MATCH' if success_rl == success_rep else 'DIFF'}  frames={len(tgts)}", flush=True)

    N = args_cli.num_episodes
    print("=" * 78, flush=True)
    print(f"[replay_actions] RL success {n_rl_succ}/{N} | REPLAY success {n_rep_succ}/{N} "
          f"| replay grasp {n_rep_grasp}/{N} | RL==REPLAY {n_match}/{N}", flush=True)
    print("=" * 78, flush=True)

    env.close()
    torch.cuda.empty_cache()
    simulation_app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay recorded RL->BC actions through eval_step.")
    parser.add_argument("--task", type=str, default="EvalPickItUpOneEnv")
    parser.add_argument("--checkpoint", type=str, required=True, help="Absolute path to model_*.pt.")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--headless", action="store_true", default=True)
    args_cli = parser.parse_args()
    main(args_cli)
