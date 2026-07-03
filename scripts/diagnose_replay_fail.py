"""Diagnose WHY replaying recorded rsl_rl actions reproduces grasp+lift but not success.

For each episode: reset -> snapshot init -> NATIVE closed-loop rollout (records _target8 +
the object/basket trajectory + the four success booleans) -> restore -> OPEN-LOOP replay #1
through eval_step (same logging) -> restore -> replay #2 (identical actions).

It answers three things with evidence (no guessing):
  1. determinism: does replay #1 == replay #2?  (replay1!=replay2 => PhysX non-determinism;
     replay1==replay2!=native => a systematic eval_step-vs-native discrepancy = a code bug)
  2. divergence onset: first frame the replayed object position departs the native one by
     >1cm / >5cm (gradual-from-0 => path mismatch; only-late => sensitive release phase)
  3. which condition fails: per-frame inside_site / low_enough / high_enough_for_basket /
     grasped_and_lifted, so we see exactly what the object fails to satisfy at the end.

Run (conda eureka; no server, no cameras):
    ~/miniconda3/envs/eureka/bin/python -u scripts/diagnose_replay_fail.py \
        --checkpoint <.../model_1499.pt> --num_episodes 3 --headless
"""

import argparse

import numpy as np

from isaaclab_eureka.utils import get_freest_gpu


def main(args_cli):
    from isaaclab.app import AppLauncher

    device = args_cli.device
    if device == "cuda":
        device = f"cuda:{get_freest_gpu()}"
    app_launcher = AppLauncher(headless=args_cli.headless, device=device)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunner

    task = args_cli.task
    env_cfg = parse_env_cfg(task, device=device, num_envs=1)
    env_cfg.sim.device = device
    env_cfg.record_cameras = False
    env = gym.make(task, cfg=env_cfg)
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = device
    env = RslRlVecEnvWrapper(env)
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    ppo_runner.load(args_cli.checkpoint, map_location=device)
    policy = ppo_runner.get_inference_policy(device=device)
    u = env.unwrapped
    u.REPLAY_CONTROLLER = "rsl_rl"
    max_steps = int(u.max_episode_length) + 5

    def capture():
        return {
            "jp": u._robot.data.joint_pos.clone(),
            "jv": u._robot.data.joint_vel.clone(),
            "objs": {k: o.data.root_state_w.clone() for k, o in u.rigid_objects.items()},
        }

    @torch.inference_mode()
    def restore(snap):
        u._robot.write_joint_state_to_sim(snap["jp"], snap["jv"])
        for k, o in u.rigid_objects.items():
            s = snap["objs"][k]
            o.write_root_pose_to_sim(s[:, :7])
            o.write_root_velocity_to_sim(s[:, 7:13])
        u.scene.write_data_to_sim()
        u.scene.update(dt=u.physics_dt)
        u.episode_length_buf[:] = 0
        u.grasped_and_lifted[:] = False
        u.grasped[:] = False
        u._get_dones()
        u._get_observations()

    def probe():
        """Snapshot the success-relevant state AFTER a _get_dones (native step or eval_step)."""
        obj = u.target_object.data.root_pos_w[0].detach().cpu().numpy().copy()  # [3]
        site = u.target_site.data.root_pos_w[0, :2].detach().cpu().numpy().copy()  # [2]
        return {
            "obj": obj,
            "dist_site": float(np.linalg.norm(obj[:2] - site)),
            "inside_site": bool(u.inside_site[0].item()),
            "low_enough": bool(u.low_enough[0].item()),
            "high_for_basket": bool(u.high_enough_for_basket[0].item()),
            "grasped_lifted": bool(u.grasped_and_lifted[0].item()),
        }

    def native_rollout():
        with torch.inference_mode():
            obs, _ = env.reset()
        snap = capture()
        tgts, traj, done, t = [], [], False, 0
        while not done:
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
            done = bool(dones[0].item()) or t >= max_steps
            if not done:
                tgts.append(u._target8().detach().cpu().numpy().copy())
                traj.append(probe())
                t += 1
        return snap, tgts, traj, bool(u.reset_terminated[0].item())

    def open_loop_replay(snap, tgts):
        restore(snap)
        traj, success = [], False
        with torch.inference_mode():
            for a in tgts:
                _, term, trunc, info = u.eval_step(torch.as_tensor(a, device=device))
                traj.append(probe())
                success = success or bool(info["success"])
                if bool(term[0].item()) or bool(trunc[0].item()):
                    break
        return traj, success

    def first_div(a, b, thresh):
        n = min(len(a), len(b))
        for i in range(n):
            if np.linalg.norm(a[i]["obj"] - b[i]["obj"]) > thresh:
                return i
        return -1

    def max_obj_diff(a, b):
        n = min(len(a), len(b))
        return max((float(np.linalg.norm(a[i]["obj"] - b[i]["obj"])) for i in range(n)), default=float("nan"))

    print(f"[diagnose] task={task} episodes={args_cli.num_episodes}", flush=True)
    for ep in range(args_cli.num_episodes):
        snap, tgts, nat, succ_rl = native_rollout()
        if len(tgts) < 2:
            print(f"ep {ep}: too short, skip", flush=True)
            continue
        rep1, s1 = open_loop_replay(snap, tgts)
        rep2, s2 = open_loop_replay(snap, tgts)

        nf, n1, n2 = nat[-1], rep1[-1], rep2[-1]
        print(f"\n==== ep {ep}  frames={len(tgts)}  RL_success={succ_rl} ====", flush=True)
        print(f"  replay determinism: max|obj_rep1 - obj_rep2| = {max_obj_diff(rep1, rep2)*1000:.2f} mm "
              f"(success rep1={s1} rep2={s2})", flush=True)
        print(f"  native-vs-replay1 obj divergence: first >1cm @ frame {first_div(nat, rep1, 0.01)}, "
              f">5cm @ frame {first_div(nat, rep1, 0.05)}  (max {max_obj_diff(nat, rep1)*1000:.1f} mm)", flush=True)
        for name, p in (("NATIVE end ", nf), ("REPLAY1 end", n1), ("REPLAY2 end", n2)):
            print(f"  {name}: obj=({p['obj'][0]:+.3f},{p['obj'][1]:+.3f},{p['obj'][2]:+.3f}) "
                  f"dist_site={p['dist_site']:.3f}  inside={int(p['inside_site'])} "
                  f"low={int(p['low_enough'])} high_basket={int(p['high_for_basket'])} "
                  f"grasp_lift={int(p['grasped_lifted'])}", flush=True)

    env.close()
    torch.cuda.empty_cache()
    simulation_app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose why recorded-action replay fails to reproduce success.")
    parser.add_argument("--task", type=str, default="EvalPickItUpOneEnv")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--headless", action="store_true", default=True)
    args_cli = parser.parse_args()
    main(args_cli)
