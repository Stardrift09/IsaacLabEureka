"""Prove the rsl_rl->BC dataset is REPLAYABLE by recording the init and replaying it.

``collect_rsl_rl.py`` records ``observation.state`` / ``action`` / videos but NOT the
randomized scene (object + basket poses + robot reset), so its episodes can be turned
into a BC dataset but CANNOT be replayed back into the sim faithfully (feed the actions
from a fresh reset and the object spawns elsewhere -> the arm grasps empty air). This
script is ``collect_rsl_rl.py`` minus the dataset (no videos / no proprio dump) PLUS the
ONE missing thing: the per-episode INIT snapshot. With the init on disk, the recorded
action sequence replays deterministically through ``eval_step`` and reproduces the
rollout -- that is what "the recorded data are replayable" means, and this verifies it.

Layout mirrors ``collect_rsl_rl.py``: same env (``EvalPickItUpOneEnv``), same policy load,
same NATIVE actuator gains, same 60 Hz control, and the SAME recorded action
(``_target8`` = the clamped absolute joint target the policy drove the arm with -- the
only representation that replays through ``eval_step``; see that file's docstring).

Per episode we write ``<record_dir>/<successful|unsuccessful>/ep##/init_actions.npz``:
  - ``robot_joint_pos`` / ``robot_joint_vel`` [num_joints] : the reset robot state
  - ``object_names``  [n_obj]              : rigid-object keys, in row order of...
  - ``object_states`` [n_obj, 13]          : ...each object's root_state_w (pose+vel)
  - ``action``        [T, 8]               : recorded ``_target8`` per control step
  - ``success_rl`` / ``grasped_rl`` (scalar): the live-rollout outcome to replay against
  - ``fps`` (scalar)                       : control rate (eval_step cadence must match)

Modes (``--mode``):
  - ``collect`` : roll out the policy, snapshot init + actions, write the npz.
  - ``replay``  : READ each npz from ``--record_dir``, restore the init, replay the
                  actions through ``eval_step``, compare replay-success vs recorded
                  ``success_rl``. This is the actual "replay the recorded data" proof.
  - ``both`` (default): collect then replay (replay still reads from disk).

Run in the IsaacLab env (conda ``eureka``); NO policy server, NO cameras:
    ~/miniconda3/envs/eureka/bin/python -u scripts/replay_with_init.py \
        --checkpoint <.../model_1999.pt> --num_episodes 20 \
        --record_dir logs/replay_with_init_v9 --headless
"""

import argparse

from isaaclab_eureka.utils import get_freest_gpu


def main(args_cli):
    from isaaclab.app import AppLauncher

    device = args_cli.device
    if device == "cuda":
        device = f"cuda:{get_freest_gpu()}"

    # No cameras: replayability is a proprio/success check, not a video check (matches
    # replay_recorded_actions.py). Skipping the RGB sensors makes this much faster.
    app_launcher = AppLauncher(headless=args_cli.headless, device=device)
    simulation_app = app_launcher.app

    import json
    import os

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401  (registers EvalPickItUpOneEnv)
    import numpy as np
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunner

    task = args_cli.task
    env_cfg = parse_env_cfg(task, device=device, num_envs=1)
    env_cfg.sim.device = device
    env_cfg.record_cameras = False  # success/proprio replay only; no rendering
    env = gym.make(task, cfg=env_cfg)

    # ---- load the rsl_rl policy (mirrors collect_rsl_rl.py) ------------------
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = device
    env = RslRlVecEnvWrapper(env)
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    ppo_runner.load(args_cli.checkpoint, map_location=device)
    policy = ppo_runner.get_inference_policy(device=device)
    u = env.unwrapped
    u.REPLAY_CONTROLLER = "rsl_rl"  # NATIVE actuator gains (the gains the dataset uses)

    # eval_step cadence must match the dataset fps so each recorded action is held for the
    # same number of physics steps as during collection (60 fps -> 2 steps @ 120 Hz).
    steps = max(1, round((1.0 / args_cli.eval_fps) / u.cfg.sim.dt))
    u.EVAL_STEPS_PER_FRAME = steps
    max_steps = int(u.max_episode_length) + 5
    print(f"[replay_init] task={task} mode={args_cli.mode} eval_fps={args_cli.eval_fps} "
          f"-> EVAL_STEPS_PER_FRAME={steps}  record_dir={args_cli.record_dir}", flush=True)

    # ---- full sim-state snapshot / restore (robot joints + every rigid object) ----
    # Same capture/restore as scripts/replay_recorded_actions.py, but persisted to disk.
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

    def ep_path(success, ep):
        split = "successful" if success else "unsuccessful"
        return os.path.join(args_cli.record_dir, split, f"ep{ep:02d}")

    # ------------------------------------------------------------------
    # COLLECT: roll out the policy, snapshot init + actions, write npz.
    # ------------------------------------------------------------------
    def collect():
        os.makedirs(args_cli.record_dir, exist_ok=True)
        n_succ = 0
        for ep in range(args_cli.num_episodes):
            with torch.inference_mode():  # _reset_idx does inplace writes on inference tensors
                obs, _ = env.reset()
            snap = capture()  # episode init: exactly the state the recorded actions start from
            tgts, grasped_rl, done, t = [], False, False, 0
            while not done:
                with torch.inference_mode():
                    actions = policy(obs)
                    obs, _, dones, _ = env.step(actions)
                done = bool(dones[0].item()) or t >= max_steps
                if not done:
                    # action = COMMANDED clamped target (`_target8`), EXACTLY as collect_rsl_rl.py
                    tgts.append(u._target8().detach().cpu().numpy().copy())
                    grasped_rl = grasped_rl or bool(u.grasped[0].item())
                    t += 1
            success_rl = bool(u.reset_terminated[0].item())
            if len(tgts) < 2:
                print(f"  ep {ep:>3}: too short ({len(tgts)} frames), skipping", flush=True)
                continue

            d = ep_path(success_rl, ep)
            os.makedirs(d, exist_ok=True)
            obj_names = list(snap["objs"].keys())
            np.savez(
                os.path.join(d, "init_actions.npz"),
                robot_joint_pos=snap["jp"][0].cpu().numpy(),
                robot_joint_vel=snap["jv"][0].cpu().numpy(),
                object_names=np.array(obj_names),
                object_states=np.stack([snap["objs"][k][0].cpu().numpy() for k in obj_names]),
                action=np.asarray(tgts, dtype=np.float32),
                success_rl=np.array(success_rl),
                grasped_rl=np.array(grasped_rl),
                fps=np.array(args_cli.eval_fps),
            )
            n_succ += success_rl
            print(f"  ep {ep:>3}: RL success={int(success_rl)} grasp={int(grasped_rl)} "
                  f"frames={len(tgts)} -> {os.path.relpath(d, args_cli.record_dir)}", flush=True)
        print(f"[replay_init] COLLECT done | RL success {n_succ}/{args_cli.num_episodes} "
              f"-> {args_cli.record_dir}", flush=True)

    # ------------------------------------------------------------------
    # REPLAY: read each npz from disk, restore the init, replay the actions
    # through eval_step, compare replay-success vs the recorded RL success.
    # ------------------------------------------------------------------
    def replay():
        import glob
        files = sorted(glob.glob(os.path.join(args_cli.record_dir, "*", "ep*", "init_actions.npz")))
        if not files:
            print(f"[replay_init] no init_actions.npz under {args_cli.record_dir} -- run --mode collect first.",
                  flush=True)
            return
        with torch.inference_mode():
            env.reset()  # bring the sim live before the first restore
        n = n_rl = n_rep = n_grasp = n_match = 0
        for f in files:
            d = np.load(f, allow_pickle=True)
            snap = {
                "jp": torch.as_tensor(d["robot_joint_pos"], device=device).unsqueeze(0),
                "jv": torch.as_tensor(d["robot_joint_vel"], device=device).unsqueeze(0),
                "objs": {str(name): torch.as_tensor(state, device=device).unsqueeze(0)
                         for name, state in zip(d["object_names"], d["object_states"])},
            }
            actions = d["action"]
            success_rl = bool(d["success_rl"])

            restore(snap)
            z0 = u.target_object.data.root_pos_w[0, 2].item()
            max_z, success_rep, grasped_rep = z0, False, False
            succ_via = "-"
            with torch.inference_mode():
                for a in actions:
                    _, term, trunc, info = u.eval_step(torch.as_tensor(a, device=device))
                    max_z = max(max_z, info["object_z"])
                    grasped_rep = grasped_rep or bool(info["grasped"])
                    if bool(info["success"]):
                        success_rep, succ_via = True, "actions"
                    if bool(term[0].item()) or bool(trunc[0].item()):
                        break
                # Settle: the arm actuator is soft and the bang-bang policy commands targets the
                # arm never fully reaches, so the recorded final target LEADS the achieved state.
                # collect_rsl_rl.py drops the terminal (success) frame; HOLD the last recorded
                # target for a few more control steps so the arm finishes tracking it into the
                # basket -- this completes the place WITHOUT needing the unrecorded terminal action.
                if not success_rep and args_cli.settle_steps > 0:
                    a = torch.as_tensor(actions[-1], device=device)
                    for _ in range(args_cli.settle_steps):
                        _, term, trunc, info = u.eval_step(a)
                        max_z = max(max_z, info["object_z"])
                        grasped_rep = grasped_rep or bool(info["grasped"])
                        if bool(info["success"]):
                            success_rep, succ_via = True, "settle"
                            break
                        if bool(term[0].item()) or bool(trunc[0].item()):
                            break

            n += 1
            n_rl += success_rl
            n_rep += success_rep
            n_grasp += grasped_rep
            n_match += (success_rl == success_rep)
            tag = os.path.relpath(os.path.dirname(f), args_cli.record_dir)
            print(f"  {tag:>20}: RL[succ={int(success_rl)}]  "
                  f"REPLAY[succ={int(success_rep)} via={succ_via} grasp={int(grasped_rep)} lift={max_z - z0:+.3f}]  "
                  f"{'MATCH' if success_rl == success_rep else 'DIFF'}  frames={len(actions)}", flush=True)
        print("=" * 78, flush=True)
        print(f"[replay_init] REPLAY done | RL success {n_rl}/{n} | replay success {n_rep}/{n} "
              f"| replay grasp {n_grasp}/{n} | RL==REPLAY {n_match}/{n}", flush=True)
        print("=" * 78, flush=True)
        if args_cli.json_out:
            with open(args_cli.json_out, "w") as fh:
                json.dump({"n": n, "rl_success": n_rl, "replay_success": n_rep,
                           "replay_grasp": n_grasp, "match": n_match}, fh, indent=2)
            print(f"[replay_init] wrote summary -> {args_cli.json_out}", flush=True)

    if args_cli.mode in ("collect", "both"):
        collect()
    if args_cli.mode in ("replay", "both"):
        replay()

    env.close()
    torch.cuda.empty_cache()
    simulation_app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record init+actions from an rsl_rl policy and replay them to prove replayability.")
    parser.add_argument("--task", type=str, default="EvalPickItUpOneEnv")
    parser.add_argument("--checkpoint", type=str, required=True, help="Absolute path to model_*.pt.")
    parser.add_argument("--num_episodes", type=int, default=20, help="Episodes to collect (collect/both).")
    parser.add_argument("--record_dir", type=str, default="logs/replay_with_init_v9",
                        help="Where init_actions.npz are written (collect) / read from (replay).")
    parser.add_argument("--mode", type=str, default="both", choices=["collect", "replay", "both"])
    parser.add_argument("--eval_fps", type=int, default=60,
                        help="Control rate; must match the dataset fps (v9 = 60). steps = round(120/fps).")
    parser.add_argument("--settle_steps", type=int, default=15,
                        help="After replaying the recorded actions, HOLD the last recorded target "
                             "for up to this many extra control steps so the soft-actuator arm "
                             "finishes tracking it into the basket (collect drops the terminal "
                             "success frame). 0 disables.")
    parser.add_argument("--json_out", type=str, default=None, help="Optional replay-summary JSON path.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--headless", action="store_true", default=True)
    args_cli = parser.parse_args()
    main(args_cli)
