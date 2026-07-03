"""Parallel LeRobot-dataset collection from a trained rsl_rl policy.

The multi-env analog of ``collect_rsl_rl.py``. Runs ``BatchEvalPickItUpClean`` with many
envs, each carrying TiledCameras (agentview + wrist) and initialized from a deterministic
grid file (``--grid_file``, built by ``scripts/make_init_grid.py``) so coverage is even and
each grid entry is collected once. Per env we buffer states/actions/frames; on each env's
episode end we **save it iff it succeeded** (task termination) and **drop failures**. Output
is the same layout as ``collect_rsl_rl.py`` (``successful/`` + ``manifest.jsonl`` +
``meta.json``), so the existing LeRobot converter is unchanged.

Action recorded = the commanded absolute joint target (``_target8``) -- the only
representation that replays (see collect_rsl_rl.py docstring). Run in conda ``eureka``:

    ~/miniconda3/envs/eureka/bin/python -u scripts/collect_rsl_rl_parallel.py \
        --checkpoint <.../model_*.pt> --grid_file logs/init_grids/grid_v1.npz \
        --num_envs 64 --cam_w 256 --record_dir logs/collect_parallel/v1 --headless
"""

import argparse

from isaaclab_eureka.utils import get_freest_gpu


def _encode_episode(record_dir, idx, cam_frames, states, actions, fps, base_task, controller, gains):
    """Write one successful episode (mp4s + arrays.npz + episode_meta.json) to disk.

    Module-level + plain-data args so it runs in a ProcessPoolExecutor worker (true parallel
    ffmpeg, bypassing the GIL). Mirrors ReplayPickItUpOneEnv._write_episode_recording for the
    successful split. Returns the manifest fields {task, ep_dir, videos}."""
    import json
    import os

    import imageio
    import numpy as np

    ep_dir = os.path.join(record_dir, "successful", f"ep{idx:02d}")
    os.makedirs(ep_dir, exist_ok=True)
    cam_names = list(cam_frames.keys())
    video_paths = {}
    for name in cam_names:
        path = os.path.join(ep_dir, f"{name}.mp4")
        imageio.mimwrite(path, cam_frames[name], fps=fps, macro_block_size=1, codec="libx264")
        video_paths[name] = os.path.relpath(path, record_dir)
    if len(cam_names) == 2:
        a, b = cam_frames[cam_names[0]], cam_frames[cam_names[1]]
        n = min(len(a), len(b))
        sbs = [np.concatenate([a[i], b[i]], axis=1) for i in range(n)]
        sp = os.path.join(ep_dir, "sidebyside.mp4")
        imageio.mimwrite(sp, sbs, fps=fps, macro_block_size=1, codec="libx264")
        video_paths["sidebyside"] = os.path.relpath(sp, record_dir)
    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    ts = np.arange(len(states), dtype=np.float32) / float(fps)
    np.savez(os.path.join(ep_dir, "arrays.npz"), observation_state=states, action=actions, timestamp=ts)
    meta = {
        "episode_index": idx, "success": True, "task": base_task, "length": int(len(states)),
        "fps": fps, "cameras": cam_names, "videos": video_paths, "controller": controller, "gains": gains,
        "state_dim": int(states.shape[1]) if states.ndim == 2 else 0,
        "action_dim": int(actions.shape[1]) if actions.ndim == 2 else 0,
        "state_desc": "robot joint pos (8): 7 arm + 1 gripper (finger_joint1)",
        "action_desc": "demo target joint pos (8): 7 arm + 1 gripper",
    }
    with open(os.path.join(ep_dir, "episode_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return {"task": base_task, "ep_dir": os.path.relpath(ep_dir, record_dir), "videos": video_paths}


def main(args_cli):
    from isaaclab.app import AppLauncher

    device = args_cli.device
    if device == "cuda":
        device = f"cuda:{get_freest_gpu()}"
    app_launcher = AppLauncher(headless=args_cli.headless, enable_cameras=True, device=device)
    simulation_app = app_launcher.app

    import multiprocessing as mp
    import os
    import threading
    from concurrent.futures import ProcessPoolExecutor

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import numpy as np
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunner

    task = args_cli.task
    env_cfg = parse_env_cfg(task, device=device, num_envs=args_cli.num_envs)
    env_cfg.sim.device = device
    env_cfg.cam_w = env_cfg.cam_h = args_cli.cam_w
    # Re-render the RTX sensors after each reset so the temporal denoiser settles on the new
    # (teleported) state before we capture -- otherwise the first frame(s) of every episode are
    # blurry/ghosted. Applied on the in-step auto-reset too (DirectRLEnv.step).
    env_cfg.num_rerenders_on_reset = args_cli.rerenders
    # Spread envs far apart so neighbor envs don't contaminate each env's render via global
    # illumination / direct visibility (at the default 3.0 spacing the wrist darkens and the
    # agentview shows neighbors -> a train/eval domain shift vs single-env inference). At ~20 m
    # each env renders identically to a single-env scene. Free (just repositions on the ground).
    env_cfg.scene.env_spacing = args_cli.env_spacing
    # Anti-aliasing mode: the default temporal AA (DLSS/TAA) needs frame history, so right after
    # a reset the first frames are blurry/upscaled. FXAA (spatial) or DLAA (full-res) are sharp
    # from frame 0. Set on the sim RenderCfg before the sim is created.
    if args_cli.antialiasing:
        env_cfg.sim.render.antialiasing_mode = args_cli.antialiasing
    if args_cli.grid_file:
        env_cfg.init_grid_file = args_cli.grid_file
    env = gym.make(task, cfg=env_cfg)

    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = device
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(args_cli.checkpoint, map_location=device)
    policy = runner.get_inference_policy(device=device)
    u = env.unwrapped
    u.REPLAY_CONTROLLER = "rsl_rl"

    assert u._grid_active(), "no init grid loaded -- pass --grid_file (cfg.init_grid_file empty)"
    n_grid = u._grid_N
    n_env = u.num_envs
    os.makedirs(args_cli.record_dir, exist_ok=True)

    # episode horizon cap (mirror collect_rsl_rl.py): shorten episode_length_s so non-success
    # rollouts truncate quickly. Set BEFORE reading max_episode_length.
    if args_cli.max_frames and args_cli.max_frames > 0:
        control_dt = u.cfg.sim.dt * u.cfg.decimation
        u.cfg.episode_length_s = args_cli.max_frames * control_dt
    max_steps = int(u.max_episode_length) + 5
    rec_fps = u._recording_fps(args_cli.record_every * u.cfg.decimation)
    cams = list(u._record_cams.keys())
    print(f"[collect_parallel] task={task} num_envs={n_env} grid={n_grid} cam={args_cli.cam_w} "
          f"fps={rec_fps} max_steps={max_steps} -> {args_cli.record_dir}", flush=True)

    # RslRlVecEnvWrapper.__init__ already called env.reset() (advancing the grid cursor); rewind
    # the cursor so OUR reset below dispatches entries 0..n_env-1 cleanly (no wasted entries).
    u._grid_cursor = 0
    u._grid_exhausted = False
    obs, _ = env.reset()
    u._aim_agentview_all()

    # ---- per-env rolling buffers ----
    buf_frames = [{c: [] for c in cams} for _ in range(n_env)]
    buf_states = [[] for _ in range(n_env)]
    buf_actions = [[] for _ in range(n_env)]
    step_in_ep = [0] * n_env
    grasped_ever = [False] * n_env

    # ---- grid bookkeeping: each entry collected ONCE; post-exhaustion clamped dups dropped ----
    entry_started = np.zeros(n_grid, dtype=bool)
    ep_grid_idx = u._grid_idx.detach().cpu().numpy().copy()  # [n_env]
    is_dup = np.zeros(n_env, dtype=bool)
    for e in range(n_env):
        gi = int(ep_grid_idx[e])
        if entry_started[gi]:
            is_dup[e] = True
        else:
            entry_started[gi] = True

    results = []
    ep_counter = 0
    n_done_eps = 0

    # Encode/write episodes in WORKER PROCESSES (true parallel ffmpeg; ThreadPool can't, the
    # imageio/ffmpeg path is GIL-bound). 'spawn' avoids forking the CUDA-initialized parent.
    # A semaphore bounds in-flight (unwritten) episodes to cap memory (~12MB of frames each).
    writer_pool = ProcessPoolExecutor(max_workers=args_cli.write_workers, mp_context=mp.get_context("spawn"))
    write_sem = threading.Semaphore(args_cli.max_pending)
    base_task = f"pick up the {u.target_object_name} and put it in the {u.target_site_name}"
    gains = u._gains_dict()

    while True:
        # ---- PAIRING (critical for BC) ----
        # The policy learns action = f(image, state); at eval it observes obs_t (achieved
        # state + image BEFORE acting) and must emit the target to apply NOW. So a dataset
        # row must be (image_t, state_t, T_t): the observation at t paired with the action
        # decided at t. We therefore capture image+state at the TOP of the loop (= obs_t,
        # the current sim configuration before stepping) and read the applied target T_t
        # AFTER the step (robot_dof_targets is set by _pre_physics_step from `actions` and
        # is unchanged for in-episode envs). Capturing frame/state AFTER the step instead
        # would store (image_{t+1}, state_{t+1}, T_t) -- the label lags one control step,
        # so the policy learns to re-command the target that produced the CURRENT state
        # and stalls. The near-bang-bang target swings ~0.12 rad/joint/step, so that lag
        # is a ~7-degree systematic error every step.
        frames = u._capture_frames_all()                  # image_t  {cam: [n_env,H,W,3] uint8}
        states = u._state8_all().detach().cpu().numpy()   # state_t  [n_env, 8]
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

        targets = u._target8_all().detach().cpu().numpy()  # T_t = target applied at obs_t [n_env,8]
        success = u.reset_terminated.detach().cpu().numpy().astype(bool)  # this step's task success
        grasped = u.grasped.detach().cpu().numpy().astype(bool)
        dones_np = dones.detach().cpu().numpy().astype(bool)
        new_grid_idx = u._grid_idx.detach().cpu().numpy()  # post-auto-reset (done envs advanced)

        for e in range(n_env):
            done_e = bool(dones_np[e]) or step_in_ep[e] >= max_steps
            if not done_e:
                if step_in_ep[e] % args_cli.record_every == 0:
                    for c in cams:
                        buf_frames[e][c].append(frames[c][e])
                    buf_states[e].append(states[e].copy())
                    buf_actions[e].append(targets[e].copy())
                grasped_ever[e] = grasped_ever[e] or bool(grasped[e])
                step_in_ep[e] += 1
                continue

            # ---- episode boundary for env e (env already auto-reset inside env.step) ----
            if success[e] and not is_dup[e] and len(buf_states[e]) > 0:
                stats = {
                    "episode": ep_counter, "controller": u.REPLAY_CONTROLLER,
                    "frames": len(buf_states[e]), "terminated_step": -1,
                    "success": True, "grasped": grasped_ever[e],
                    "grid_idx": int(ep_grid_idx[e]),
                }
                # stack per-cam frame lists into arrays and hand to a worker process;
                # block here only if too many episodes are already awaiting write.
                write_sem.acquire()
                cam_arr = {c: np.asarray(buf_frames[e][c]) for c in cams}
                fut = writer_pool.submit(
                    _encode_episode, args_cli.record_dir, ep_counter, cam_arr,
                    np.asarray(buf_states[e]), np.asarray(buf_actions[e]),
                    rec_fps, base_task, u.REPLAY_CONTROLLER, gains)
                fut.add_done_callback(lambda f: write_sem.release())
                stats["_future"] = fut
                results.append(stats)
                ep_counter += 1
            if not is_dup[e]:
                n_done_eps += 1
                if n_done_eps % 200 == 0:
                    print(f"  done={n_done_eps}/{n_grid}  saved={ep_counter}", flush=True)

            # reset env e's buffers + advance to the entry the auto-reset assigned
            buf_frames[e] = {c: [] for c in cams}
            buf_states[e] = []
            buf_actions[e] = []
            step_in_ep[e] = 0
            grasped_ever[e] = False
            gi = int(new_grid_idx[e])
            ep_grid_idx[e] = gi
            if entry_started[gi]:
                is_dup[e] = True          # clamped duplicate after exhaustion -> will be dropped
            else:
                entry_started[gi] = True
                is_dup[e] = False

        # stop once every grid entry has been started AND all envs are now on duplicates
        # (i.e. all real episodes have finished and flushed)
        if u._grid_exhausted and bool(is_dup.all()):
            break
        if args_cli.max_episodes and ep_counter >= args_cli.max_episodes:
            break

    # drain async writers, merge each writer's returned paths (task/ep_dir/videos) into stats
    print(f"[collect_parallel] draining {len(results)} pending episode writes...", flush=True)
    for r in results:
        r.update(r.pop("_future").result())
    writer_pool.shutdown(wait=True)
    if results:
        u._write_dataset_manifest(args_cli.record_dir, results, len(results), rec_fps)
    print(f"[collect_parallel] DONE | saved {len(results)} successful episodes "
          f"(of {n_done_eps} grid entries attempted) -> {args_cli.record_dir}", flush=True)
    print(f"[collect_parallel] coverage: grid_N={n_grid} entries_started={int(entry_started.sum())} "
          f"cursor={u._grid_cursor} exhausted={u._grid_exhausted}", flush=True)

    env.close()
    torch.cuda.empty_cache()
    simulation_app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel LeRobot collection from an rsl_rl policy.")
    parser.add_argument("--task", type=str, default="BatchEvalPickItUpClean")
    parser.add_argument("--checkpoint", type=str, required=True, help="Absolute path to model_*.pt.")
    parser.add_argument("--grid_file", type=str, required=True, help="Init grid npz (make_init_grid.py).")
    parser.add_argument("--num_envs", type=int, default=64, help="Parallel envs (cameras are the bottleneck).")
    parser.add_argument("--cam_w", type=int, default=256, help="Square camera resolution (cam_w=cam_h).")
    parser.add_argument("--env_spacing", type=float, default=20.0,
                        help="Distance (m) between envs. Large (~20) so neighbor envs don't leak "
                             "into each env's camera via GI/visibility -> images match single-env eval.")
    parser.add_argument("--rerenders", type=int, default=0,
                        help="num_rerenders_on_reset: RTX re-renders after each reset (left at 0).")
    parser.add_argument("--antialiasing", type=str, default=None,
                        choices=["Off", "FXAA", "DLSS", "TAA", "DLAA"],
                        help="RenderCfg anti-aliasing. Default temporal (DLSS/TAA) is blurry at "
                             "episode start; FXAA or DLAA are sharp from frame 0.")
    parser.add_argument("--record_dir", type=str, default="logs/collect_parallel/v1")
    parser.add_argument("--record_every", type=int, default=1, help="Record every Nth control step (fps=60/N).")
    parser.add_argument("--max_frames", type=int, default=100, help="Truncate each episode after N control steps.")
    parser.add_argument("--max_episodes", type=int, default=0, help="Optional cap on saved episodes (0=all grid).")
    parser.add_argument("--write_workers", type=int, default=8,
                        help="Threads that encode/write episodes (mp4+npz) off the sim loop.")
    parser.add_argument("--max_pending", type=int, default=128,
                        help="Max in-flight unwritten episodes (memory backpressure; ~12MB each).")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--headless", action="store_true", default=True)
    args_cli = parser.parse_args()
    main(args_cli)
