"""Parallel re-record of MimicGen-generated demos into the ``logs/replay_record`` format.

The multi-env analog of ``record_generated_in_replay_format.py``. That script replays one
generated demo at a time (num_envs=1) and encodes 3 mp4s inline -> ~2 demos/min. This one
runs ``Isaac-Franka-PickBasket-IK-Rel-BatchRecord-v0`` with N envs carrying **TiledCameras**
(one batched render over all envs) and replays demos in **waves of N**: per-env
``reset_to(initial_state)``, step the (length-padded) IK-rel action batch, capture batched
frames, then hand each episode's mp4 encoding to a ``ProcessPoolExecutor`` (true parallel
ffmpeg, off the sim loop). Output is byte-for-byte the same schema as the single-env script
(successful/ + unsuccessful/ + manifest.jsonl + meta.json), so the LeRobot converter is unchanged.

Run (conda eureka):
    HDF5_USE_FILE_LOCKING=FALSE ~/miniconda3/envs/eureka/bin/python scripts/record_generated_parallel.py \
        --input_file logs/pb_generated_1000_r003.hdf5 \
        --output_dir logs/replay_record_mimicgen_r003 \
        --num_envs 32 --fps 20 --enable_cameras --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--task", type=str, default="Isaac-Franka-PickBasket-IK-Rel-BatchRecord-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--max_demos", type=int, default=0, help="0 = all demos in the file")
parser.add_argument("--fps", type=int, default=20)
parser.add_argument(
    "--keep_stride",
    type=int,
    default=3,
    help=(
        "Decimate the recorded frames by this stride (= libero bridge --dwell). The source "
        "demos hold each LIBERO waypoint for DWELL=3 env steps for IK convergence, which "
        "stretches the demo 3x (effective 6.67 Hz) and fills the data with converged plateau "
        "frames (action~=state). Keeping every DWELL-th frame (the FIRST of each group: pose at "
        "the start of the move + the large initial joint target toward the next waypoint) "
        "restores the native 20 Hz rate and the per-step lead, so deploy at eval_fps=20 reaches "
        "each waypoint instead of creeping. 1 = keep all (legacy)."
    ),
)
parser.add_argument(
    "--action_source",
    type=str,
    default="ik_target",
    choices=["ik_target", "achieved_lead"],
    help=(
        "What to store as the BC action. 'ik_target' = the raw IK joint_pos_target (legacy) — "
        "smooth EEF but the joint command thrashes (jerk ~0.05, 4x the achieved motion), which "
        "ACT can't fit and collapses to the mean pose (move-then-freeze). 'achieved_lead' = the "
        "achieved joint_pos LEAD steps in the future for the arm (smooth, jerk ~0.012 ~= the RL "
        "dataset) plus the commanded gripper target (kept sharp). This is the RL-data semantics "
        "(target ~= achieved + small lead) and the achieved path provably grasps."
    ),
)
parser.add_argument(
    "--lead",
    type=int,
    default=2,
    help="achieved_lead: store arm action = achieved joint_pos this many KEPT (20 Hz) steps ahead.",
)
parser.add_argument("--write_workers", type=int, default=6, help="Encoder threads (each drives an ffmpeg subprocess).")
parser.add_argument("--max_pending", type=int, default=24, help="Max in-flight unwritten episodes (RAM backpressure).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks.manager_based.manipulation.pick_basket.config.franka  # noqa: F401
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

BASE_TASK = "pick up the alphabet_soup and put it in the basket"
CONTROLLER = "ik_rel_mimicgen"
GAINS = None
AGENTVIEW_EYE = (1.05, 0.0, 0.55)
AGENTVIEW_TARGET = (0.0, 0.0, 0.08)


def _encode_episode(out_dir, idx, success, frames, states, acts, fps):
    """Write one episode (mp4s + arrays.npz + episode_meta.json) to its split dir.

    Runs in a ThreadPoolExecutor worker; imageio.mimwrite drives an ffmpeg subprocess (the
    encode runs there, off the GIL). Mirrors record_generated_in_replay_format._write_episode.
    Returns the manifest row."""
    import imageio

    split = "successful" if success else "unsuccessful"
    task = BASE_TASK if success else BASE_TASK + " unsuccessful"
    ep_dir = os.path.join(out_dir, split, f"ep{idx:02d}")
    os.makedirs(ep_dir, exist_ok=True)

    video_paths = {}
    for name in ("agentview", "wrist"):
        path = os.path.join(ep_dir, f"{name}.mp4")
        imageio.mimwrite(path, frames[name], fps=fps, macro_block_size=1, codec="libx264")
        video_paths[name] = os.path.relpath(path, out_dir)
    n = min(len(frames["agentview"]), len(frames["wrist"]))
    sbs = [np.concatenate([frames["agentview"][i], frames["wrist"][i]], axis=1) for i in range(n)]
    sbs_path = os.path.join(ep_dir, "sidebyside.mp4")
    imageio.mimwrite(sbs_path, sbs, fps=fps, macro_block_size=1, codec="libx264")
    video_paths["sidebyside"] = os.path.relpath(sbs_path, out_dir)

    states = np.asarray(states, dtype=np.float32)
    acts = np.asarray(acts, dtype=np.float32)
    timestamps = np.arange(len(states), dtype=np.float32) / float(fps)
    np.savez(os.path.join(ep_dir, "arrays.npz"),
             observation_state=states, action=acts, timestamp=timestamps)

    ep_meta = {
        "episode_index": idx, "success": bool(success), "task": task,
        "length": int(len(states)), "fps": fps, "cameras": ["agentview", "wrist"],
        "videos": video_paths, "controller": CONTROLLER, "gains": GAINS,
        "state_dim": 8, "action_dim": 8,
        "state_desc": "robot joint pos (8): 7 arm + 1 gripper (finger_joint1)",
        "action_desc": "IK-controller joint pos target (8): 7 arm + 1 gripper",
    }
    with open(os.path.join(ep_dir, "episode_meta.json"), "w") as f:
        json.dump(ep_meta, f, indent=2)
    return {
        "episode_index": idx, "success": bool(success), "task": task,
        "length": int(len(states)), "ep_dir": os.path.relpath(ep_dir, out_dir),
        "videos": video_paths, "controller": CONTROLLER, "gains": GAINS,
    }


def main():
    N = args.num_envs
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=N)
    env_cfg.terminations = None  # no auto-reset mid-replay
    env = gym.make(args.task, cfg=env_cfg)
    env_u = env.unwrapped
    scene = env_u.scene
    device = env_u.device
    robot = scene["robot"]
    soup = scene["alphabet_soup"]
    basket = scene["basket"]
    agentview = scene["agentview"]
    wrist = scene["wrist"]

    arm_ids = [robot.find_joints(f"panda_joint{i}")[0][0] for i in range(1, 8)]
    finger1_id = robot.find_joints("panda_finger_joint1")[0][0]
    state8_ids = torch.tensor(arm_ids + [finger1_id], device=device)

    handler = HDF5DatasetFileHandler()
    handler.open(args.input_file)
    names = list(handler.get_episode_names())
    if args.max_demos > 0:
        names = names[: args.max_demos]
    n_demos = len(names)

    env.reset()
    origins = scene.env_origins  # [N,3]
    eye = torch.tensor([AGENTVIEW_EYE], device=device) + origins
    tgt = torch.tensor([AGENTVIEW_TARGET], device=device) + origins

    os.makedirs(args.output_dir, exist_ok=True)
    # ThreadPool (NOT ProcessPool): threads share memory (no 157MB/episode pickle copy) and
    # don't re-import this module, so encode workers never re-launch Isaac Sim. Encoding still
    # parallelizes because each imageio.mimwrite drives a separate ffmpeg subprocess.
    writer_pool = ThreadPoolExecutor(max_workers=args.write_workers)
    write_sem = threading.Semaphore(args.max_pending)
    futures, n_success = [], 0

    def capture():
        # pull the latest rendered frame. update() re-reads the annotator buffer (filled by the
        # render inside env.step, or by an explicit sim.render() before the per-wave priming read).
        agentview.update(env_u.physics_dt)
        wrist.update(env_u.physics_dt)
        av = agentview.data.output["rgb"][..., :3].detach().cpu().numpy().astype(np.uint8)  # [N,H,W,3]
        wr = wrist.data.output["rgb"][..., :3].detach().cpu().numpy().astype(np.uint8)
        return av, wr

    n_waves = (n_demos + N - 1) // N
    for w in range(n_waves):
        wave = names[w * N : w * N + N]
        k = len(wave)

        # load this wave's actions; reset each env to its demo's initial_state
        wave_actions = []
        for i, name in enumerate(wave):
            ep = handler.load_episode(name, device)
            a = ep.data["actions"]
            a = a.cpu().numpy() if hasattr(a, "cpu") else np.asarray(a)
            wave_actions.append(np.asarray(a, dtype=np.float32))
            env_u.reset_to(ep.data["initial_state"], torch.tensor([i], device=device), is_relative=True)
        agentview.set_world_poses_from_view(eye, tgt)  # re-aim all (reset re-applies cfg offset)

        lengths = [a.shape[0] for a in wave_actions]
        Tmax = max(lengths)
        buf_av = [[] for _ in range(k)]
        buf_wr = [[] for _ in range(k)]
        buf_st = [[] for _ in range(k)]
        buf_ac = [[] for _ in range(k)]

        # Record convention (s_t, a_t): the state/image are observed BEFORE the action is
        # applied (what a policy sees when choosing a_t), paired with the joint target a_t
        # commanded that step. So we PRIME the post-reset obs (frame 0), then each step:
        # record current obs -> apply action -> read the applied target -> capture next obs.
        env_u.sim.render()  # render the reset state so the cameras hold frame 0
        av_cur, wr_cur = capture()
        st_cur = robot.data.joint_pos[:, state8_ids].detach().cpu().numpy()  # [N,8] pre-action

        for t in range(Tmax):
            a = torch.zeros((N, wave_actions[0].shape[1]), dtype=torch.float32, device=device)
            for i in range(k):
                a[i] = torch.from_numpy(wave_actions[i][min(t, lengths[i] - 1)]).to(device)
            env_u.step(a)  # apply action t
            ac = robot.data.joint_pos_target[:, state8_ids].detach().cpu().numpy()  # [N,8] target applied at t
            # Keep one frame per dwell group (stride = bridge --dwell): the FIRST step of each
            # group, where the state lags at the previous waypoint and the joint target is the
            # large initial command toward the next one. This restores the native 20 Hz rate
            # (one waypoint per kept frame) and the per-step lead; the dropped steps are the
            # converged plateau (action~=state) that otherwise teaches the policy to hover.
            keep = (t % args.keep_stride) == 0
            for i in range(k):
                if t < lengths[i] and keep:
                    # pre-action obs (captured before this step) paired with the action applied
                    buf_av[i].append(av_cur[i]); buf_wr[i].append(wr_cur[i])
                    buf_st[i].append(st_cur[i].copy()); buf_ac[i].append(ac[i].copy())
            # capture obs after step t == the pre-action obs for step t+1
            av_cur, wr_cur = capture()
            st_cur = robot.data.joint_pos[:, state8_ids].detach().cpu().numpy()

        # success per env + submit encoding
        sp = soup.data.root_pos_w.detach().cpu().numpy()    # [N,3]
        bp = basket.data.root_pos_w.detach().cpu().numpy()
        org = origins.detach().cpu().numpy()
        for i in range(k):
            xy = float(np.linalg.norm(sp[i, :2] - bp[i, :2]))
            sz = float(sp[i, 2] - org[i, 2]); bz = float(bp[i, 2] - org[i, 2])
            success = (xy < 0.08) and (bz + 0.01 < sz < bz + 0.25)
            n_success += int(success)
            ep_idx = w * N + i
            states = np.asarray(buf_st[i], dtype=np.float32)        # [T,8] achieved joint_pos
            tgt = np.asarray(buf_ac[i], dtype=np.float32)           # [T,8] IK joint_pos_target
            if args.action_source == "achieved_lead":
                # arm action = achieved joint_pos LEAD kept-steps ahead (smooth + leads);
                # gripper = commanded target (sharp binary open/close, no PD lag).
                T = states.shape[0]
                fut_idx = np.minimum(np.arange(T) + args.lead, T - 1)
                acts = states[fut_idx].copy()
                acts[:, 7] = tgt[:, 7]
            else:
                acts = tgt
            write_sem.acquire()
            fr = {"agentview": np.asarray(buf_av[i]), "wrist": np.asarray(buf_wr[i])}
            fut = writer_pool.submit(_encode_episode, args.output_dir, ep_idx, success,
                                     fr, states, acts, args.fps)
            fut.add_done_callback(lambda f: write_sem.release())
            futures.append(fut)
            print(f"[ep{ep_idx}] {wave[i]} success={success} steps={lengths[i]} xy={xy:.3f}", flush=True)
        print(f"  -- wave {w + 1}/{n_waves} done ({k} demos) --", flush=True)

    print(f"[parallel] draining {len(futures)} pending episode writes...", flush=True)
    manifest = [f.result() for f in futures]
    writer_pool.shutdown(wait=True)
    manifest.sort(key=lambda r: r["episode_index"])
    _write_dataset_meta(args.output_dir, manifest, n_demos, n_success, args.fps)
    print(f"\nDone. {n_demos} demos -> {args.output_dir}  ({n_success} successful)", flush=True)
    env.close()


def _write_dataset_meta(out_dir, manifest, n, n_success, fps):
    with open(os.path.join(out_dir, "manifest.jsonl"), "w") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")
    meta = {
        "fps": fps, "cameras": ["agentview", "wrist"],
        "image_keys": ["observation.images.agentview", "observation.images.wrist"],
        "state_dim": 8, "action_dim": 8,
        "state_desc": "robot joint_pos (8): 7 arm + 1 gripper",
        "action_desc": "IK-controller target joint_pos (8)",
        "controller": CONTROLLER, "gains": GAINS, "base_task": BASE_TASK,
        "num_episodes": n, "num_successful": n_success, "num_unsuccessful": n - n_success,
        "source": "MimicGen-generated (Isaac-Franka-PickBasket-IK-Rel-Mimic-v0), parallel record",
        "note": "Convert with scripts/convert_to_lerobot.py in a dedicated lerobot env.",
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
    simulation_app.close()
