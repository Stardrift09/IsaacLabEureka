"""Re-record MimicGen-generated demos into the original ``logs/replay_record`` format.

MimicGen emits IK end-effector-action HDF5 (initial_state + 7-D actions). The
existing LeRobot converter expects per-episode dirs with ``arrays.npz``
(observation_state[T,8], action[T,8], timestamp) + agentview/wrist mp4s +
episode_meta.json, plus a top-level manifest.jsonl / meta.json (see
logs/replay_record/). This script replays each generated demo through the
camera-equipped pick-basket env and writes that exact format.

Run (conda eureka):
    python scripts/record_generated_in_replay_format.py \
        --input_file /tmp/pb_generated.hdf5 \
        --output_dir logs/replay_record_mimicgen \
        --enable_cameras --headless
"""

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default="/tmp/pb_generated.hdf5")
parser.add_argument("--output_dir", type=str, default="logs/replay_record_mimicgen")
parser.add_argument("--task", type=str, default="Isaac-Franka-PickBasket-IK-Rel-Record-v0")
parser.add_argument("--max_demos", type=int, default=0, help="0 = all demos in the file")
parser.add_argument("--fps", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import imageio
import numpy as np
import torch
import gymnasium as gym

import isaaclab_tasks.manager_based.manipulation.pick_basket.config.franka  # noqa: F401
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

BASE_TASK = "pick up the alphabet_soup and put it in the basket"
CONTROLLER = "ik_rel_mimicgen"
GAINS = None  # IK-relative control (not joint-PD); kept for schema parity with replay_record


def main():
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.terminations = None  # no auto-reset during replay
    env = gym.make(args.task, cfg=env_cfg)
    env_u = env.unwrapped
    scene = env_u.scene
    device = env_u.device
    robot = scene["robot"]
    soup = scene["alphabet_soup"]
    basket = scene["basket"]
    agentview = scene["agentview"]
    wrist = scene["wrist"]

    # 9 -> 8 joint mapping: 7 arm joints (ordered) + finger_joint1
    arm_ids = [robot.find_joints(f"panda_joint{i}")[0][0] for i in range(1, 8)]
    finger1_id = robot.find_joints("panda_finger_joint1")[0][0]
    state8_ids = torch.tensor(arm_ids + [finger1_id], device=device)

    handler = HDF5DatasetFileHandler()
    handler.open(args.input_file)
    names = list(handler.get_episode_names())
    if args.max_demos > 0:
        names = names[: args.max_demos]

    env.reset()
    # Aim the static agentview camera with a look-at (matches the original
    # replay_pick_it_up_one_env._aim_agentview); the cfg offset quat is overridden.
    AGENTVIEW_EYE = (1.05, 0.0, 0.55)
    AGENTVIEW_TARGET = (0.0, 0.0, 0.08)
    _origin = scene.env_origins[0]
    _eye = torch.tensor([AGENTVIEW_EYE], device=device) + _origin
    _tgt = torch.tensor([AGENTVIEW_TARGET], device=device) + _origin
    agentview.set_world_poses_from_view(_eye, _tgt)

    os.makedirs(args.output_dir, exist_ok=True)
    manifest, n_success = [], 0

    def grab(cam):
        rgb = cam.data.output["rgb"][0]  # (H,W,3/4)
        rgb = rgb.detach().cpu().numpy()
        return rgb[..., :3].astype(np.uint8)

    for ep_idx, name in enumerate(names):
        ep = handler.load_episode(name, device)
        initial_state = ep.data["initial_state"]
        actions = ep.data["actions"]
        if hasattr(actions, "cpu"):
            actions = actions.cpu().numpy()
        actions = np.asarray(actions, dtype=np.float32)

        env_u.reset_to(initial_state, None, is_relative=True)
        agentview.set_world_poses_from_view(_eye, _tgt)  # re-aim (reset may re-apply cfg offset)

        # Record convention (s_t, a_t): state/image observed BEFORE the action is applied
        # (what a policy sees when choosing a_t), paired with the joint target a_t commanded
        # that step. Prime the post-reset obs (frame 0), then per step: record current obs ->
        # apply action -> read applied target -> capture next obs.
        states, acts, frames = [], [], {"agentview": [], "wrist": []}
        env_u.sim.render()  # render the reset state so the cameras hold frame 0
        st_cur = robot.data.joint_pos[0, state8_ids].detach().cpu().numpy()
        av_cur, wr_cur = grab(agentview), grab(wrist)
        for t in range(actions.shape[0]):
            a = torch.tensor(actions[t], dtype=torch.float32, device=device).reshape(1, -1)
            env_u.step(a)  # apply action t
            states.append(st_cur)                                                    # pre-action obs
            acts.append(robot.data.joint_pos_target[0, state8_ids].detach().cpu().numpy())  # target applied at t
            frames["agentview"].append(av_cur)
            frames["wrist"].append(wr_cur)
            # capture obs after step t == pre-action obs for step t+1
            st_cur = robot.data.joint_pos[0, state8_ids].detach().cpu().numpy()
            av_cur, wr_cur = grab(agentview), grab(wrist)

        # success = soup in basket (matches object_in_basket / bridge check)
        sp = soup.data.root_pos_w[0]
        bp = basket.data.root_pos_w[0]
        org = scene.env_origins[0]
        xy = float(torch.linalg.norm(sp[:2] - bp[:2]))
        sz = float(sp[2] - org[2]); bz = float(bp[2] - org[2])
        success = (xy < 0.08) and (bz + 0.01 < sz < bz + 0.25)
        n_success += int(success)

        _write_episode(args.output_dir, ep_idx, success, frames, states, acts, args.fps, manifest)
        print(f"[{name}] success={success} steps={len(states)} xy={xy:.3f}", flush=True)

    _write_dataset_meta(args.output_dir, manifest, len(names), n_success, args.fps)
    print(f"\nDone. {len(names)} demos -> {args.output_dir}  ({n_success} successful)")
    env.close()


def _write_episode(out_dir, idx, success, frames, states, acts, fps, manifest):
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
    manifest.append({
        "episode_index": idx, "success": bool(success), "task": task,
        "length": int(len(states)), "ep_dir": os.path.relpath(ep_dir, out_dir),
        "videos": video_paths, "controller": CONTROLLER, "gains": GAINS,
    })


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
        "controller": CONTROLLER, "gains": GAINS,
        "base_task": BASE_TASK,
        "num_episodes": n, "num_successful": n_success, "num_unsuccessful": n - n_success,
        "source": "MimicGen-generated (Isaac-Franka-PickBasket-IK-Rel-Mimic-v0)",
        "note": "Convert with scripts/convert_to_lerobot.py in a dedicated lerobot env.",
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
    simulation_app.close()
