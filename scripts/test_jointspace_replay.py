"""TEST: can we get a SMOOTH joint-target that STILL GRASPS?

Per demo, compare:
  A) original IK replay (env.step on the stored IK-rel actions) -> grasps, but jerky target.
  C) low-pass-filtered IK joint target, replayed via direct joint-PD -> keeps the lead
     (so the grasp should survive) but removes the high-frequency jitter (BC-friendly).

(B = commanding the achieved pose under-leads -> grasp fails; already shown, omitted.)

Run (conda eureka):
    HDF5_USE_FILE_LOCKING=FALSE ~/miniconda3/envs/eureka/bin/python scripts/test_jointspace_replay.py \
        --input_file logs/pb_generated_1000_r003.hdf5 --num 5 --win 5 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default="logs/pb_generated_1000_r003.hdf5")
parser.add_argument("--task", type=str, default="Isaac-Franka-PickBasket-IK-Rel-v0")
parser.add_argument("--num", type=int, default=5)
parser.add_argument("--win", type=int, default=5, help="centered moving-average window (odd) for the low-pass")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch
import gymnasium as gym

import isaaclab_tasks.manager_based.manipulation.pick_basket.config.franka  # noqa: F401
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def jerk(a):
    return float(np.abs(np.diff(np.asarray(a), n=2, axis=0)).mean())


def moving_avg(x, w):
    """Centered moving average along time (axis 0), edge-padded. x: [T, J]."""
    if w <= 1:
        return x.copy()
    pad = w // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    ker = np.ones(w) / w
    return np.stack([np.convolve(xp[:, j], ker, mode="valid") for j in range(x.shape[1])], axis=1)


def main():
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.terminations = None
    env = gym.make(args.task, cfg=env_cfg)
    env_u = env.unwrapped
    scene = env_u.scene
    device = env_u.device
    robot = scene["robot"]
    soup = scene["alphabet_soup"]
    basket = scene["basket"]
    decim = env_u.cfg.decimation
    pdt = env_u.physics_dt
    arm_ids = [robot.find_joints(f"panda_joint{i}")[0][0] for i in range(1, 8)]
    fing_ids = robot.find_joints("panda_finger_joint.*")[0]
    s8 = arm_ids + [fing_ids[0]]

    handler = HDF5DatasetFileHandler()
    handler.open(args.input_file)
    names = list(handler.get_episode_names())[: args.num]

    def in_basket():
        sp = soup.data.root_pos_w[0]; bp = basket.data.root_pos_w[0]; org = scene.env_origins[0]
        xy = float(torch.linalg.norm(sp[:2] - bp[:2]))
        sz = float(sp[2] - org[2]); bz = float(bp[2] - org[2])
        return (xy < 0.08) and (bz + 0.01 < sz < bz + 0.25), xy

    env.reset()
    okA = 0
    totals = {}
    with torch.inference_mode():
        for name in names:
            ep = handler.load_episode(name, device)
            init = ep.data["initial_state"]
            acts = ep.data["actions"]
            acts = acts.detach().cpu().numpy() if hasattr(acts, "cpu") else np.asarray(acts)
            T = acts.shape[0]

            # ---- Pass A: original IK replay, record the IK joint target ----
            env_u.reset_to(init, torch.tensor([0], device=device), is_relative=True)
            tgt_full = []
            for t in range(T):
                env_u.step(torch.tensor(acts[t], dtype=torch.float32, device=device).reshape(1, -1))
                tgt_full.append(robot.data.joint_pos_target[0].detach().cpu().numpy().copy())
            okA_i, xyA = in_basket(); okA += int(okA_i)
            tgt_full = np.asarray(tgt_full)                       # [T, J] full joint target

            # variants: (name, filter the gripper too?, window)
            variants = [("all_w3", True, 3), ("arm_w3", False, 3), ("arm_w5", False, 5)]
            res = {}
            for vname, filt_grip, w in variants:
                lp = moving_avg(tgt_full, w)
                if not filt_grip:                                # keep gripper (finger joints) SHARP
                    lp[:, fing_ids] = tgt_full[:, fing_ids]
                env_u.reset_to(init, torch.tensor([0], device=device), is_relative=True)
                for t in range(T):
                    robot.set_joint_position_target(torch.tensor(lp[t], dtype=torch.float32, device=device).unsqueeze(0))
                    for _ in range(decim):
                        scene.write_data_to_sim(); env_u.sim.step(render=False); scene.update(dt=pdt)
                ok_v, xy_v = in_basket()
                res[vname] = (ok_v, xy_v, jerk(lp[:, s8]))
                totals[vname] = totals.get(vname, 0) + int(ok_v)

            msg = f"[{name}] A(IK): ok={okA_i} jerk={jerk(tgt_full[:, s8]):.4f}"
            for v in res:
                ok_v, xy_v, jk = res[v]
                msg += f"  | {v}: ok={ok_v} xy={xy_v:.3f} jerk={jk:.4f}"
            print(msg, flush=True)

    print(f"\nA original-IK: {okA}/{len(names)}", flush=True)
    for v in totals:
        print(f"  {v}: {totals[v]}/{len(names)} grasp", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
