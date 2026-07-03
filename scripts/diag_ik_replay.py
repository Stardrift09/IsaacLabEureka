"""Diagnose why IK-action replay of pick_basket source demos fails annotation.

Replays the recorded 7-D IK delta actions from a source HDF5 through the *same*
env path annotate_demos uses (env.step on Isaac-Franka-PickBasket-IK-Rel-Mimic-v0,
reset_to(initial_state, is_relative=True)), and reports per demo:

  - EEF drift: replayed measured EEF vs the intended path reconstructed by
    cumulating scale*delta from the initial EEF (gross IK divergence?).
  - grasp+lift: did object_grasped_and_lifted ever fire? peak soup z_local.
  - place: final soup<->basket xy_dist, soup z_local, basket z_local, and the
    object_in_basket bool (the success_term annotate checks).

Run (conda eureka):
    ~/miniconda3/envs/eureka/bin/python -u scripts/diag_ik_replay.py \
        --input_file /tmp/pb_src_raw.hdf5 --num_demos 3 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default="/tmp/pb_src_raw.hdf5")
parser.add_argument("--task", type=str, default="Isaac-Franka-PickBasket-IK-Rel-Mimic-v0")
parser.add_argument("--num_demos", type=int, default=3)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch
import gymnasium as gym

import isaaclab_tasks.manager_based.manipulation.pick_basket.config.franka  # noqa: F401
import isaaclab_mimic.envs  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.utils.datasets import HDF5DatasetFileHandler

IK_SCALE = 0.5


def main():
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env = gym.make(args.task, cfg=env_cfg)
    env_u = env.unwrapped
    scene = env_u.scene
    soup = scene["alphabet_soup"]
    basket = scene["basket"]
    ee_frame = scene["ee_frame"]
    robot = scene["robot"]
    grip_ids, _ = robot.find_joints(env_u.cfg.gripper_joint_names)

    env.reset()  # satisfy the order-enforcing gym wrapper once

    handler = HDF5DatasetFileHandler()
    handler.open(args.input_file)
    names = list(handler.get_episode_names())[: args.num_demos]

    for name in names:
        ep = handler.load_episode(name, env_u.device)
        initial_state = ep.data["initial_state"]
        actions = ep.data["actions"]  # (T,7) numpy or tensor
        if hasattr(actions, "cpu"):
            actions = actions.cpu().numpy()
        actions = np.asarray(actions)

        env_u.sim.reset()
        env_u.reset_to(initial_state, None, is_relative=True)
        ee_frame.update(env_u.physics_dt)

        intended = ee_frame.data.target_pos_w[0, 0].clone()  # reconstruct intended path
        max_lift_local = -1e9
        ever_grasped_lifted = False
        drift_max = 0.0

        for t in range(actions.shape[0]):
            a = torch.tensor(actions[t], dtype=torch.float32, device=env_u.device).reshape(1, -1)
            env_u.step(a)
            ee_frame.update(env_u.physics_dt)

            intended = intended + IK_SCALE * a[0, :3]
            meas = ee_frame.data.target_pos_w[0, 0]
            drift_max = max(drift_max, float(torch.linalg.norm(meas - intended)))

            soup_z_local = float(soup.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
            max_lift_local = max(max_lift_local, soup_z_local)

            ee_pos_w = ee_frame.data.target_pos_w[0, 0]
            pose_diff = float(torch.linalg.norm(soup.data.root_pos_w[0] - ee_pos_w))
            g0 = abs(float(robot.data.joint_pos[0, grip_ids[0]]) - env_u.cfg.gripper_open_val) > env_u.cfg.gripper_threshold
            g1 = abs(float(robot.data.joint_pos[0, grip_ids[1]]) - env_u.cfg.gripper_open_val) > env_u.cfg.gripper_threshold
            if pose_diff < 0.08 and g0 and g1 and soup_z_local > 0.12:
                ever_grasped_lifted = True

        sp = soup.data.root_pos_w[0]
        bp = basket.data.root_pos_w[0]
        xy = float(torch.linalg.norm(sp[:2] - bp[:2]))
        sz = float(sp[2] - scene.env_origins[0, 2])
        bz = float(bp[2] - scene.env_origins[0, 2])
        in_basket = (xy < 0.08) and (bz + 0.01 < sz < bz + 0.10 + 0.15)
        print(
            f"[{name}] steps={actions.shape[0]} drift_max={drift_max:.3f}m "
            f"grasp+lift_ever={ever_grasped_lifted} peak_soup_z={max_lift_local:.3f} "
            f"| FINAL xy_dist={xy:.3f} soup_z={sz:.3f} basket_z={bz:.3f} in_basket={in_basket}"
        )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
