# Physics-faithful replay → gain sweep → LIBERO-view recording → LeRobot dataset

End-to-end pipeline for replaying LIBERO demos through Isaac Sim physics (not
kinematic teleport), finding the controller/gains that best reproduce them,
recording two LIBERO-style camera videos, and exporting a LeRobot v3 dataset.

Task used throughout: **`pick up the alphabet_soup and put it in the basket`**
(50 LIBERO demos). Everything runs in the **`eureka`** conda env except the
LeRobot conversion (separate `~/lerobot/.venv`).

---

## 1. Why

`TestPickItUp.run_replay` is **kinematic** — it teleports both the arm joints and
the object every step, so it can't tell whether a demo trajectory physically
grasps/lifts the object. We want a **physics-faithful** replay: drive the arm
with a controller and let the object obey physics, so grasp/lift/place must
actually emerge. Then judge controllers/gains by the `terminated` (task-success)
flag, watch the result, and turn it into a training dataset.

## 2. Controllers & the kp/kd question

LIBERO demos store **states only** (joint/object poses) — no controller gains.
LIBERO generates them with robosuite **`OSC_POSE`**, `control_freq=20`, default
**`kp=150, damping_ratio=1`** (`LIBERO/.../env_wrapper.py`,
`robosuite/.../osc_pose.json`). Those are **Cartesian** impedance gains, not
joint PD, so there's no joint kp/kd to copy.

Two replay controllers (env `ReplayPickItUpOneEnv`, flag `REPLAY_CONTROLLER`):

| mode | how |
|------|-----|
| `joint_pd` | arm driven by joint position targets via `ImplicitActuator` PD (`set_joint_position_target`). Gains overridable: `PD_KP`, `PD_KD`. |
| `osc` | arm driven by IsaacLab `OperationalSpaceController` toward per-frame EE-pose targets (FK'd from demo joints). Gains = LIBERO's `OSC_KP=150`, `OSC_DAMPING_RATIO=1`. |

No interpolation: each demo frame (20 Hz) is held as the target for
`REPLAY_STEPS_PER_FRAME=6` physics steps (120 Hz). Object is never teleported.

Core file: [`replay_pick_it_up_one_env.py`](IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/franka_cabinet/replay_pick_it_up_one_env.py)
(`ReplayPickItUpOneEnv(TestPickItUp)`, registered as `ReplayPickItUpOneEnv`).

## 3. Scripts

| script | what |
|--------|------|
| [`scripts/replay_one_env.py`](scripts/replay_one_env.py) | one episode, one env; prints a report + TensorBoard. `--episode_idx`, `--headless`. |
| [`scripts/replay_all.py`](scripts/replay_all.py) | all 50 episodes of the current config; `summary.txt`/`summary.csv`. |
| [`scripts/replay_sweep.py`](scripts/replay_sweep.py) | sweeps many controller/gain configs (soft→rigid), ranks by success. `--episodes N`. |
| [`scripts/replay_record.py`](scripts/replay_record.py) | records two LIBERO cameras over all episodes of one config (see §5). |
| [`~/lerobot/scripts/convert_replay_to_lerobot.py`](file:///home/shaotongchen/lerobot/scripts/convert_replay_to_lerobot.py) | recorded dir → LeRobot v3 dataset (run in `~/lerobot/.venv`). |

The replay env returns per-episode stats (`terminated_step`, `grasped_step`,
`lift`, `mean_track_err`, `success`); `run_replay_all` aggregates them.

## 4. Gain sweep results (25 configs × 50 episodes)

Ranked by task-success rate. Output: `logs/replay_sweep/sweep_ranking.txt` + `sweep_results.csv`.

| rank | config | success | mean lift | joint track err |
|---|---|---|---|---|
| 1 | **pd_kp3000_kd260** | **48/50** | +0.177 | **0.029** |
| 2 | osc_kp800_dr1.0 | 47/50 | +0.172 | 0.306 |
| 4 | osc_kp600_dr1.0 | 46/50 | +0.167 | 0.306 |
| 7 | pd_kp2000_kd200 | 45/50 | +0.166 | 0.035 |
| 12 | osc_kp150_dr1.0 (LIBERO default) | 44/50 | +0.158 | 0.309 |
| 25 | pd_kp100_kd20 | 0/50 | +0.000 | 0.151 |

- **Stiff joint-PD wins** (kp3000): best success + ~10× lower joint-tracking error,
  and cheaper than OSC (no jacobian/mass solve).
- joint-PD is strongly monotonic — soft gains (≤ kp600, incl. the env default)
  fail to lift. OSC is robust and flat across kp 100–800.

## 5. Recording (two LIBERO-style cameras, 20 fps)

`scripts/replay_record.py` launches with `enable_cameras=True`, sets
`cfg.record_cameras=True`, and records env_0 at 256² for every episode:

- **agentview** — external front view (`set_world_poses_from_view`).
- **wrist (eye_in_hand)** — rigidly on `panda_hand` with **LIBERO's robosuite
  mount**: `pos=(0.05,0,0)`, `quat=(0,0.7071,0.7071,0)`, `fovy=75°`
  (`convention="opengl"`). Renders OUR physics replay but matches the LIBERO view.
- Debug-vis markers (grasp arrows, per-asset frame triads, origin visualizer) are
  disabled so videos are clean.

```
python scripts/replay_record.py --episodes 50                      # joint_pd kp3000 (default)
python scripts/replay_record.py --controller osc --osc_kp 800
python scripts/replay_record.py --controller joint_pd --kp 2000 --kd 200
```

Output (`logs/replay_record/`), success/failure split by the `terminated` flag:
```
meta.json                 # fps, cameras, dims, controller, gains, base task
manifest.jsonl            # one line per episode
successful/epNN/   agentview.mp4  wrist.mp4  sidebyside.mp4  arrays.npz  episode_meta.json
unsuccessful/epNN/ ...    # task string gets " unsuccessful" suffix
```
`arrays.npz`: `observation_state` [T,9], `action` [T,9] (demo joint targets),
`timestamp` [T]. (Last run: 48 successful, 2 unsuccessful.)

## 6. LeRobot v3 dataset

Do **not** install lerobot into `eureka` — it pulls numpy≥2 and breaks Isaac Sim.
Convert in the separate `~/lerobot/.venv` (lerobot 0.5.2):

```
~/lerobot/.venv/bin/python ~/lerobot/scripts/convert_replay_to_lerobot.py \
  --in /home/shaotongchen/workspace_eureka/IsaacLabEureka/logs/replay_record \
  --repo-id local/pickitup_replay \
  --root ~/lerobot/datasets/pickitup_replay --overwrite
```

- Converts **successful episodes only** (`--include-unsuccessful` to add fails).
- **State/action simplified 9 → 8**: 7 arm joints + 1 gripper (the two mirrored
  finger joints collapse to one).
- Schema mirrors `~/lerobot/datasets/libero/meta/info.json` (codebase **v3.0**,
  `robot_type=panda`): `observation.images.image` (agentview) +
  `observation.images.image2` (wrist), `observation.state`(8), `action`(8).

Produced: `~/lerobot/datasets/pickitup_replay` — 48 episodes, 6663 frames, fps 20,
av1 video, 26 MB. Loads via `LeRobotDataset("local/pickitup_replay", root=...)`.

## 7. Notes / next steps

- Switch controllers by editing `REPLAY_CONTROLLER` (and `PD_KP/PD_KD` or
  `OSC_KP/OSC_DAMPING_RATIO`) on the env, or via `replay_record.py` flags.
- `image`/`image2` follow the LIBERO key convention; rename to `agentview`/`wrist`
  in the converter if preferred.
- `_set_replay_gains(kp, kd)` is a tuning hook; joint kp/kd was swept (§4), winner
  `pd_kp3000_kd260`.
