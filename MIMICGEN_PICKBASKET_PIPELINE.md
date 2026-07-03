# MimicGen Pick-Basket Pipeline (as actually built & run)

Synthetic dataset for **"pick up the alphabet_soup and put it in the basket"**, generated
with IsaacLab MimicGen from the 50 hand-replayed LIBERO demos, and re-recorded in the
original `logs/replay_record/` format so the existing LeRobot converter ingests it unchanged.

> This supersedes `PICK_BASKET_MIMICGEN.md` (an early **plan** — its LIBERO coordinate
> mapping, scripted-demo / RMPFlow / CuRobo paths, and "randomized object only" scene are
> **not** what was built). This file documents the pipeline that actually produced the
> deliverable.

---

## TL;DR — what exists

- **Deliverable:** `logs/replay_record_mimicgen/` — 1000 demos (944 successful / 56 unsuccessful),
  fps=20, 256×256 agentview + wrist, **8-dim** state/action, ~630 MB. Schema-identical to
  `logs/replay_record/` → convert with `~/lerobot/scripts/convert_to_lerobot.py` in the lerobot env.
- **Generated HDF5 (intermediate):** `logs/pb_generated_1000.hdf5` (1000 success, IK actions +
  `initial_state`); `..._failed.hdf5` = dropped trials.
- **Annotated source (reusable):** `/tmp/pb_src_annotated.hdf5` (12 demos, with `datagen_info`).
  Annotation is randomization-independent → reuse it to regenerate at any randomization level.
- **Playback / BC eval:** **no new env** — `EvalPickItUpClean` + `eval_client.py` runtime flags
  (see [Playback](#6--playback--bc-policy-eval)).

---

## Pipeline overview

```
LIBERO pkl (joint traj, 50 demos)
  │  IsaacLab/scripts/tools/libero_to_isaaclab_demos.py   (closed-loop IK replay + dwell)
  ▼
source HDF5  /tmp/pb_src_raw.hdf5   (initial_state + 7-D IK-rel actions, success-gated)
  │  annotate_demos.py  --auto
  ▼
annotated    /tmp/pb_src_annotated.hdf5   (+ obs/datagen_info/{eef_pose,object_pose,target_eef_pose,subtask_term_signals})
  │  generate_dataset.py  --generation_num_trials 1000   (warp + linear connector, generation_guarantee)
  ▼
generated    logs/pb_generated_1000.hdf5   (1000 success; ~34% per-trial yield)
  │  scripts/record_generated_in_replay_format.py  --fps 20  --enable_cameras
  ▼
DATASET      logs/replay_record_mimicgen/   (replay_record schema: 8-dim + agentview/wrist mp4)
  │  ~/lerobot/scripts/convert_to_lerobot.py   (separate lerobot env)
  ▼
LeRobotDataset → train ACT/BC
```

All Isaac steps run in conda env **`eureka`**. Always export `HDF5_USE_FILE_LOCKING=FALSE`.

---

## Why a new manager-based env (not TestPickItUp)

`isaaclab_mimic` requires a `ManagerBasedRLMimicEnv`; `TestPickItUp` is a `DirectRLEnv`
(incompatible). So a **manager-based pick-basket env** mirrors TestPickItUp's *scene* (the
authoritative task) but in manager form, only keeping what MimicGen needs (object-centric
subtasks `grasp`/`lift`, IK-rel action, subtask term signals).

### Scene = TestPickItUp's, in the LIBERO world frame
`IsaacLab/.../manager_based/manipulation/pick_basket/pick_basket_env_cfg.py`
- **Objects + basket only** — distractors (ketchup / cream_cheese / tomato_sauce) removed, so
  the scene matches PickItUpClean for a fair cross-method comparison.
- **No table** — TestPickItUp rests objects on the ground plane at **z=0**. A finite table made
  the basket (LIBERO y≈0.26) spawn off the edge and fall through → place subtask failed.
- **LIBERO world frame (critical for the agentview to match the 50 demos):**
  - robot base at `(-0.51, 0, 0)` (set in the IK-rel cfg)
  - soup nominal `pos=[0.0541, -0.1001, 0.0552]`, `rot=[0,0,0.7071,0.7071]`
  - basket `pos=[-0.0024, 0.2613, 0.0122]`, `rot=[0.707,-0.002,0.002,0.707]`
  - nominals are the **mean over the 50 demos** (`logs/.../init_state_distribution.npz`),
    same nominal PickItUpClean uses.
  - In the bridge, object world pos is the LIBERO pos directly (`env_origin + libero_pos`) —
    **not** `libero_world − robot_world` (the old doc's mapping was wrong).

### Key env files
| File | Purpose |
|------|---------|
| `.../pick_basket/pick_basket_env_cfg.py` | scene (soup+basket, no table, z=0), obs groups, subtask terms, terminations (`object_in_basket` success) |
| `.../pick_basket/mdp/observations.py` | `randomize_object_position` (ring), `object_grasped(_and_lifted)`, `object_in_basket`, pos obs |
| `.../pick_basket/config/franka/pick_basket_ik_rel_env_cfg.py` | Franka @ (-0.51,0,0), differential **IK-rel** action (dls, scale=0.5), binary gripper, `randomize_object_pose` event |
| `.../pick_basket/config/franka/pick_basket_record_env_cfg.py` | adds agentview + wrist `CameraCfg`; gym id `...-Record-v0` |
| `isaaclab_mimic/.../envs/franka_pick_basket_mimic_env{,_cfg}.py` | MimicEnv: 2 subtasks **grasp**(ref=soup) → **place**(ref=basket); reads `grasp`/`lift` signals |

### Gym IDs
| ID | Use |
|----|-----|
| `Isaac-Franka-PickBasket-IK-Rel-Mimic-v0` | annotate + generate (MimicGen) |
| `Isaac-Franka-PickBasket-IK-Rel-Record-v0` | re-record generated demos (adds cameras) |

---

## Steps (real commands)

### 1 — LIBERO pkl → IK source HDF5
`IsaacLab/scripts/tools/libero_to_isaaclab_demos.py`
```bash
HDF5_USE_FILE_LOCKING=FALSE python IsaacLab/scripts/tools/libero_to_isaaclab_demos.py \
  --libero_pkl "libero/trajs/libero90/libero_90_living_room_scene1_pick_up_the_alphabet_soup_and_put_it_in_the_basket_traj_v2.pkl" \
  --output_file /tmp/pb_src_raw.hdf5 --num_demos 12 --dwell 3 --headless
```
The pkl is **joint-space**; the bridge recovers IK-rel actions by **closed-loop replay** in the
IK-rel env (two passes):
1. capture per-waypoint `ref_pos`/`ref_quat` + a binary `grip_cmd` (close when LIBERO finger
   target mean < `grip_close_target=0.035` — LIBERO finger targets span 0.032–0.040).
2. chase each waypoint via `env.target_eef_pose_to_action(...)`, **dwelling `--dwell` steps** per
   waypoint to beat the IK `scale=0.5` chase lag (DWELL=3 → ~89% per-demo yield; open-loop IK
   deltas gave 0/10 and ~0.2 m drift).

Success-gated (soup-in-basket). Yields ~12 clean source demos.

### 2 — Annotate subtask boundaries
```bash
HDF5_USE_FILE_LOCKING=FALSE python IsaacLab/scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --input_file /tmp/pb_src_raw.hdf5 --output_file /tmp/pb_src_annotated.hdf5 \
  --task Isaac-Franka-PickBasket-IK-Rel-Mimic-v0 --auto --headless
```
`--auto` writes `obs/datagen_info/{eef_pose,object_pose,target_eef_pose,subtask_term_signals}`.
**Reusable across randomization levels** — annotation does not depend on init randomization.

### 3 — Generate diverse dataset
```bash
HDF5_USE_FILE_LOCKING=FALSE python IsaacLab/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --input_file /tmp/pb_src_annotated.hdf5 --output_file logs/pb_generated_1000.hdf5 \
  --task Isaac-Franka-PickBasket-IK-Rel-Mimic-v0 --num_envs 10 \
  --generation_num_trials 1000 --headless
```
Per trial: reset (object+basket randomized by the env event, see below) → warp each subtask's
EEF segment by the object-pose **delta** → **linear**-interpolate the connector → replay through
IK. `generation_guarantee=True` retries until 1000 successes; failures go to `..._failed.hdf5`
(`EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES`). Per-trial yield ≈ **34%** at the 0.24 ring.

> **Randomization keeps orientation fixed.** Early runs randomized object yaw → 0.1% yield
> (warp rotation no longer ~identity). `randomize_object_position` displaces **XY only**, keeps z
> + orientation → yield recovered.

### 4 — Re-record in `replay_record` format
`scripts/record_generated_in_replay_format.py`
```bash
HDF5_USE_FILE_LOCKING=FALSE python scripts/record_generated_in_replay_format.py \
  --input_file logs/pb_generated_1000.hdf5 \
  --output_dir logs/replay_record_mimicgen \
  --fps 20 --enable_cameras --headless
```
Per generated demo: `reset_to(initial_state)` → step the stored IK actions → record
- `observation_state[T,8]` = `robot.joint_pos[7 arm + finger1]`
- `action[T,8]` = `robot.joint_pos_target[7 arm + finger1]` (the IK controller's commanded
  absolute joint target — **this is what a BC policy learns to output**)
- `agentview.mp4` + `wrist.mp4` + `sidebyside.mp4`

Agentview is aimed with `set_world_poses_from_view(eye, target)` after reset/each `reset_to`,
using the LIBERO constants `AGENTVIEW_EYE=(1.05,0,0.55)`, `AGENTVIEW_TARGET=(0,0,0.08)`. Splits
`successful/` vs `unsuccessful/` by the soup-in-basket check; writes `manifest.jsonl` + `meta.json`.
A few generated-success demos miss on re-record (IK chase non-determinism) → ~94% record yield.

`--max_demos N` records a small sample for eyeballing first.

### 5 — Convert to LeRobot (separate `lerobot` env)
`~/lerobot/scripts/convert_to_lerobot.py` reads `manifest.jsonl` + per-ep `arrays.npz` +
agentview/wrist mp4; its `to8()` takes the 8-dim arrays as-is. **Do not** install lerobot into
`eureka` (pulls numpy≥2, breaks Isaac Sim).

---

## Randomization (the knob you actually tune)

`pick_basket_ik_rel_env_cfg.py → EventCfg.randomize_object_pose` calls
`mdp.randomize_object_position(asset_cfgs=[alphabet_soup, basket], ring_outer_diam, ring_inner_diam, area_uniform=True)`.

Each listed asset's **XY** is displaced from its nominal by an **area-uniform disk** of radius
`ring_outer_diam/2` (z + orientation kept). This mirrors PickItUpClean's `_sample_init_ring`
(`r = sqrt(r_in² + u·(r_out² − r_in²))`).

| Level | `ring_outer_diam` | radius | note |
|-------|-------------------|--------|------|
| PickItUpClean parity | 0.24 | 0.12 m | original deliverable; ACT eval was poor (state space too large) |
| **tight** | **0.06** | **0–0.03 m** | object **and** basket; for a learnable distribution |

To regenerate at a new level: edit `ring_outer_diam` → rerun **Step 3 + Step 4** (reuse the
annotated source from Step 2; no re-annotate). Write to a **new** output dir to keep prior levels.

---

## Recorded format (matches `logs/replay_record/`)

```
logs/replay_record_mimicgen/
  manifest.jsonl                       # one line per episode
  meta.json                            # fps=20, state_dim=8, action_dim=8, cameras=[agentview,wrist]
  successful/epNN/   unsuccessful/epNN/
    arrays.npz   # observation_state[T,8] float32, action[T,8] float32, timestamp[T] float32
    agentview.mp4  wrist.mp4  sidebyside.mp4
    episode_meta.json
```
8-dim = 7 arm joints + 1 gripper (finger_joint1). The 2nd finger is symmetric and dropped
(synthesized by copying the gripper at apply time) — not worth learning.

---

## 6 — Playback / BC-policy eval

The action a BC policy outputs = **absolute 8-dim joint targets**, *not* the env's RL delta
action. `ReplayPickItUpOneEnv` (base of `EvalPickItUpClean`) already has the matching path:
`eval_reset` / `eval_step` (applies absolute targets via `set_joint_position_target`, gripper
copied to both fingers) / `eval_observation` (state8 + agentview/wrist). Same clean scene, LIBERO
frame, cameras, and 0.24-ring as the record env → **no new env class needed**.

`eval_client.py` exposes the two settings that must match the recording, at runtime:
```bash
~/miniconda3/envs/eureka/bin/python -u scripts/eval_client.py \
  --task EvalPickItUpClean --controller joint_pd \
  --eval_fps 20 --kp 400 --kd 80 \
  --server_url http://<policy-host>:8080 \
  --episodes 50 --enable_cameras --headless \
  --pretrained_path <ACT checkpoint>     # if using the local-policy path
```
- `--eval_fps 20` → `steps = round((1/20)/sim.dt)` = 0.05 s/action = the recording rate.
- `--controller joint_pd --kp 400 --kd 80` → the 7 arm joints get the record env's HIGH_PD arm
  gains (shoulder+forearm both 400/80, uniform); hand stays 2000/100 (already matches).

**Minor residual gap:** record env sim.dt=0.01 (100 Hz physics) vs EvalPickItUpClean 1/180
(180 Hz). Control period (0.05 s) and gains match, so the policy sees identical 20 Hz state +
PD response; only the integration substep differs (finer in eval). Negligible; exact match =
1-line `sim.dt=0.01` cfg override.

---

## Debug history / gotchas (don't re-hit these)
- **`[9,1]` joint tensor shape** in the bridge → `float(np.asarray(dof[name]).reshape(-1)[0])`.
- **Annotate 0/10** (open-loop IK deltas don't reproduce the grasp, ~0.2 m drift) → closed-loop
  recording via `target_eef_pose_to_action`.
- **Gripper never closed** (always +1) → LIBERO finger targets are 0.032–0.040; threshold 0.035.
- **Grasp misses from IK lag** (22% yield) → DWELL=3 (89%).
- **Generation 0.1%** → object yaw randomization broke the warp; switched to position-only.
- **Basket fell through** → removed the finite table, ground at z=0.
- **Agentview framing wrong** → `set_world_poses_from_view` + put robot/objects in the LIBERO
  world frame (robot at -0.51), matching the 50 demos.
- **HDF5 BlockingIOError** → an orphaned gen process held the lock; `HDF5_USE_FILE_LOCKING=FALSE`.
- **Distractors barely changed yield** (33.6%→33.8%) — removing them is a *scene-parity / clean-
  video* choice, not a yield fix. MimicGen warp/interpolation is obstacle-blind; distractors only
  cause sim-replay collisions that `generation_guarantee` already drops.
- **RTX `frames discarded` log lines** are benign — don't treat as fatal when monitoring.

## Object assets
`libero/COMMON/stable_hope_objects/{alphabet_soup,basket}/usd/*.usd`
