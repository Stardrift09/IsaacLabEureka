# Cabinet Open-Drawer MimicGen Pipeline

MimicGen-style synthetic data generation for Franka open-top-drawer using the Sektion Cabinet.

---

## Design

### Subtask decomposition

The task has **one subtask boundary** → **two segments**:

| Segment | Name | Signal | object_ref | Who executes |
|---------|------|--------|-----------|--------------|
| 1 | approach | `approached` (dist < 0.1 m) | `cabinet` | **motion planner** (replaces this segment for each new cabinet pose) |
| 2 | open | — (final) | `cabinet` | **your trained policy** (replayed + warped from source demo) |

Segment 2 maps exactly to the requirement:  
> "when distance from hand to the drawer is smaller than 0.1, the motion will be taken over by my policy."

### Why `object_ref="cabinet"` for both segments

The cabinet is an `Articulation`. `PickPlaceRelMimicEnv.get_object_poses` already handles articulations (processes `scene_state["articulation"]` for all non-robot entries).  
Both segments warp their EEF waypoints by the **delta in cabinet root pose**, which captures XY position randomisation correctly since the drawer-handle offset in the cabinet frame is fixed when the drawer is closed.

---

## Files added / modified

### Task env (extends existing manager-based cabinet)

| File | Change |
|------|--------|
| `IsaacLab/source/isaaclab_tasks/.../cabinet/mdp/subtask_observations.py` | New — `drawer_approached(env, threshold)` using existing `ee_frame` + `cabinet_frame` FrameTransformers |
| `IsaacLab/source/isaaclab_tasks/.../cabinet/mdp/__init__.py` | Added `drawer_approached` export |
| `IsaacLab/source/isaaclab_tasks/.../cabinet/config/franka/franka_cabinet_mimic_env_cfg.py` | New — `FrankaCabinetMimicEnvCfg`: overrides observations (non-concat + subtask group), adds cabinet XY randomisation event, configures 2 subtask segments |
| `IsaacLab/source/isaaclab_tasks/.../cabinet/config/franka/__init__.py` | Registered `Isaac-Open-Drawer-Franka-IK-Rel-Mimic-v0` + added `rsl_rl_cfg_entry_point` to `Isaac-Open-Drawer-Franka-IK-Rel-v0` |

### MimicEnv interface

| File | Change |
|------|--------|
| `IsaacLab/source/isaaclab_mimic/isaaclab_mimic/envs/franka_cabinet_mimic_env.py` | New — `FrankaCabinetMimicEnv(PickPlaceRelMimicEnv)`: returns `{"approached": ...}` subtask signals |
| `IsaacLab/source/isaaclab_mimic/isaaclab_mimic/envs/__init__.py` | Registered `Isaac-Open-Drawer-Franka-IK-Rel-Mimic-v0` |

### Policy demo collector

| File | Purpose |
|------|---------|
| `IsaacLab/scripts/reinforcement_learning/rsl_rl/collect_policy_demos_cabinet.py` | Rolls out a trained RSL-RL policy, records successful episodes (drawer_top_joint > 0.35 m) to HDF5 using `ActionStateRecorderManagerCfg`. |

---

## Registered gym IDs

| ID | Class | Notes |
|----|-------|-------|
| `Isaac-Open-Drawer-Franka-IK-Rel-v0` | `ManagerBasedRLEnv` | RL training (existing) |
| `Isaac-Open-Drawer-Franka-IK-Rel-Mimic-v0` | `ManagerBasedRLMimicEnv` | MimicGen data generation |

---

## Policy requirement

The source-demo collector runs the policy in **`Isaac-Open-Drawer-Franka-IK-Rel-v0`** (IK-relative 7-DoF actions: `[Δpos/0.5, Δrot/0.5, gripper]`).  
**The policy must have been trained on this env.**  
Policies from `Isaac-Franka-Cabinet-Direct-v0` (joint-position, 9-DoF) use a different action space and **cannot be used with MimicGen** — retrain on the IK-rel env first.

---

## Full pipeline commands

### Step 1 — Collect source demonstrations from your policy

```bash
cd /home/shaotongchen/workspace_eureka/IsaacLabEureka

python IsaacLab/scripts/reinforcement_learning/rsl_rl/collect_policy_demos_cabinet.py \
    --task Isaac-Open-Drawer-Franka-IK-Rel-v0 \
    --checkpoint /path/to/your/model.pt \
    --output_file /tmp/cabinet_src_raw.hdf5 \
    --num_demos 20 \
    --headless
```

The script:
1. Loads your RSL-RL checkpoint into the IK-rel cabinet env
2. Rolls out the policy; records `initial_state` + `actions` for each episode
3. Automatically detects success (`drawer_top_joint > 0.35 m`) via a built-in termination term
4. Exports only successful episodes to HDF5 (format compatible with `annotate_demos.py`)

### Step 2 — Annotate with subtask termination signals

```bash
python IsaacLab/scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
    --input_file /tmp/cabinet_src_raw.hdf5 \
    --output_file /tmp/cabinet_src_annotated.hdf5 \
    --task Isaac-Open-Drawer-Franka-IK-Rel-Mimic-v0 \
    --auto \
    --headless
```

Replays each source demo in the Mimic env and records when `approached` (EEF dist to handle < 0.1 m) transitions 0→1. This marks the subtask boundary.

### Step 3 — Generate diverse dataset (linear interpolation)

```bash
python IsaacLab/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --input_file /tmp/cabinet_src_annotated.hdf5 \
    --output_file /tmp/cabinet_generated.hdf5 \
    --task Isaac-Open-Drawer-Franka-IK-Rel-Mimic-v0 \
    --num_envs 32 \
    --generation_num_trials 1000 \
    --headless
```

For new cabinet positions (±0.04 m X, ±0.1 m Y):
- Segment 1 (approach): waypoints warped by cabinet pose delta + planner interpolation
- Segment 2 (open): actions warped by cabinet pose delta, replayed directly

### Step 3 (alt) — Collision-free interpolation with CuRobo

```bash
python IsaacLab/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --input_file /tmp/cabinet_src_annotated.hdf5 \
    --output_file /tmp/cabinet_generated_curobo.hdf5 \
    --task Isaac-Open-Drawer-Franka-IK-Rel-Mimic-v0 \
    --num_envs 32 \
    --generation_num_trials 1000 \
    --use_skillgen \
    --headless
```

CuRobo plans collision-free approach paths — important here because the cabinet body is a significant obstacle when approaching from unusual angles.

### Step 4 — Train policy on generated data

```bash
python IsaacLab/scripts/rsl_rl/train.py \
    --task Isaac-Open-Drawer-Franka-IK-Rel-v0 \
    --demo_file /tmp/cabinet_generated.hdf5 \
    --headless
```

---

## Collision avoidance note

The approach segment is replayed from source demos and warped by the cabinet pose delta. For Segment 1, **CuRobo (`--use_skillgen`) is recommended** because:
- Linear interpolation may clip through the cabinet face when the source demo approach angle differs from the warped pose
- CuRobo respects the cabinet collision mesh automatically

---

## Key parameters to tune

| Parameter | Location | Effect |
|-----------|----------|--------|
| `threshold` | `_SubtaskCfg.approached` (mimic_env_cfg) | EEF-to-handle distance for subtask boundary |
| `pose_range` x/y | `events.randomize_cabinet_pose` | Cabinet spawn randomisation range |
| `subtask_term_offset_range` | `SubTaskConfig` approach segment | Boundary jitter (frames) |
| `action_noise` | `SubTaskConfig` both segments | Noise on replayed actions |
| `num_interpolation_steps` | `SubTaskConfig` approach segment | Smoothness of planner interpolation |
| `--num_demos` | `collect_policy_demos_cabinet.py` | Source demo count (more = better diversity) |
| `--max_episodes` | `collect_policy_demos_cabinet.py` | Retry budget if policy has low success rate |
