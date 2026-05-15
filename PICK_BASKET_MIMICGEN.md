# MimicGen Pick-Basket Pipeline

MimicGen-style synthetic data generation for Franka pick-and-place-in-basket using LIBERO HOPE objects.

---

## What was built

### Problem
`isaaclab_mimic` (complete MimicGen port) requires `ManagerBasedRLMimicEnv`.  
`TestPickItUp` is a `DirectRLEnv` — incompatible.  
Solution: new manager-based pick-basket task wired into the mimic pipeline.

### Files created

#### Task environment
| File | Purpose |
|------|---------|
| `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_basket/pick_basket_env_cfg.py` | Base scene: Franka + alphabet_soup (randomised XY) + basket (fixed). Subtask signals `grasp` and `lift` in `obs_buf["subtask_terms"]`. |
| `.../pick_basket/mdp/observations.py` | `object_grasped_and_lifted`, `object_in_basket`, position observations |
| `.../pick_basket/config/franka/pick_basket_ik_rel_env_cfg.py` | Franka + differential IK relative pose control + object randomisation |
| `.../pick_basket/config/franka/__init__.py` | Gym registrations (see IDs below) |

#### MimicEnv interface
| File | Purpose |
|------|---------|
| `IsaacLab/source/isaaclab_mimic/isaaclab_mimic/envs/franka_pick_basket_mimic_env.py` | Reads `grasp`/`lift` subtask signals from obs_buf |
| `IsaacLab/source/isaaclab_mimic/isaaclab_mimic/envs/franka_pick_basket_mimic_env_cfg.py` | 2 subtasks: **grasp** (object_ref=alphabet_soup), **place** (object_ref=basket) |
| `IsaacLab/source/isaaclab_mimic/isaaclab_mimic/envs/__init__.py` | Registered `Isaac-Franka-PickBasket-IK-Rel-Mimic-v0` here |

#### RMPFlow motion planner (interpolation diversity)
| File | Purpose |
|------|---------|
| `IsaacLab/source/isaaclab_mimic/isaaclab_mimic/motion_planners/rmpflow/rmpflow_planner.py` | Implements `MotionPlannerBase` via shadow-pass: save state → run LULA RmpFlow toward target → record EEF poses → restore state |
| `IsaacLab/source/isaaclab_mimic/isaaclab_mimic/motion_planners/rmpflow/rmpflow_planner_cfg.py` | Config dataclass (Franka defaults, max_steps, goal_tolerance) |

#### Scripted source demo generator
| File | Purpose |
|------|---------|
| `IsaacLab/scripts/tools/generate_scripted_demos_pick_basket.py` | State-machine controller: approach → descend → grasp → lift → move_over_basket → release. Records successful episodes to HDF5. |

---

## Registered gym IDs

| ID | Class | Notes |
|----|-------|-------|
| `Isaac-Franka-PickBasket-IK-Rel-v0` | `ManagerBasedRLEnv` | Plain RL / teleoperation |
| `Isaac-Franka-PickBasket-IK-Rel-Mimic-v0` | `ManagerBasedRLMimicEnv` | MimicGen data generation |

---

## Full pipeline commands

### Step 1 — Generate source demonstrations

**Option A — From LIBERO trajectories (recommended)**

```bash
cd /home/shaotongchen/workspace_eureka/IsaacLabEureka

# libero dataset: 3000 eps, fixed object position
python IsaacLab/scripts/tools/libero_to_isaaclab_demos.py \
    --libero_pkl libero/trajs/libero/pick_up_the_alphabet_soup_and_place_it_in_the_basket/v2/franka_v2.pkl.gz \
    --output_file /tmp/pick_basket_src_raw.hdf5 \
    --num_demos 10 \
    --headless

# libero90 dataset: 50 eps, varying object positions (more diverse)
python IsaacLab/scripts/tools/libero_to_isaaclab_demos.py \
    --libero_pkl "libero/trajs/libero90/libero_90_living_room_scene1_pick_up_the_alphabet_soup_and_put_it_in_the_basket_traj_v2.pkl" \
    --output_file /tmp/pick_basket_src_raw.hdf5 \
    --num_demos 10 \
    --headless
```

Replays each LIBERO episode in the IsaacLab env with direct joint position control, computes delta-EEF actions, saves in `HDF5DatasetFileHandler` format compatible with `annotate_demos.py`.

LIBERO coordinate mapping: `obj_pos_isaaclab = libero_obj_world - libero_robot_world`.  
Action format: `[delta_pos/0.5 (3), axis_angle_delta/0.5 (3), gripper (1)]`.

**Option B — Scripted state machine**

```bash
python IsaacLab/scripts/tools/generate_scripted_demos_pick_basket.py \
    --output_file /tmp/pick_basket_src_raw.hdf5 \
    --num_demos 5 \
    --headless
```
Produces scripted pick-place episodes (fallback if LIBERO replay fails).

### Step 2 — Annotate with subtask termination signals
```bash
python IsaacLab/scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
    --input_file /tmp/pick_basket_src_raw.hdf5 \
    --output_file /tmp/pick_basket_src_annotated.hdf5 \
    --env_id Isaac-Franka-PickBasket-IK-Rel-Mimic-v0
```
Adds `grasp` and `lift` 0→1 edge signals to each trajectory.

### Step 3 — Generate diverse dataset (linear interpolation)
```bash
python IsaacLab/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --input_file /tmp/pick_basket_src_annotated.hdf5 \
    --output_file /tmp/pick_basket_generated.hdf5 \
    --env_id Isaac-Franka-PickBasket-IK-Rel-Mimic-v0 \
    --num_envs 32 \
    --generation_num_trials 1000
```

### Step 3 (alt) — Generate with CuRobo interpolation (collision-free)
```bash
python IsaacLab/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --input_file /tmp/pick_basket_src_annotated.hdf5 \
    --output_file /tmp/pick_basket_generated_skillgen.hdf5 \
    --env_id Isaac-Franka-PickBasket-IK-Rel-Mimic-v0 \
    --num_envs 32 \
    --generation_num_trials 1000 \
    --use_skillgen
```

### Step 3 (alt) — Using the new RMPFlow planner
The `RMPFlowPlanner` is in `isaaclab_mimic/motion_planners/rmpflow/`.  
Wire it into `generate_dataset.py` as an alternative to `CuroboPlanner`:

```python
from isaaclab_mimic.motion_planners.rmpflow import RMPFlowPlanner, RMPFlowPlannerCfg

planner_cfg = RMPFlowPlannerCfg()   # uses Franka defaults; override for other robots
planner = RMPFlowPlanner(env=env, robot=robot, cfg=planner_cfg, env_id=0)
```
Pass `planner` wherever `CuroboPlanner` is passed in the skillgen path of `generate_dataset.py`.

### Step 4 — Train RL or BC on generated data
```bash
# Example: train with rsl_rl using the generated demos as imitation warmup
python IsaacLab/scripts/rsl_rl/train.py \
    --task Isaac-Franka-PickBasket-IK-Rel-v0 \
    --demo_file /tmp/pick_basket_generated.hdf5 \
    --headless
```

---

## Motion diversity summary

| Interpolation strategy | Description | How to enable |
|------------------------|-------------|---------------|
| Linear | Straight-line EEF delta, fast, may collide | default (no flag) |
| CuRobo | GPU optimal collision-free path | `--use_skillgen` |
| **RMPFlow** (new) | Reactive LULA motion policy, curved natural paths | use `RMPFlowPlanner` directly |

Using all three strategies on the same source demos produces **diverse motion styles** in the generated dataset — beneficial for training robust imitation policies.

---

## Architecture answer (motion policy vs planners)

> "Do I need a motion policy or many motion planners?"

For **data generation** you need planners, not a learned policy:
- No chicken-and-egg (policy needs data; data needs policy)
- Planners are deterministic enough to reliably reach waypoints
- Different planners produce different motion profiles → diversity

For **deployment** (after training on generated data), the trained BC/RL policy IS the motion policy.

The long-term loop:
```
scripted demos → MimicGen (diverse planners) → 1000+ demos
    → train RL/BC policy
    → use trained policy as "skill" in next MimicGen round
    → more diverse data → better policy
```

---

## Object assets
Located at:
```
IsaacLabEureka/libero/COMMON/stable_hope_objects/
    alphabet_soup/usd/alphabet_soup.usd   ← target object
    basket/usd/basket.usd                 ← goal container
```

## Key parameters to tune
| Parameter | Location | Effect |
|-----------|----------|--------|
| `pose_range` x/y | `EventCfg.randomize_object_pose` | Object spawn randomisation range |
| `subtask_term_offset_range` | `FrankaPickBasketMimicEnvCfg` | Subtask boundary jitter |
| `action_noise` | `FrankaPickBasketMimicEnvCfg` | Noise on replayed subtask actions |
| `num_interpolation_steps` | `FrankaPickBasketMimicEnvCfg` | Smoothness of subtask transitions |
| `lift_height_threshold` | `SubtaskCfg.lift` obs term | Height above table for "lifted" signal |
| `max_steps` | `RMPFlowPlannerCfg` | Shadow-pass budget for RMPFlow planner |
