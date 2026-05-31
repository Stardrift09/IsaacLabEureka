# PickItUpNoGrasp — ablation environment

## What it is

Ablation variant of `PickItUp` that removes all helper utilities so the LLM must
discover reward shaping from raw geometry and contact forces only.

Inherits: `PickItUpNoGrasp → PickItUp → TestPickItUp`

---

## What is removed

| Helper | Original behaviour | Ablation |
|--------|--------------------|----------|
| `_grasp_detection()` | ContactSensor force history → bool per env | Always `False` |
| `_pregrasp_detection()` | Finger-width heuristic → bool per env | Always `False` |
| `_current_stage_detection()` | One-hot stage tensor `[N, 5]` | All zeros |
| `self.stage` in obs | 5 extra dims in obs vector | **Dropped** |

---

## Termination

`PickItUp._get_dones` gates termination on `self.grasped` (from `_grasp_detection`).
Since that is always False, termination would never fire.

Fix: override `_get_dones` to call `TestPickItUp._get_dones(self)` directly (which
populates `self.high_enough`, `self.small_rotation`, `self.truncated`, etc.), then
terminate on:

```python
terminated = self.high_enough & self.small_rotation
```

No grasped check — object just needs to be lifted above basket rim with small rotation.

---

## Observations

Identical to `TestPickItUp._get_observations` except `self.stage` is NOT in the
`torch.cat`. Obs dims:

```
dof_pos_scaled          (8)
joint_vel               (8)
corners_obj_to_hand     (24)   # 8 corners × 3
target_to_hand_pos      (3)
to_desired_rot          (4)    # quaternion
site_to_target_pos      (3)
target_to_hand_vel      (3)
site_to_target_vel      (3)
prev_actions - actions  (8)
─────────────────────────────
total                   (64)   # vs 69 in TestPickItUp (69 = 64 + 5 stage dims)
```

The docstring inside `_get_observations` explicitly tells the LLM:

> NOTE: `_grasp_detection()`, `_current_stage_detection()`, and `self.stage`
> are NOT available. Shape rewards using raw geometry: finger positions, object
> height, distances, contact forces via `self.scene['left_contact_sensor']` and
> `self.scene['right_contact_sensor']`.

---

## Task score — success in 5 iterations

`_compute_best_metric` in `eureka.py` now uses `mean(metric_data[-5:])` — the
average of the last 5 logged data points — instead of the single best point in the
full training curve.

The task is declared solved when this 5-point tail mean is within `tolerance` of
`success_metric_to_win`. Prevents lucky one-step spikes from triggering early stop.

---

## Run command

```bash
python scripts/train_sequential_vlm.py \
    --task PickItUpNoGrasp \
    --max_eureka_iterations 30 \
    --max_training_iterations 2000 \
    --no_vlm
```
