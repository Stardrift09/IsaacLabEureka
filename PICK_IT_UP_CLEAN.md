# PickItUpClean — clean env, RL training, and parallel data collection

End-to-end pipeline for the cleaned-up "pick up the object, put it in the basket" task:
a clean RL env → PPO training (with a reverse curriculum) → even-coverage parallel
dataset collection from the trained policy, in LeRobot-convertible format.

`TestPickItUp` and `ReplayPickItUpOneEnv` are **left untouched** (existing checkpoints/datasets
depend on them); all of this is new, additive code.

## Components

### 1. Clean env — `IsaacLab/.../direct/franka_cabinet/pick_it_up_clean.py`
`PickItUpClean(TestPickItUp)` overrides only the init surface:
- **Scene**: spawns ONLY the target object + basket (drops the demo's distractor objects).
- **Nominal init** = mean over the 50 demos (robot home joints + base pose + object/basket
  rotations; object/basket XY from `init_state_distribution.npz` means).
- **One flag** `randomize_init` gates ALL init noise; when on, object & basket XY get an
  area-uniform **ring** displacement (`init_ring_inner/outer_diam`). `randomize_rotation` is an
  independent wrist sub-toggle. No `start_in_air`.
- **Curriculum**: the ring outer diameter ramps `start→end` over `curriculum_steps` env steps
  (reverse curriculum: easy→hard). NOTE it keys off `common_step_counter`, which resets per
  process — so for play/eval/collection set `curriculum_enabled=False` (the Eval/Batch cfgs do).
- **CCD** on; robot actuator gains exposed as cfg fields (the base's `cfg.robot` gains are dead;
  `_setup_scene` rebuilds the robot — these cfg fields are the REAL gains, now recorded in `env.yaml`).
- Recording/eval variants reuse `ReplayPickItUpOneEnv` by **diamond inheritance**:
  `EvalPickItUpClean`, `OodEvalPickItUpClean`, `BatchEvalPickItUpClean` (tiled cameras).

Gym ids: `PickItUpClean`, `EvalPickItUpClean`, `OodEvalPickItUpClean`, `BatchEvalPickItUpClean`.
Runner cfg `PickItUpCleanPPORunnerCfg` (experiment_name `pick_it_up_clean`) → logs land in
`logs/rsl_rl/pick_it_up_clean/` (parallel to `test_pick_it_up/`).

### 2. Training — `IsaacLab/scripts/reinforcement_learning/rsl_rl/train_snapshot.py`
Standard rsl_rl `train.py` for any `--task`, plus a self-documenting snapshot written into each
run folder: `run_command.txt`, `console.log` (tee), `source/` (script copy), on top of the usual
`params/env.yaml` + `params/agent.yaml`. Resume with `--resume --load_run <dir> --checkpoint <pt>`;
tune init noise via hydra, e.g. `env.arm_joint_noise=0.10 env.curriculum_ring_end_diam=0.24`.

```bash
cd IsaacLab
python scripts/reinforcement_learning/rsl_rl/train_snapshot.py --task PickItUpClean --headless --seed 42
```

### 3. Correct success logging
`PickItUpClean._get_rewards` logs `success_rate = reset_terminated[done].float()` (only the envs
that finished that step). rsl_rl concatenates these across the rollout → **exact
successes/finished-episodes**. The previously-reported "~2%" success was an artifact of averaging
over all (env, step) pairs; the trained policy is actually **~90%+ at the full 0.24 ring**.

### 4. Init grid — `scripts/make_init_grid.py` (no Isaac app needed)
Even grid over object XY + basket XY (disk radius = `outer_diam/2` around the nominal); robot arm
joints sampled per entry (7-D can't be gridded) and stored, so every init is fully reproducible.
```bash
python scripts/make_init_grid.py --out logs/init_grids/grid_v1.npz \
    --n_object 18 --n_basket 18 --mode product --outer_diam 0.24   # -> 46,656 entries
```
The env consumes it via cfg `init_grid_file` (precedence: grid > ring > nominal), each entry once.

### 5. Parallel collection — `scripts/collect_rsl_rl_parallel.py`
Many envs with TiledCameras (agentview + wrist), init from the grid, **save successes / drop
failures**, output = the existing LeRobot layout (`successful/<epNN>/` with mp4s + `arrays.npz` +
`episode_meta.json`, plus `manifest.jsonl` + `meta.json`). mp4 encoding runs in a
`ProcessPoolExecutor` so the sim never blocks on ffmpeg.
```bash
python scripts/collect_rsl_rl_parallel.py --task BatchEvalPickItUpClean \
    --checkpoint IsaacLab/logs/rsl_rl/pick_it_up_clean/<run>/model_XXXX.pt \
    --grid_file logs/init_grids/grid_v1.npz \
    --num_envs 64 --cam_w 256 --write_workers 16 --record_dir logs/collect_parallel/v1 --headless
```

## Performance notes (collection)
- **Do not** call `sim.render()` during frame capture — `DirectRLEnv.step` already renders RTX
  sensors + `scene.update()`; the extra render was the main stall. Just read `cam.data.output`.
- mp4 encoding is GIL-bound → `ProcessPoolExecutor` (spawn), NOT threads. A semaphore caps
  in-flight episodes for memory.
- `RslRlVecEnvWrapper.__init__` calls `env.reset()`, so the grid cursor is rewound to 0 before the
  collector's own reset (else the first `num_envs` entries are wasted).
- On a 4090: 64 envs @ 256² ≈ 13 GB VRAM, **~17 episodes/s, GPU ~90%**.

## Produced dataset (v1)
`logs/collect_parallel/v1/` — **45,888 successful episodes** (of 46,656 grid entries, 98.3%),
60 fps, 256×256 agentview + wrist, state/action dim 8, ~3.8 GB, full even coverage of the 0.24-ring
state space. Checkpoint: `pick_it_up_clean/2026-06-17_18-04-11/model_2998.pt`. A small inspection
sample (all at the hard outer ring) is in `logs/collect_parallel/sample/`.

Convert to LeRobot with `scripts/convert_to_lerobot.py` in a dedicated lerobot env (do NOT install
lerobot into `eureka` — it pulls numpy>=2 and breaks Isaac Sim).
