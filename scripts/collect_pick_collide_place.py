"""Parallel LeRobot-dataset collection of the FULL pick -> collide -> place pipeline.

Combines:
  * the 3-policy phase machine from
    ``IsaacLab/scripts/reinforcement_learning/rsl_rl/play_pick_collide_place.py``
    (PICK / COLLIDE / PLACE checkpoints with different obs shapes, obs-slicing for the
    71-dim pick/place policies vs the 78-dim collide policy, rule-based phase switching), and
  * the parallel camera recording / encoding from ``scripts/collect_rsl_rl_parallel.py``
    (many envs each with TiledCameras, per-env state/action/frame buffers, ProcessPool mp4
    encoding, ``successful/`` + ``manifest.jsonl`` + ``meta.json`` output).

The env, the three checkpoints and the phase rules are HARD-CODED below. The env is
``BatchEvalPickItUpCollideClean`` (camera + tomato_sauce can). The episode starts with the
object on the table; the recorded action is the commanded absolute joint target (``_target8``),
so the resulting dataset is a single behavior-cloning corpus of the whole pick+collide+place
task regardless of which sub-policy produced each step.

Only episodes that COMPLETE THE PLACE (object inside the basket) are saved; timeouts/failures
are dropped. Run in conda ``eureka``:

    ~/miniconda3/envs/eureka/bin/python -u scripts/collect_pick_collide_place.py \
        --num_envs 16 --num_episodes 200 --record_dir logs/collect_pick_collide_place/v1 --headless
"""

import argparse
import os
import sys

from isaaclab_eureka.utils import get_freest_gpu

# Reuse the episode encoder (mp4 + arrays.npz + episode_meta.json) from the parallel collector.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_rsl_rl_parallel import _encode_episode  # noqa: E402

# ----------------------------------------------------------------------------
# HARD-CODED env + checkpoints (the pick/collide/place pipeline).
# ----------------------------------------------------------------------------
TASK = "BatchEvalPickItUpCollideClean"
PICK_CKPT = "/home/shaotongchen/workspace_eureka/IsaacLabEureka/IsaacLab/logs/rsl_rl/pick_it_up_clean/2026-06-17_18-04-11/model_2998.pt"
COLLIDE_CKPT = "/home/shaotongchen/workspace_eureka/IsaacLabEureka/logs/rl_runs/rsl_rl_eureka/pick_it_up_clean/2026-06-30_10-44-01_Run-0_tueilsy-st-13/model_1499.pt"
PLACE_CKPT = "/home/shaotongchen/workspace_eureka/IsaacLabEureka/IsaacLab/logs/rsl_rl/pick_it_up_clean/2026-06-30_23-57-31/model_1999.pt"


def main(args_cli):
    from isaaclab.app import AppLauncher

    device = args_cli.device
    if device == "cuda":
        device = f"cuda:{get_freest_gpu()}"
    app_launcher = AppLauncher(headless=args_cli.headless, enable_cameras=True, device=device)
    simulation_app = app_launcher.app

    import multiprocessing as mp
    import threading
    from concurrent.futures import ProcessPoolExecutor

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import numpy as np
    import torch
    from tensordict import TensorDict
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunner

    # ---- obs-slicing helpers (so the 71-dim pick/place policies load against the 78-dim env) ----
    def _slice_obs(obs, n):
        return TensorDict({k: v[:, :n] for k, v in obs.items()}, batch_size=obs.batch_size)

    class _ObsSliceEnv:
        """Proxy exposing only the first n_obs obs dims; delegates everything else."""

        def __init__(self, base, n_obs):
            object.__setattr__(self, "_base", base)
            object.__setattr__(self, "_n_obs", n_obs)

        def get_observations(self):
            return _slice_obs(self._base.get_observations(), self._n_obs)

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_base"), name)

    def _ckpt_obs_dim(path):
        sd = torch.load(path, map_location="cpu")["model_state_dict"]
        if "actor_obs_normalizer._mean" in sd:
            return int(sd["actor_obs_normalizer._mean"].shape[-1])
        for k, v in sd.items():
            if k.startswith("actor.") and k.endswith(".weight") and v.ndim == 2:
                return int(v.shape[1])
        raise RuntimeError(f"cannot infer obs dim from {path}")

    # ---- env: cameras + can; object on the table; collision termination disabled ----
    env_cfg = parse_env_cfg(TASK, device=device, num_envs=args_cli.num_envs)
    env_cfg.sim.device = device
    env_cfg.cam_w = env_cfg.cam_h = args_cli.cam_w
    env_cfg.num_rerenders_on_reset = args_cli.rerenders
    env_cfg.scene.env_spacing = args_cli.env_spacing
    if args_cli.antialiasing:
        env_cfg.sim.render.antialiasing_mode = args_cli.antialiasing
    env_cfg.init_states_file = ""                      # object starts on the TABLE (pick phase)
    env_cfg.collision_displacement_threshold = 1.0e9   # don't auto-terminate on the collision
    env_cfg.curriculum_enabled = False
    env_cfg.init_grid_file = "/home/shaotongchen/workspace_eureka/IsaacLabEureka/logs/init_grids/grid_v1.npz"                         # random table inits (no fixed grid)

    # We reset every env manually (below) at our own --max_frames budget, and we DISABLE the
    # env's own timeout so it never auto-resets under us -- otherwise a done env's collision /
    # placement state is wiped inside env.step() before we can read the episode outcome. Push
    # the env truncation well past our budget; episode_length_buf is zeroed on each manual reset.
    if args_cli.max_frames and args_cli.max_frames > 0:
        control_dt = env_cfg.sim.dt * env_cfg.decimation
        env_cfg.episode_length_s = (args_cli.max_frames + 50) * control_dt

    env = gym.make(TASK, cfg=env_cfg)
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg.device = device
    env = RslRlVecEnvWrapper(env)
    u = env.unwrapped
    n_env = u.num_envs
    device = u.device
    u.REPLAY_CONTROLLER = "pick_collide_place"  # label recorded in the dataset manifest/meta

    # ---- load the three policies (pick/place 71-dim sliced, collide 78-dim full) ----
    full_obs_dim = env.get_observations()["policy"].shape[-1]
    dims = {p: _ckpt_obs_dim(p) for p in (PICK_CKPT, COLLIDE_CKPT, PLACE_CKPT)}
    print(f"[collect] obs dims env={full_obs_dim} pick={dims[PICK_CKPT]} "
          f"collide={dims[COLLIDE_CKPT]} place={dims[PLACE_CKPT]}", flush=True)

    def _load_policy(path):
        obs_dim = dims[path]
        view = env if obs_dim == full_obs_dim else _ObsSliceEnv(env, obs_dim)
        runner = OnPolicyRunner(view, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(path, map_location=agent_cfg.device)
        try:
            nn = runner.alg.policy
        except AttributeError:
            nn = runner.alg.actor_critic
        return runner.get_inference_policy(device=device), nn, obs_dim

    pick_policy, pick_nn, pick_obs = _load_policy(PICK_CKPT)
    collide_policy, collide_nn, collide_obs = _load_policy(COLLIDE_CKPT)
    place_policy, place_nn, place_obs = _load_policy(PLACE_CKPT)

    # ---- recording setup ----
    os.makedirs(args_cli.record_dir, exist_ok=True)
    max_steps = int(u.max_episode_length) + 5
    rec_fps = u._recording_fps(args_cli.record_every * u.cfg.decimation)
    cams = list(u._record_cams.keys())
    base_task = f"pick up the {u.target_object_name}, collide it with the {u.collision_object_name}, and place it in the {u.target_site_name}"
    gains = u._gains_dict()
    print(f"[collect] task={TASK} num_envs={n_env} target_eps={args_cli.num_episodes} cam={args_cli.cam_w} "
          f"fps={rec_fps} max_steps={max_steps} -> {args_cli.record_dir}", flush=True)

    # ---- grid bookkeeping: each init used ONCE; stop when ALL entries have been visited ----
    assert u._grid_active(), (
        "This collector needs a deterministic init grid (one demo per init). Set init_grid_file.")
    n_grid = u._grid_N
    # RslRlVecEnvWrapper.__init__ already reset once (advancing the grid cursor); rewind so OUR
    # reset dispatches entries 0..n_env-1 cleanly.
    u._grid_cursor = 0
    u._grid_exhausted = False
    obs, _ = env.reset()
    u._aim_agentview_all()
    gd = np.load(env_cfg.init_grid_file)
    grid_object_xy, grid_basket_xy = gd["object_xy"], gd["basket_xy"]

    # per-grid-entry outcome record (index == grid entry)
    g_visited = np.zeros(n_grid, dtype=bool)
    g_success = np.zeros(n_grid, dtype=bool)
    g_collided = np.zeros(n_grid, dtype=bool)   # was the can knocked this episode
    g_placed = np.zeros(n_grid, dtype=bool)     # was the object in the basket at the end
    g_wrong_order = np.zeros(n_grid, dtype=bool)
    g_reason = np.array([""] * n_grid, dtype=object)

    # per-env episode state + phase machine (0 pick, 1 collide, 2 place)
    phase = torch.zeros(n_env, dtype=torch.long, device=device)
    buf_frames = [{c: [] for c in cams} for _ in range(n_env)]
    buf_states = [[] for _ in range(n_env)]
    buf_actions = [[] for _ in range(n_env)]
    step_in_ep = [0] * n_env
    ep_collided = np.zeros(n_env, dtype=bool)   # can knocked at any point this episode (latched)
    ep_premature = np.zeros(n_env, dtype=bool)  # object in basket BEFORE the collide (latched)

    # which grid entry each env is attempting + whether it is a post-exhaustion duplicate (drop)
    ep_grid_idx = u._grid_idx.detach().cpu().numpy().copy()
    entry_started = np.zeros(n_grid, dtype=bool)
    is_dup = np.zeros(n_env, dtype=bool)
    for e in range(n_env):
        gi = int(ep_grid_idx[e])
        if entry_started[gi]:
            is_dup[e] = True
        else:
            entry_started[gi] = True

    writer_pool = ProcessPoolExecutor(max_workers=args_cli.write_workers, mp_context=mp.get_context("spawn"))
    write_sem = threading.Semaphore(args_cli.max_pending)
    results = []
    ep_counter = 0
    n_visited = 0

    def _placed():
        obj = u.target_object.data.root_pos_w
        bz = u.basket_corners_world[:, :, 2]
        site = u.target_site.data.root_pos_w[:, :2]
        inside = ((obj[:, :2] - site) ** 2).sum(-1) < float(u.target_site_radius) ** 2
        return inside & (obj[:, 2] < bz.max(dim=1).values) & (obj[:, 2] > bz.min(dim=1).values)

    # Stop only when every grid entry has been handed out AND all envs are now on duplicates
    # (i.e. every real init has finished its single attempt). --num_episodes is IGNORED here.
    while not (u._grid_exhausted and bool(is_dup.all())):
        frames = u._capture_frames_all()                  # {cam: [n_env,H,W,3] uint8} (obs_t)
        states = u._state8_all().detach().cpu().numpy()   # [n_env, 8]
        with torch.inference_mode():
            pick_a = pick_policy(_slice_obs(obs, pick_obs))
            collide_a = collide_policy(obs)
            place_a = place_policy(_slice_obs(obs, place_obs))
            actions = torch.where((phase == 1).unsqueeze(-1), collide_a,
                                  torch.where((phase == 2).unsqueeze(-1), place_a, pick_a))
            obs, _, dones, _ = env.step(actions)
        targets = u._target8_all().detach().cpu().numpy()  # T_t applied at obs_t

        grasped = u.grasped
        can_disp = torch.norm(u.collision_obj_pos - u.collision_object_init_pos, dim=-1)
        collided_t = can_disp > args_cli.collide_threshold
        placed_t = _placed()
        collided = collided_t.detach().cpu().numpy()
        placed = placed_t.detach().cpu().numpy()

        # action phase machine
        phase[(phase == 0) & grasped] = 1
        phase[(phase == 1) & collided_t] = 2

        # per-episode latches (no auto-reset -> can_disp/placed stay valid every step)
        collided_ever = ep_collided | collided
        ep_premature |= placed & ~collided_ever          # in basket before ANY collision
        ep_collided = collided_ever

        # success = object in basket now AND collided already AND correct order (collide->place)
        success = placed & collided_ever & ~ep_premature
        over = np.asarray(step_in_ep) >= args_cli.max_frames
        boundary = success | ep_premature | over | dones.detach().cpu().numpy()

        manual_reset = []
        for e in range(n_env):
            if not boundary[e]:
                for c in cams:
                    buf_frames[e][c].append(frames[c][e])
                buf_states[e].append(states[e].copy())
                buf_actions[e].append(targets[e].copy())
                step_in_ep[e] += 1
                continue

            gi = int(ep_grid_idx[e])
            record = not is_dup[e]   # post-exhaustion duplicates are neither saved nor recorded

            if success[e]:
                reason = "success"
                if record and len(buf_states[e]) > 0:
                    write_sem.acquire()
                    cam_arr = {c: np.asarray(buf_frames[e][c]) for c in cams}
                    fut = writer_pool.submit(
                        _encode_episode, args_cli.record_dir, ep_counter, cam_arr,
                        np.asarray(buf_states[e]), np.asarray(buf_actions[e]),
                        rec_fps, base_task, "pick_collide_place", gains)
                    fut.add_done_callback(lambda f: write_sem.release())
                    results.append({"_future": fut, "episode": ep_counter, "success": True,
                                    "frames": len(buf_states[e]), "controller": "pick_collide_place",
                                    "grid_idx": gi})
                    ep_counter += 1
            elif ep_premature[e]:
                reason = "wrong_order:placed_before_collision"
            else:  # over-budget (timeout) -> report which sub-goals are missing
                miss = []
                if not ep_collided[e]:
                    miss.append("no_collision")
                if not placed[e]:
                    miss.append("not_in_basket")
                reason = "timeout:" + "+".join(miss) if miss else "timeout"

            if record:
                g_visited[gi] = True
                g_success[gi] = bool(success[e])
                g_collided[gi] = bool(ep_collided[e])
                g_placed[gi] = bool(placed[e])
                g_wrong_order[gi] = bool(ep_premature[e])
                g_reason[gi] = reason
                n_visited += 1
                if n_visited % 100 == 0:
                    print(f"  visited {n_visited}/{n_grid}  success={int(g_success.sum())}  "
                          f"saved_demos={ep_counter}", flush=True)

            # clear this env's episode state (every boundary env is reset manually below)
            buf_frames[e] = {c: [] for c in cams}
            buf_states[e] = []
            buf_actions[e] = []
            step_in_ep[e] = 0
            phase[e] = 0
            ep_collided[e] = False
            ep_premature[e] = False
            manual_reset.append(e)

        if manual_reset:
            ids = torch.tensor(manual_reset, dtype=torch.long, device=device)
            with torch.inference_mode():
                u._reset_idx(ids)
                u.episode_length_buf[ids] = 0
                obs = env.get_observations()
            # each reset advanced the grid cursor -> assign each reset env its NEW entry
            new_grid = u._grid_idx.detach().cpu().numpy()
            for e in manual_reset:
                gi = int(new_grid[e])
                ep_grid_idx[e] = gi
                is_dup[e] = bool(entry_started[gi])   # already visited -> this attempt is a dup
                entry_started[gi] = True

        pick_nn.reset(dones)
        collide_nn.reset(dones)
        place_nn.reset(dones)

    # ---- save per-grid-entry outcomes (success/failure + reason for every init) ----
    np.savez(os.path.join(args_cli.record_dir, "grid_outcomes.npz"),
             visited=g_visited, success=g_success, collided=g_collided, placed=g_placed,
             wrong_order=g_wrong_order, reason=g_reason.astype("U64"),
             object_xy=grid_object_xy, basket_xy=grid_basket_xy)
    import json as _json
    fail_idx = np.nonzero(g_visited & ~g_success)[0]
    reason_counts = {}
    for i in fail_idx:
        reason_counts[g_reason[i]] = reason_counts.get(g_reason[i], 0) + 1
    with open(os.path.join(args_cli.record_dir, "grid_outcomes.json"), "w") as f:
        _json.dump({
            "grid_file": env_cfg.init_grid_file,
            "n_grid": int(n_grid),
            "n_visited": int(g_visited.sum()),
            "n_success": int(g_success.sum()),
            "n_failure": int(len(fail_idx)),
            "failure_reason_counts": reason_counts,
            "success_grid_idx": np.nonzero(g_success)[0].tolist(),
            "failures": [{"grid_idx": int(i), "reason": g_reason[i],
                          "collided": bool(g_collided[i]), "placed": bool(g_placed[i])}
                         for i in fail_idx],
        }, f, indent=2)

    print(f"[collect] draining {len(results)} pending writes...", flush=True)
    for r in results:
        r.update(r.pop("_future").result())
    writer_pool.shutdown(wait=True)
    if results:
        u._write_dataset_manifest(args_cli.record_dir, results, len(results), rec_fps)
    print(f"[collect] DONE | grid {int(g_visited.sum())}/{n_grid} visited, "
          f"{int(g_success.sum())} success ({int(len(fail_idx))} fail), saved {ep_counter} demos "
          f"-> {args_cli.record_dir}", flush=True)

    env.close()
    torch.cuda.empty_cache()
    simulation_app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel LeRobot collection of pick->collide->place demos.")
    parser.add_argument("--num_envs", type=int, default=16, help="Parallel envs (cameras are the bottleneck).")
    parser.add_argument("--num_episodes", type=int, default=200, help="Successful episodes to save.")
    parser.add_argument("--record_dir", type=str, default="logs/collect_pick_collide_place/v1")
    parser.add_argument("--cam_w", type=int, default=256, help="Square camera resolution (cam_w=cam_h).")
    parser.add_argument("--collide_threshold", type=float, default=0.1,
                        help="Can displacement (m) that switches collide->place.")
    parser.add_argument("--record_every", type=int, default=1, help="Record every Nth control step (fps=60/N).")
    parser.add_argument("--max_frames", type=int, default=200, help="Truncate each episode after N control steps.")
    parser.add_argument("--env_spacing", type=float, default=20.0,
                        help="Distance (m) between envs so neighbors don't leak into each camera.")
    parser.add_argument("--rerenders", type=int, default=0, help="num_rerenders_on_reset (RTX settle).")
    parser.add_argument("--antialiasing", type=str, default=None,
                        choices=["Off", "FXAA", "DLSS", "TAA", "DLAA"], help="RenderCfg AA mode.")
    parser.add_argument("--write_workers", type=int, default=8, help="ProcessPool mp4/npz encoders.")
    parser.add_argument("--max_pending", type=int, default=128, help="Max in-flight unwritten episodes.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--headless", action="store_true", default=True)
    args_cli = parser.parse_args()
    main(args_cli)
