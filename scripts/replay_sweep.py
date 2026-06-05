"""Sweep controller/gain combinations for the PickItUp physics replay.

Runs the batch replay (headless, no render) for many controller + gain settings,
soft -> rigid, for both OSC and joint-PD, then ranks them by task-success rate.
Each config is evaluated on the first ``--episodes`` demos (subset, to stay
tractable); rerun the winner with scripts/replay_all.py for the full 50.

Outputs per config:  <log_dir>/<config_name>/summary.{txt,csv}
Master ranking:       <log_dir>/sweep_results.csv  (+ printed table)

Usage:
    python scripts/replay_sweep.py                 # 10 episodes/config
    python scripts/replay_sweep.py --episodes 5
"""

import os
import torch

from isaaclab_eureka.utils import eureka_root_dir


# ----------------------------------------------------------------------
# the sweep grid (soft -> rigid). Edit freely.
#   OSC:      (kp, damping_ratio)   -- LIBERO default is (150, 1.0)
#   joint_pd: (kp, kd) applied to all 7 arm joints
# ----------------------------------------------------------------------
def build_configs():
    configs = []

    def osc(kp, dr):
        configs.append(dict(name=f"osc_kp{kp}_dr{dr}", controller="osc",
                            osc_kp=float(kp), osc_dr=float(dr), pd_kp=None, pd_kd=None))

    def pd(kp, kd):
        configs.append(dict(name=f"pd_kp{kp}_kd{kd}", controller="joint_pd",
                            osc_kp=None, osc_dr=None, pd_kp=float(kp), pd_kd=float(kd)))

    # --- OSC: stiffness soft -> rigid, critically damped ---
    for kp in [25, 50, 100, 150, 250, 400, 600, 800]:
        osc(kp, 1.0)
    # --- OSC: damping ratio under/over at two stiffnesses ---
    for dr in [0.5, 0.7, 1.5, 2.0]:
        osc(150, dr)
    for dr in [0.7, 1.5]:
        osc(400, dr)

    # --- joint_pd: stiffness soft -> rigid (kd ~ critically damped) ---
    for kp, kd in [(100, 20), (200, 30), (300, 40), (500, 60), (700, 90),
                   (1000, 120), (1500, 160), (2000, 200), (3000, 260)]:
        pd(kp, kd)
    # --- joint_pd: under / over damped at a stiff setpoint ---
    pd(1000, 60)
    pd(1000, 240)

    return configs


def aggregate(results):
    n = max(len(results), 1)
    n_term = sum(r["terminated_step"] >= 0 for r in results)
    n_grasp = sum(r["grasped_step"] >= 0 for r in results)
    return {
        "episodes": len(results),
        "success": n_term,
        "success_rate": n_term / n,
        "grasp": n_grasp,
        "grasp_rate": n_grasp / n,
        "mean_lift": sum(r["lift"] for r in results) / n,
        "mean_track_err": sum(r["mean_track_err"] for r in results) / n,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10, help="episodes per config (subset of 50)")
    parser.add_argument("--log_dir", type=str, default=None)
    args = parser.parse_args()

    task = "ReplayPickItUpOneEnv"
    root = eureka_root_dir()
    log_dir = args.log_dir if args.log_dir else os.path.join(root, "logs", "replay_sweep")
    os.makedirs(log_dir, exist_ok=True)

    from isaaclab.app import AppLauncher
    from isaaclab_eureka.utils import get_freest_gpu

    device = f"cuda:{get_freest_gpu()}"
    app_launcher = AppLauncher(headless=True, device=device)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg: DirectRLEnvCfg = parse_env_cfg(task)
    env_cfg.sim.device = device
    env_cfg.scene.num_envs = 1
    env = gym.make(task, cfg=env_cfg)
    env = env.unwrapped
    env.reset()

    configs = build_configs()
    rows = []
    for i, cfg in enumerate(configs):
        print(f"\n########## sweep {i + 1}/{len(configs)}: {cfg['name']} ##########", flush=True)
        # override the env's controller + gains for this config
        env.REPLAY_CONTROLLER = cfg["controller"]
        env.OSC_KP = cfg["osc_kp"] if cfg["osc_kp"] is not None else env.OSC_KP
        env.OSC_DAMPING_RATIO = cfg["osc_dr"] if cfg["osc_dr"] is not None else env.OSC_DAMPING_RATIO
        env.PD_KP = cfg["pd_kp"]
        env.PD_KD = cfg["pd_kd"]

        results = env.run_replay_all(os.path.join(log_dir, cfg["name"]), num_episodes=args.episodes)
        agg = aggregate(results)
        rows.append({**cfg, **agg})

    # rank: success_rate desc, then mean_lift desc, then mean_track_err asc
    rows.sort(key=lambda r: (-r["success_rate"], -r["mean_lift"], r["mean_track_err"]))

    # write master CSV
    csv_path = os.path.join(log_dir, "sweep_results.csv")
    with open(csv_path, "w") as f:
        f.write("rank,name,controller,osc_kp,osc_dr,pd_kp,pd_kd,episodes,"
                "success,success_rate,grasp,grasp_rate,mean_lift,mean_track_err\n")
        for rank, r in enumerate(rows, 1):
            f.write(f"{rank},{r['name']},{r['controller']},{r['osc_kp']},{r['osc_dr']},"
                    f"{r['pd_kp']},{r['pd_kd']},{r['episodes']},{r['success']},"
                    f"{r['success_rate']:.3f},{r['grasp']},{r['grasp_rate']:.3f},"
                    f"{r['mean_lift']:.4f},{r['mean_track_err']:.4f}\n")

    # printed ranking table
    table = [f"\n{'=' * 92}",
             f"SWEEP RANKING  ({len(configs)} configs x {args.episodes} episodes)  ->  {csv_path}",
             f"{'=' * 92}",
             f"{'rank':>4}  {'config':<20} {'success':>8} {'grasp':>7} {'mean_lift':>10} {'track_err':>10}"]
    for rank, r in enumerate(rows, 1):
        table.append(f"{rank:>4}  {r['name']:<20} "
                     f"{r['success']:>3}/{r['episodes']:<4} "
                     f"{r['grasp']:>3}/{r['episodes']:<3} "
                     f"{r['mean_lift']:>+10.3f} {r['mean_track_err']:>10.4f}")
    table.append("=" * 92)
    report = "\n".join(table)
    print(report, flush=True)
    with open(os.path.join(log_dir, "sweep_ranking.txt"), "w") as f:
        f.write(report + "\n")

    print("sweep finished, exiting")
    env.close()
    torch.cuda.empty_cache()
    env.sim.stop()
    env.sim.clear()
    simulation_app.close()
