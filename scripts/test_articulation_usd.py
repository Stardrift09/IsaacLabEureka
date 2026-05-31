"""
Test script: load any USD as an Isaac Lab Articulation and print joint info.

Run with:
    ./IsaacLab/isaaclab.sh -p scripts/test_articulation_usd.py --usd <path>
    ./IsaacLab/isaaclab.sh -p scripts/test_articulation_usd.py --usd <path> --headless

Results written to /tmp/test_articulation_usd.log (kit suppresses stdout in headless mode).
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd", type=str, required=True, help="Path to the USD to load.")
parser.add_argument("--num_steps", type=int, default=200, help="Sim steps to run.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext

LOG = "/tmp/test_articulation_usd.log"
_log = open(LOG, "w", buffering=1)

def log(msg=""):
    _log.write(msg + "\n")
    _log.flush()
    sys.stderr.write(msg + "\n")


def design_scene(usd_path):
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg())
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())

    cfg = ArticulationCfg(
        prim_path="/World/Asset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.4),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=1000.0,
                stiffness=0.0,
                damping=1.0,
            ),
        },
    )
    return Articulation(cfg=cfg)


def main():
    log(f"USD: {args_cli.usd}")
    sim_cfg = sim_utils.SimulationCfg(dt=1 / 60)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.0, 2.0, 2.0], target=[0.0, 0.0, 0.4])

    asset = design_scene(args_cli.usd)
    sim.reset()

    log("=" * 60)
    log("ARTICULATION — JOINT INFO")
    log("=" * 60)

    joint_names = asset.joint_names
    if not joint_names:
        log("  !! No movable joints found (all fixed, or USD has no physics joints).")
    else:
        limits = asset.data.joint_limits
        for i, name in enumerate(joint_names):
            lo = limits[0, i, 0].item()
            hi = limits[0, i, 1].item()
            log(f"  [{i}] {name:40s}  limits: [{lo:.4f}, {hi:.4f}]")

    log("=" * 60)
    log(f"Total movable joints: {len(joint_names)}")

    if joint_names:
        limits = asset.data.joint_limits
        log("\n── Paste into ArticulationCfg ──")
        log("joint_pos={")
        for name in joint_names:
            log(f'    "{name}": 0.0,')
        log("}")
        log("\nactuators={")
        for i, name in enumerate(joint_names):
            lo = limits[0, i, 0].item()
            hi = limits[0, i, 1].item()
            log(f'    # "{name}"  range [{lo:.4f}, {hi:.4f}]')
        log("}")

    log(f"\nStepping sim for {args_cli.num_steps} steps...")
    count = 0
    ok = True
    try:
        while simulation_app.is_running() and count < args_cli.num_steps:
            sim.step()
            asset.update(sim_cfg.dt)
            count += 1
            if count % 50 == 0 and joint_names:
                log(f"  step {count:4d} | joint_pos: {asset.data.joint_pos.tolist()}")
    except Exception as e:
        log(f"EXCEPTION during sim: {e}")
        ok = False

    log(f"\n{'SUCCESS' if ok else 'FAILED'} — {count}/{args_cli.num_steps} steps completed.")
    _log.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
