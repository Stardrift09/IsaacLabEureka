"""
Fix a physics USD: remove extra PhysicsArticulationRootAPI from all prims except
the stage's defaultPrim (which is the intended articulation root).

Works for URDF-converted assets (root = worldBody, which is the defaultPrim) and
MJCF-converted assets (root = model name, which is the defaultPrim).

Run with:
    ./IsaacLab/isaaclab.sh -p scripts/fix_articulation_root_usd.py \
        --usd <path_to_physics_usd> --headless
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd", type=str, required=True, help="Path to the physics USD to fix.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import sys
from pxr import Usd, UsdPhysics

LOG = "/tmp/fix_articulation_root_usd.log"
log = open(LOG, "w")

usd_path = args_cli.usd
if not os.path.exists(usd_path):
    log.write(f"ERROR: {usd_path} not found.\n")
    log.flush()
    log.close()
    sys.stderr.write(f"[fix_articulation_root_usd] ERROR: file not found: {usd_path}\n")
    simulation_app.close()
    raise SystemExit(1)

log.write(f"USD: {usd_path}\n")
stage = Usd.Stage.Open(usd_path)
default_prim_path = "/" + stage.GetDefaultPrim().GetName()
log.write(f"defaultPrim: {default_prim_path}\n\n")

default_prim = stage.GetDefaultPrim()
needs_add = not default_prim.HasAPI(UsdPhysics.ArticulationRootAPI)
if needs_add:
    UsdPhysics.ArticulationRootAPI.Apply(default_prim)
    log.write(f"Added ArticulationRootAPI to defaultPrim: {default_prim_path}\n")

removed = []
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        path = str(prim.GetPath())
        if path != default_prim_path:
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            removed.append(path)
            log.write(f"Removed ArticulationRootAPI from: {path}\n")

if not removed and not needs_add:
    log.write("No extra ArticulationRootAPI found.\n")
else:
    stage.GetRootLayer().Save()
    log.write(f"Saved. Added={needs_add}, removed from {len(removed)} prim(s).\n")

log.write("\nRemaining ArticulationRootAPI prims:\n")
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        log.write(f"  {prim.GetPath()}\n")

log.flush()
log.close()
sys.stderr.write(f"[fix_articulation_root_usd] Done. See {LOG}\n")

simulation_app.close()
