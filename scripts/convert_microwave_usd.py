"""
Convert microwave MJCF → USD using Isaac Lab's MjcfConverter.
Source MJCF from workspace (libero/COMMON/articulated_objects/microwave/mjcf/).

After this, run fix and test:
    ./IsaacLab/isaaclab.sh -p scripts/fix_articulation_root_usd.py \
        --usd .../microwave/usd/configuration/microwave_physics.usd --headless
    ./IsaacLab/isaaclab.sh -p scripts/test_articulation_usd.py \
        --usd .../microwave/usd/microwave.usd --headless

Run with:
    ./IsaacLab/isaaclab.sh -p scripts/convert_microwave_usd.py --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.kit.app
omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.asset.importer.mjcf", True)

from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

MJCF_PATH = "/home/shaotongchen/workspace_eureka/IsaacLabEureka/libero/COMMON/articulated_objects/microwave/mjcf/microwave.xml"
USD_DIR   = "/home/shaotongchen/workspace_eureka/IsaacLabEureka/libero/COMMON/articulated_objects/microwave/usd"
USD_FILE  = "microwave.usd"

os.makedirs(USD_DIR, exist_ok=True)

cfg = MjcfConverterCfg(
    asset_path=MJCF_PATH,
    usd_dir=USD_DIR,
    usd_file_name=USD_FILE,
    fix_base=False,
    import_sites=True,
    force_usd_conversion=True,
    make_instanceable=False,
)

converter = MjcfConverter(cfg)
sys.stderr.write(f"[convert_microwave_usd] Done.\n  USD: {converter.usd_path}\n")

simulation_app.close()
