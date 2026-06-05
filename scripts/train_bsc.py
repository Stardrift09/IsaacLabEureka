#!/usr/bin/env python3
"""BSC cluster entry point: patches LLMManager → FileLLMManager, then runs train_sequential_vlm.py.

All arguments are forwarded to the original train script unchanged.
Any future changes to train_sequential_vlm.py take effect automatically.
"""

import os
import sys

# Importing FileLLMManager triggers isaaclab_eureka/__init__.py → eureka.py →
# "from isaaclab_eureka.managers import LLMManager", binding the original class.
# "from X import Y" makes a local copy — patching the module attribute alone doesn't
# update that copy. We must patch eureka.py's namespace directly after import.
from isaaclab_eureka.managers.file_llm_manager import FileLLMManager
import isaaclab_eureka.managers as _managers
import isaaclab_eureka.managers.llm_manager as _llm_mod
import isaaclab_eureka.eureka as _eureka

_managers.LLMManager = FileLLMManager
_llm_mod.LLMManager = FileLLMManager
_eureka.LLMManager = FileLLMManager  # overwrite the already-bound local name in eureka.py

# --- Offline Isaac assets (BSC has no internet) -----------------------------
# Point Isaac's cloud asset root at our local mirror under offline_assets/. The
# task spawns USDs from {ISAAC_NUCLEUS_DIR} = {cloud}/Isaac, so cloud must be the
# directory that *contains* the "Isaac/" level: .../offline_assets/Assets/Isaac/5.1.
#
# Mechanism: kit reads "--/persistent/isaac/asset_root/cloud=<path>" from sys.argv at
# SimulationApp startup. The RL/obs/VLM workers are spawned with start_method="spawn",
# which copies the parent's sys.argv into each child (verified), so AppLauncher in every
# worker picks this up — even though this wrapper's top-level code never runs in a child.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_asset_root = os.path.join(_repo_root, "offline_assets", "Assets", "Isaac", "5.1")
_asset_setting = f"--/persistent/isaac/asset_root/cloud={_asset_root}"
sys.argv.append(_asset_setting)

# train_sequential_vlm.py calls argparse's strict parse_args(), which would reject the
# kit "--/..." token. Strip such tokens before parsing (they stay in sys.argv for kit).
import argparse as _argparse

_orig_parse_args = _argparse.ArgumentParser.parse_args


def _parse_args_ignoring_kit_settings(self, args=None, namespace=None):
    if args is None:
        args = [a for a in sys.argv[1:] if not a.startswith("--/")]
    return _orig_parse_args(self, args=args, namespace=namespace)


_argparse.ArgumentParser.parse_args = _parse_args_ignoring_kit_settings
# ---------------------------------------------------------------------------

import runpy

script = os.path.join(os.path.dirname(__file__), "train_sequential_vlm.py")
sys.argv[0] = script
runpy.run_path(script, run_name="__main__")
