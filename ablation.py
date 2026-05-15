"""Ablation runner for train_sequential_vlm.py.

Usage:
    python ablation.py --task TurnOnTheStove --ablation baseline
    python ablation.py --task TurnOnTheStove --ablation all --dry_run
    python ablation.py --task TurnOnTheStove --ablation no_vlm,no_replay
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent / "scripts" / "train_sequential_vlm.py"

# Base config: matches train_sequential_vlm.py defaults
BASE = {
    "max_eureka_iterations": 30,
    "max_training_iterations": 2000,
    "num_parallel_runs": 1,
    "temperature": 1.0,
    "gpt_model": "gpt-5.4",
    "rl_library": "rsl_rl",
    "feedback_subsampling": 100,
    "env_seed": 42,
}

# Each ablation is (description, flag overrides)
# Flag overrides: key=argname value=new_value, or key=flag value=True for boolean flags
ABLATIONS: dict[str, tuple[str, dict]] = {
    "baseline": (
        "Full pipeline: VLM + replay + keep_best_reward",
        {},
    ),
    "no_vlm": (
        "Disable VLM evaluation — measures VLM contribution",
        {"no_vlm": True},
    ),
    "no_replay": (
        "Disable demo replay — measures replay contribution",
        {"no_replay": True},
    ),
    "no_keep_best_reward": (
        "Don't carry best reward to next iteration",
        {"no_keep_best_reward": True},
    ),
}

# Boolean flags (presence = True, absence = False)
BOOL_FLAGS = {
    "no_vlm",
    "no_replay",
    "no_keep_best_reward",
    "consider_stage_in_success_metric",
    "resume",
}


def build_cmd(task: str, device: str, overrides: dict) -> list[str]:
    merged = {**BASE, **overrides}
    cmd = [sys.executable, str(SCRIPT), "--task", task, "--device", device]
    for k, v in merged.items():
        if k in BOOL_FLAGS:
            if v:
                cmd.append(f"--{k}")
        else:
            cmd.extend([f"--{k}", str(v)])
    return cmd


def run_ablation(name: str, task: str, device: str, dry_run: bool) -> None:
    desc, overrides = ABLATIONS[name]
    cmd = build_cmd(task, device, overrides)
    print(f"\n{'='*60}")
    print(f"Ablation : {name}")
    print(f"Task     : {task}")
    print(f"Desc     : {desc}")
    print(f"Command  : {' '.join(cmd)}")
    print(f"{'='*60}")
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablations for train_sequential_vlm.py")
    parser.add_argument("--task", type=str, default="TurnOnTheStove", help="Isaac Lab task name")
    parser.add_argument(
        "--ablation",
        type=str,
        default="baseline",
        help=f"Ablation name, comma-separated list, or 'all'. Available: {', '.join(ABLATIONS)}",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Training device")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    parser.add_argument("--list", action="store_true", help="List all ablations and exit")
    args = parser.parse_args()

    if args.list:
        print("Available ablations:")
        for name, (desc, overrides) in ABLATIONS.items():
            print(f"  {name:<25} {desc}")
            if overrides:
                print(f"  {'':25} overrides: {overrides}")
        return

    if args.ablation == "all":
        names = list(ABLATIONS.keys())
    else:
        names = [n.strip() for n in args.ablation.split(",")]

    for name in names:
        if name not in ABLATIONS:
            print(f"Unknown ablation '{name}'. Run with --list to see options.")
            sys.exit(1)

    for name in names:
        run_ablation(name, args.task, args.device, args.dry_run)


if __name__ == "__main__":
    main()
