# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

"""Train with Eureka but run VLM evaluation in a separate subprocess.

After each RL training run the training IsaacLab instance is shut down before
the VLM evaluation starts, so both never compete for GPU memory simultaneously.

Usage (same args as train.py):
    python scripts/train_sequential_vlm.py --task TestPickItUp [...]
"""

import argparse
import os

from isaaclab_eureka.eureka import Eureka
from isaaclab_eureka.managers.eureka_task_manager_sequential_vlm import EurekaTaskManagerSequentialVLM


class EurekaSequentialVLM(Eureka):
    """Eureka subclass that uses EurekaTaskManagerSequentialVLM.

    The only difference from the base Eureka class is that RL training and VLM
    evaluation run in separate IsaacLab processes, freeing GPU memory between
    the two stages.
    """

    _task_manager_class = EurekaTaskManagerSequentialVLM


def main(args_cli):
    if args_cli.task in ["Isaac-Franka-Cabinet-Direct-v0", "AToB", "PickItUp", "PlaceInBasket", "PickItUpNoGrasp", "PickItUpNoGraspGamma99"] and not args_cli.no_replay:
        args_cli.no_replay = True
        print("Current task doesn't have successful demos, setting replay to False")

    eureka = EurekaSequentialVLM(
        task=args_cli.task,
        checkpoint_to_resume_from=args_cli.checkpoint_to_resume_from,
        rl_library=args_cli.rl_library,
        num_parallel_runs=args_cli.num_parallel_runs,
        device=args_cli.device,
        env_seed=args_cli.env_seed,
        max_training_iterations=args_cli.max_training_iterations,
        feedback_subsampling=args_cli.feedback_subsampling,
        temperature=args_cli.temperature,
        gpt_model=args_cli.gpt_model,
        replay=not args_cli.no_replay,
        keep_best_reward=not args_cli.no_keep_best_reward,
        resume=args_cli.resume,
        use_vlm=not args_cli.no_vlm,
        consider_stage_in_success_metric=args_cli.consider_stage_in_success_metric,
    )

    eureka.run(max_eureka_iterations=args_cli.max_eureka_iterations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an RL agent with Eureka (sequential VLM variant)."
    )
    parser.add_argument(
        "--no_keep_best_reward",
        action="store_true",
        help="Whether to keep the best reward function for better reward generation.",
    )
    parser.add_argument("--no_replay", action="store_true", help="Whether to replay the task.")
    parser.add_argument("--task", type=str, default="OpenDrawerAndPutCreamCheese", help="Name of the task.")
    parser.add_argument(
        "--checkpoint_to_resume_from",
        type=str,
        default=None,
        help="Path to the checkpoint where training should resume from.",
    )
    parser.add_argument(
        "--num_parallel_runs",
        type=int,
        default=1,
        help="Number of Eureka runs to execute in parallel.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="The device to run training on.")
    parser.add_argument(
        "--env_seed", type=int, default=42, help="The random seed to use for the environment."
    )
    parser.add_argument(
        "--max_eureka_iterations",
        type=int,
        default=30,
        help="The number of Eureka iterations to run.",
    )
    parser.add_argument(
        "--max_training_iterations",
        type=int,
        default=2000,
        help="The number of RL training iterations to run for each Eureka iteration.",
    )
    parser.add_argument(
        "--feedback_subsampling",
        type=int,
        default=100,
        help="The subsampling of the metrics given as feedback to the LLM.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1,
        help="Controls the randomness of the GPT output.",
    )
    parser.add_argument("--gpt_model", type=str, default="gpt-5.4", help="The GPT model to use.")
    parser.add_argument(
        "--rl_library",
        type=str,
        default="rsl_rl",
        choices=["rsl_rl", "rl_games"],
        help="The RL training library to use.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from a certain reward function.",
    )
    parser.add_argument(
        "--no_vlm",
        action="store_true",
        help="Skip VLM evaluation after training (no cameras loaded, saves GPU memory).",
    )
    parser.add_argument(
        "--consider_stage_in_success_metric",
        action="store_true",
        help="Use stage-weighted task score as success metric (overrides task config default).",
    )
    args_cli = parser.parse_args()

    # Allow passing --checkpoint_to_resume_from=None from the command line
    if args_cli.checkpoint_to_resume_from in ("None", "none", ""):
        args_cli.checkpoint_to_resume_from = None

    if os.name == "nt" and args_cli.num_parallel_runs > 1:
        print(
            "[WARNING]: Running with num_parallel_runs > 1 is not supported on Windows. "
            "Setting num_parallel_runs = 1."
        )
        args_cli.num_parallel_runs = 1

    main(args_cli)
