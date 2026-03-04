# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to train an RL agent with Isaac Lab Eureka."""

import argparse
import os

from isaaclab_eureka.eureka import Eureka
# import logging

# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
# )
# logger = logging.getLogger(__name__)

def main(args_cli):
    if args_cli.task in ["Isaac-Franka-Cabinet-Direct-v0","AToB"] and not args_cli.no_replay:
        args_cli.no_replay = True
        print("Current task doesn't have successful demos, setting replay to False")
    eureka = Eureka(
        task=args_cli.task,
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
        resume=args_cli.resume
    )

    eureka.run(max_eureka_iterations=args_cli.max_eureka_iterations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL agent with Eureka.")
    parser.add_argument("--no_keep_best_reward", action="store_true", help="Whether to keep the best reward function for better reward generation. It is more costly")
    parser.add_argument("--no_replay", action="store_true", help="Whether replay the task")
    parser.add_argument("--task", type=str, default="TestPutItInTheBasket", help="Name of the task.")
    parser.add_argument(
        "--num_parallel_runs", type=int, default=1, help="Number of Eureka runs to execute in parallel."
    )
    parser.add_argument("--device", type=str, default="cuda", help="The device to run training on.")
    parser.add_argument("--env_seed", type=int, default=42, help="The random seed to use for the environment.")
    parser.add_argument("--max_eureka_iterations", type=int, default=20, help="The number of Eureka iterations to run.")
    parser.add_argument(
        "--max_training_iterations",
        type=int,   
        default=1500,
        help="The number of RL training iterations to run for each Eureka iteration.",
    )
    parser.add_argument(
        "--feedback_subsampling",
        type=int,
        default=100,
        help="The subsampling of the metrics given as feedack to the LLM.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1,
        help="Controls the randomness of the GPT output (0 is deterministic, 1 is highly diverse).",
    )
    parser.add_argument("--gpt_model", type=str, default="gpt-5.2", help="The GPT model to use.")
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
        help="Resume training from checkpoint"
    )
    args_cli = parser.parse_args()

    # Check parameter validity
    if os.name == "nt" and args_cli.num_parallel_runs > 1:
        print(
            "[WARNING]: Running with num_parallel_runs > 1 is not supported on Windows. Setting num_parallel_runs = 1."
        )
        args_cli.num_parallel_runs = 1

    # Run the main function
    main(args_cli)
