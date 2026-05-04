# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

import datetime
import os
from typing import Literal

# we import this here to avoid GLIBCXX_3.4.30 error in Isaac Sim 5.1
from isaaclab.app import AppLauncher
from isaaclab_eureka import EUREKA_ROOT_DIR
from isaaclab_eureka.config import (
    DIRECT_WORKFLOW_INITIAL_PROMPT,
    DIRECT_WORKFLOW_INITIAL_PROMPT_CURRICULUM,
    DIRECT_WORKFLOW_TASK_PROMPT,
    TASK_FAILURE_FEEDBACK_PROMPT,

    TASK_SUCCESS_PRE_FEEDBACK_PROMPT,
    LLM_TASK_FEEDBACK_PROMPT,
    TASK_SUCCESS_POST_FEEDBACK_PROMPT,
    TASKS_CFG,
    REPLAY_FEEDBACK_PROMPT,
    BEST_ITERATION_FEEDBACK_PROMPT,
)
from isaaclab_eureka.managers import EurekaTaskManager, LLMManager
from isaaclab_eureka.utils import load_tensorboard_logs


class Eureka:
    """Orchestrates the training of the RL agent using the LLM."""

    def __init__(
        self,
        task: str,
        checkpoint_to_resume_from: str | None,
        device: str = "cuda",
        env_seed: int = 42,
        rl_library: Literal["rsl_rl", "rl_games"] = "rsl_rl",
        max_training_iterations: int = 100,
        feedback_subsampling: int = 10,
        temperature: float = 1.0,
        gpt_model: str = "gpt-4",
        num_parallel_runs: int = 1,
        replay: bool = False,
        keep_best_reward: bool = False,
        resume: bool = False,
        use_vlm: bool = True,
        consider_stage_in_success_metric: bool = False,
    ):
        """Initialize the Eureka class.

        Args:

            task: The task to train the agent on.
            device: The device to run the training on.
            env_seed: The seed to use for the environment
            rl_library: The RL library to use for training.
            max_training_iterations: The maximum number of training iterations for the RL agent.
            feedback_subsampling: The subsampling of the metrics given as feedack to the LLM.
            temperature: The temperature to use for the GPT model.
            gpt_model: The GPT model to use.
            num_parallel_runs: The number of runs to execute in parallel.
        """

        # Load the task description and success metric
        self._debug = True
        self.num_stages=5
        self.keep_best_reward = keep_best_reward
        self.smooth_metric = False
        self.replay = replay
        if task in TASKS_CFG:
            task_description = TASKS_CFG[task]["description"]
            self._success_metric_string = TASKS_CFG[task].get("success_metric")
            self._success_metric_to_win = TASKS_CFG[task].get("success_metric_to_win")
            self._success_metric_tolerance = TASKS_CFG[task].get("success_metric_tolerance")
            # Task config may override the constructor argument; constructor arg is the fallback.
            self.consider_stage_in_success_metric = TASKS_CFG[task].get(
                "consider_stage_in_success_metric", consider_stage_in_success_metric
            )
        else:
            raise ValueError(
                f"Task configuration for {task} not found in the `TASKS_CFG` dictionary in config/tasks.py."
            )

        self._task_description = task_description
        self._feedback_subsampling = feedback_subsampling
        self._num_processes = num_parallel_runs
        import multiprocessing
        multiprocessing.set_start_method("spawn")
        print("[INFO]: Setting up the LLM Manager...")
        self._llm_manager = LLMManager(
            gpt_model=gpt_model,
            num_suggestions=self._num_processes,
            temperature=temperature,
            system_prompt=DIRECT_WORKFLOW_INITIAL_PROMPT,
            resume=resume
        )

        print("[INFO]: Setting up the Task Manager...")
        import inspect
        task_manager_class = getattr(self, "_task_manager_class", EurekaTaskManager)
        tm_kwargs = dict(
            task=task,
            checkpoint_to_resume_from=checkpoint_to_resume_from,
            device=device,
            env_seed=env_seed,
            rl_library=rl_library,
            num_processes=self._num_processes,
            max_training_iterations=max_training_iterations,
            success_metric_string=self._success_metric_string,
            replay=replay,
        )
        # Pass use_vlm only if the task manager class accepts it
        if "use_vlm" in inspect.signature(task_manager_class.__init__).parameters:
            tm_kwargs["use_vlm"] = use_vlm
        self._task_manager = task_manager_class(**tm_kwargs)
        self.use_vlm = use_vlm

        # Logging
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._log_dir = os.path.join(EUREKA_ROOT_DIR, "logs", "eureka", task, timestamp)
        os.makedirs(self._log_dir)

        # We import here because doing this before launching Kit causes GLIBCXX errors
        from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

        self._tensorboard_writer = TensorboardSummaryWriter(log_dir=self._log_dir, flush_secs=10)

    def run(self, max_eureka_iterations: int):
        """Run the Eureka training loop.

        Args:
            max_eureka_iterations: The maximum number of Eureka iterations to run.
        """
        # We import here because doing this before launching Kit causes GCC_12.0 errors
        import numpy as np

        # Initial prompts
        user_prompt = DIRECT_WORKFLOW_TASK_PROMPT.format(
            task_description=self._task_description,
            success_metric = self._success_metric_string,
            success_metric_to_win=self._success_metric_to_win,
            get_observations_method_as_string=self._task_manager.get_observations_method_as_string,
        )
        if self._debug:
            print(user_prompt)
            # print(self._task_manager._success_metric_string)
        # The assistant prompt is used to feed the previous LLM output back into the LLM
        assistant_prompt = None

        # The best run across all iterations
        best_run_results = {"success_metric": None}

        for iter in range(max_eureka_iterations):
            print(f"\n{'#' * 20} Running Eureka Iteration {iter} {'#' * 20} \n")
            # Generate the GPT reward methods

            llm_outputs = self._llm_manager.prompt(user_prompt=user_prompt, assistant_prompt=assistant_prompt)
            gpt_reward_method_strings = llm_outputs["reward_strings"]
            # Log the llm outputs
            for idx, gpt_reward_method_string in enumerate(gpt_reward_method_strings):
                self._tensorboard_writer.add_text(f"Run_{idx}/raw_llm_output", llm_outputs["raw_outputs"][idx], iter)
            # Train the RL agent
            results = self._task_manager.train(gpt_reward_method_strings)
            # Evaluate the results
            iter_best_success_metric = None
            best_run_idx = 0
            for idx, result in enumerate(results): # idx is for different runs
                if not result["success"]:
                    user_feedback_prompt = TASK_FAILURE_FEEDBACK_PROMPT.format(traceback_msg=result["exception"])
                else:
                    # Compute the performance metrics
                    eureka_task_feedback, success_metric_max, rewards_correlation = self._get_eureka_task_feedback(
                        result["log_dir"], self._feedback_subsampling
                    )
                    if self.use_vlm:
                        vlm_output_file = os.path.join(result["log_dir"], "vlm_output.txt")
                        with open(vlm_output_file, "r") as f:
                            llm_task_feedback = f.read()
                    else:
                        llm_task_feedback = ""

                    if self.replay:
                        replay_eureka_task_feedback = self._get_replay_task_feedback(
                            result["log_dir"]
                        )
                    else:
                        replay_eureka_task_feedback = ""
                    if self.keep_best_reward and best_run_results["success_metric"] is not None and best_run_results["success_metric"] > 0.1:
                        best_iter_feeback = BEST_ITERATION_FEEDBACK_PROMPT.format(
                            success_metric=best_run_results["success_metric"],
                            gpt_reward_method=best_run_results["gpt_reward_method"],
                            task_feedback=best_run_results["task_feedback"],
                        )
                    else:
                        best_iter_feeback = ""
                    # Generate the user feedback prompt
                    vlm_feedback_section = LLM_TASK_FEEDBACK_PROMPT.format(llm_task_feedback=llm_task_feedback) if llm_task_feedback else ""
                    user_feedback_prompt = (
                        TASK_SUCCESS_PRE_FEEDBACK_PROMPT.format(feedback_subsampling=self._feedback_subsampling)
                        + eureka_task_feedback
                        + vlm_feedback_section
                        + replay_eureka_task_feedback
                        + best_iter_feeback
                        + TASK_SUCCESS_POST_FEEDBACK_PROMPT
                    )
                    print(user_feedback_prompt)
                    # Store the results
                    results[idx]["eureka_task_feedback"] = eureka_task_feedback
                    results[idx]["success_metric_max"] = success_metric_max
                    results[idx]["rewards_correlation"] = rewards_correlation

                    # Check the best performing metric, determined by the minimum distance from the win target
                    if success_metric_max is not None and ( # TODO: this need to be fixed
                        iter_best_success_metric is None
                        or np.abs(success_metric_max - self._success_metric_to_win)
                        < np.abs(iter_best_success_metric - self._success_metric_to_win)
                    ):
                        # Store the best run for this iteration
                        iter_best_success_metric = success_metric_max
                        best_run_idx = idx

                        # Store the best metric across all iterations
                        if best_run_results["success_metric"] is None or (
                            np.abs(iter_best_success_metric - self._success_metric_to_win)
                            < np.abs(best_run_results["success_metric"] - self._success_metric_to_win)
                        ):
                            best_run_results["success_metric"] = iter_best_success_metric
                            best_run_results["gpt_reward_method"] = gpt_reward_method_strings[idx]
                            best_run_results["task_feedback"] = eureka_task_feedback

                # Add the prompts
                results[idx]["user_prompt"] = user_feedback_prompt
                results[idx]["assistant_prompt"] = llm_outputs["raw_outputs"][idx]

            self._log_iteration_results(iter, results)

            if (
                best_run_results["success_metric"] is not None
                and np.abs(best_run_results["success_metric"] - self._success_metric_to_win)
                < self._success_metric_tolerance
            ):
                print(f"Task solved with success metric: {best_run_results['success_metric']}")
                break

            assistant_prompt = results[best_run_idx]["assistant_prompt"]
            user_prompt = results[best_run_idx]["user_prompt"]

        self._log_final_results(best_run_results)
        # Close the task manager
        self._task_manager.close()

    def _get_eureka_task_feedback(self, log_dir: str, feedback_subsampling: int) -> tuple[str, float, float]:
        """Get the feedback for the Eureka task.

        Args:
            log_dir: The directory where the tensorboard logs are stored.
            feedback_subsampling: The subsampling of the metrics' trajectories.
        Returns:
            A tuple containing the feedback string, the maximum of the success metric, and the correlation between the oracle and GPT rewards.
        """
        # We import here because doing this before launching Kit causes GCC_12.0 errors
        import numpy as np

        data = load_tensorboard_logs(log_dir)
        # Compute correlation between the oracle and GPT rewards
        eureka_rewards = np.array(
            next((data[key] for key in data if key.endswith("Eureka/eureka_total_rewards")), None)
        )
        oracle_rewards = np.array(
            next((data[key] for key in data if key.endswith("Eureka/oracle_total_rewards")), None)
        )
        # Sometimes, the tensorboard logging is not complete, we take the minimum length between the two buffers
        min_length = min(eureka_rewards.shape[0], oracle_rewards.shape[0])
        rewards_correlation = np.corrcoef(eureka_rewards[:min_length], oracle_rewards[:min_length])[0, 1]

        success_metric_max = None
        # Make a summary of each plot in the tensorboard logs
        total_feed_back_string = ""
        if self.consider_stage_in_success_metric:
            stage_weights = [0.0, 0.05, 0.1, 0.3, 1.0] # The last one is not used
            stage_dict = {}
        for metric_name, metric_data in data.items():
            if "Eureka/" in metric_name:
                # Remove the first two data points as they are usually outliers
                metric_data = metric_data[100:]
                metric_name = metric_name.split("Eureka/", 1)[-1]
                metric_min = min(metric_data)
                metric_max = max(metric_data)
                metric_mean = sum(metric_data) / len(metric_data)
                # Best metric is the one closest to the target
                # Smooth the data with a moving average to get a stable max
                
                if metric_name == "success_metric": # if not considering multiple stages: then use success metric as the only judge
                    if not self.consider_stage_in_success_metric: # success metric is only task score when stages are not used
                        metric_name = "task_score"
                    success_metric_max = self._compute_best_metric(metric_data)
                data_string = [f"{data:.2f}" for data in metric_data[::feedback_subsampling]]
                feedback_string = (
                    f"{metric_name}: {data_string}, Min: {metric_min:.2f}, Max: {metric_max:.2f}, Mean:"
                    f" {metric_mean:.2f} \n"
                )
                if "Eureka/success_metric" in data and metric_name == "Eureka/oracle_total_rewards":
                    # If success metric is available, we do not provide the oracle feedback
                    feedback_string = ""
                if self.consider_stage_in_success_metric and metric_name == "success_metric":
                    # Raw success_metric suppressed; replaced below by weighted task_score
                    feedback_string = ""
                total_feed_back_string += feedback_string

                # If using stages for task score: do the following:
                if self.consider_stage_in_success_metric:
                    if metric_name == "success_metric":
                        stage_dict["success_metric"] = metric_data
                    elif metric_name.startswith("stage_"):
                        stage_dict[metric_name] = metric_data

        if self.consider_stage_in_success_metric:
            metric_name = "task_score"
            # Build weighted stage score dict
            weighted_metric_data = np.zeros(len(stage_dict["success_metric"]))
            for key, metric_data in stage_dict.items():
                if key == "success_metric":
                    weighted_metric_data += np.array(metric_data)
                if key.startswith("stage_"):
                    try:
                        stage_idx = int(key.split("stage_")[-1])
                        if 0 < stage_idx < len(stage_weights):
                            weighted_metric_data += np.array(metric_data) * stage_weights[stage_idx-1]
                    except ValueError:
                        continue
            success_metric_max = self._compute_best_metric(weighted_metric_data)
            metric_min = min(weighted_metric_data)
            metric_max = max(weighted_metric_data)
            metric_mean = sum(weighted_metric_data) / len(weighted_metric_data)
            data_string = [f"{data:.2f}" for data in weighted_metric_data[::feedback_subsampling]]
            feedback_string = (
                f"{metric_name}: {data_string}, Min: {metric_min:.2f}, Max: {metric_max:.2f}, Mean:"
                f" {metric_mean:.2f} \n"
            )
            stage_weight_desc = ", ".join(
                [f"stage_{i+1}×{stage_weights[i]}" for i in range(len(stage_weights) - 1)]
            )
            feedback_string = (
                f"task_score replaces raw success_metric. Formula: success_metric + ({stage_weight_desc})."
                f" Best value closest to win target is selected.\n"
            ) + feedback_string
            total_feed_back_string += feedback_string
            
        total_feed_back_string += f"\nThe desired task_score to win is: {self._success_metric_to_win:.2f}\n"
        return total_feed_back_string, success_metric_max, rewards_correlation

    def _log_iteration_results(self, iter: int, results: list):
        """Log the results of the iteration."""
        for idx, result in enumerate(results):
            print(f"{'*' * 20} Iteration {iter} / Process: {idx} {'*' * 20}")
            if result["success"]:
                print(f"Training successful with the following metrics:\n{result['eureka_task_feedback']}")
                print(f"Reward correlation with oracle rewards: {result['rewards_correlation']}")
            else:
                print(f"Training failed with the following exception:\n{result['exception']}\n")

        # write the iterations results to file
        with open(f"{self._log_dir}/eureka_iterations.txt", "a") as f:
            for idx, result in enumerate(results):
                f.write(f"{'#' * 20} Iteration: {iter} {'#' * 20}\n\n")
                f.write(f"{'*' * 20} Run: {idx} {'*' * 20}\n")
                f.write(f"- GPT reward method {result['assistant_prompt']}\n")
                if result["success"]:
                    f.write(f"Training successful with the following metrics:\n{result['eureka_task_feedback']}\n")
                    f.write(f"Reward correlation with oracle rewards:\n{result['rewards_correlation']}\n")
                    self._tensorboard_writer.add_scalar(f"Run_{idx}/success_metric", result["success_metric_max"], iter)
                else:
                    f.write(f"Training failed with the following exception:\n{result['exception']}\n")
                    self._tensorboard_writer.add_scalar(f"Run_{idx}/success_metric", 0.0, iter)
                self._tensorboard_writer.add_text(f"Run_{idx}/run_feedback", result["user_prompt"], iter)
                f.write("\n")

    def _log_final_results(self, best_run_results: dict):
        """Log the final results of the Eureka run."""
        output = ""
        if best_run_results["success_metric"] is not None:
            output += f"- Success metric: {best_run_results['success_metric']}\n"
            output += f"- GPT reward method: {best_run_results['gpt_reward_method']}\n"
            output += f"- Task metrics:\n{best_run_results['task_feedback']}\n"
        else:
            output += "- No successful training run\n"

        print("Final results:\n", output)
        import socket
        host_name = socket.gethostname()
        with open(f"{self._log_dir}/eureka_final_result.txt_{host_name}", "w") as f:
            f.write(output)


    def _get_replay_task_feedback(self, log_dir: str) -> str:
        """Get the feedback for the Eureka task.

        Args:
            log_dir: The directory where the tensorboard logs are stored.
            feedback_subsampling: The subsampling of the metrics' trajectories.
        Returns:
            A tuple containing the feedback string, the maximum of the success metric, and the correlation between the oracle and GPT rewards.
        """
        # We import here because doing this before launching Kit causes GCC_12.0 errors
        import numpy as np

        data = load_tensorboard_logs(log_dir)
        # Make a summary of each plot in the tensorboard logs
        replay_feed_back_string = ""
        for metric_name, metric_data in data.items():
            if "Replay/" in metric_name:
                metric_name = metric_name.split("Replay/", 1)[-1]
                data_string = f"{metric_data[0]:.2f}"
                feedback_string = (
                    f"{metric_name}: {data_string}"
                )
                if "Replay/success_metric" in data and metric_name == "Replay/oracle_total_rewards":
                    # If success metric is available, we do not provide the oracle feedback
                    feedback_string = ""
                replay_feed_back_string += feedback_string
        full_replay_feedback_string = REPLAY_FEEDBACK_PROMPT.format(replay_feedback_string=replay_feed_back_string)
        return full_replay_feedback_string
    

    def _compute_best_metric(self,metric_data) -> float:
        import numpy as np
        if self.smooth_metric:
            window_size = 5
            if len(metric_data) >= window_size:
                smoothed = np.convolve(metric_data, np.ones(window_size)/window_size, mode='same')
                metric_best = smoothed[np.abs(smoothed - self._success_metric_to_win).argmin()]
        else:
            metric_best = metric_data[np.abs(np.array(metric_data) - self._success_metric_to_win).argmin()]
        return metric_best
