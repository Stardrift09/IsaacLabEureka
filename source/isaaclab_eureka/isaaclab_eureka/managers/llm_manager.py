# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

import os
import re

import openai


class LLMManager:
    """Manager to interface with the LLM API.

    This class is responsible for interfacing with the LLM API to generate rewards.
    It establishes a connection either to native OpenAI API, or to the Azure OpenAI API.

    The Openai API relies on the following environment variables to be set:
    - For the native OpenAI API, the environment variable OPENAI_API_KEY must be set.
    - For the Azure OpenAI API, the environment variables AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set.
    """

    def __init__(self, gpt_model: str, num_suggestions: int, temperature: float, system_prompt: str):
        """Initialize the LLMManager

        Args:
            gpt_model: The model to use for the LLM API
            num_suggestions: The number of independent suggestions to generate
            temperature: The temperature to use for the LLM API
            system_prompt: The system prompt to provide to the LLM API
        """

        self._gpt_model = gpt_model
        self._num_suggestions = num_suggestions
        self._temperature = temperature
        self._prompts = [{"role": "system", "content": system_prompt}]
        self._debug = True

        if "AZURE_OPENAI_API_KEY" in os.environ:
            self._client = openai.AzureOpenAI(api_version="2024-02-01")
        elif "OPENAI_API_KEY" in os.environ:
            self._client = openai.OpenAI()
        else:
            raise RuntimeError("No Openai API key found in environment variables")

    def extract_code_from_response(self, response: str) -> str:
        """Extract the code component from the LLM response

        If the response contains a code block of the form "```python ... ```", extract the code block from the response.
        Otherwise, return an empty string.

        Args:
            response: The response from the LLM API
        """
        pattern = r"```python(.*?)```"
        result = re.findall(pattern, response, re.DOTALL)
        code_string = ""
        if result is not None and len(result) > 0:
            code_string = result[-1]
            # Remove leading newline characters
            code_string = code_string.lstrip("\n")
        return code_string

    def prompt(self, user_prompt: str, assistant_prompt: str = None) -> list[str]:
        """Call the LLM API to collect responses

        Args:
            user_prompt: The user prompt to provide to the LLM API
            assistant_prompt: The assistant prompt to provide to the LLM API

        Returns:
            A dictionary containing the reward strings and raw outputs from the LLM

        Raises:
            Exception: If there is an error with the LLM API
        """
        if assistant_prompt is not None:
            self._prompts.append({"role": "assistant", "content": assistant_prompt})
        self._prompts.append({"role": "user", "content": user_prompt})
        if self._debug:
            # print(self._prompts)
            pass
        # The official Eureka code only keeps the last round of feedback
        if len(self._prompts) == 6:
            self._prompts.pop(2)
            self._prompts.pop(2)
        if self._debug:
            constant_reward_string = """def _get_rewards_eureka(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            import torch

            device = self.device
            num_envs = self.num_envs
            eps = 1e-6

            # --------------------------------------------------------------------------
            # Gather and sanitize state
            # --------------------------------------------------------------------------
            joint_pos = torch.nan_to_num(self._robot.data.joint_pos, nan=0.0, posinf=0.0, neginf=0.0)
            joint_vel = torch.nan_to_num(self._robot.data.joint_vel, nan=0.0, posinf=0.0, neginf=0.0)
            cream_pos = torch.nan_to_num(self.rigid_objects["cream_cheese"].data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
            cream_height = cream_pos[:, 2]

            # Robot grasp position (assumed to be precomputed each step in env)
            robot_grasp_pos = torch.nan_to_num(self.robot_grasp_pos, nan=0.0, posinf=0.0, neginf=0.0)

            # Distance from gripper to object
            to_target = cream_pos - robot_grasp_pos
            dist_to_target = torch.norm(to_target, dim=-1)

            # --------------------------------------------------------------------------
            # Heuristic initial height estimate
            # --------------------------------------------------------------------------
            # Use the minimum height seen so far as a proxy for the initial height.
            # This avoids requiring self.cream_initial_height in the env class.
            if not hasattr(self, "cream_min_height"):
                # Initialize stateful min-height tracker on device
                self.cream_min_height = cream_height.clone().detach().to(device)
            else:
                # Update running minimum per env (for environments that aren't reset-aware,
                # this still provides a stable "rest" height close to the table)
                self.cream_min_height = torch.minimum(
                    self.cream_min_height.to(device),
                    cream_height
                )

            lift_start_height = self.cream_min_height

            # Height thresholds for "pick up"
            lift_success_margin = 0.05  # 5 cm above the (approx.) initial height
            lift_success_height = lift_start_height + lift_success_margin

            # Binary success: cream cheese has been lifted above threshold
            is_lifted = (cream_height > lift_success_height).float()

            # --------------------------------------------------------------------------
            # Reward components
            # --------------------------------------------------------------------------

            # 1. Distance shaping: encourage the gripper to move towards the cream cheese
            # Use an exponential shaping with temperature so reward stays in (0, 1]
            dist_temp = 10.0  # temperature for distance shaping
            dist_shaping = torch.exp(-dist_temp * torch.clamp(dist_to_target, 0.0, 0.5))  # clamp for stability
            w_dist = 1.0
            r_dist = w_dist * dist_shaping

            # 2. Lift shaping: reward higher object (smooth before full success)
            # Normalized height gain: 0 at initial, 1 at success height, clipped
            height_gain = (cream_height - lift_start_height) / (lift_success_margin + eps)
            height_gain = torch.nan_to_num(height_gain, nan=0.0, posinf=0.0, neginf=0.0)
            height_gain_clipped = torch.clamp(height_gain, 0.0, 1.0)

            lift_temp = 4.0
            # Map [0,1] -> (e^-4, 1]; more sensitive near top of range
            r_lift_shaping = torch.exp(lift_temp * (height_gain_clipped - 1.0))
            w_lift_shaping = 1.5
            r_lift = w_lift_shaping * r_lift_shaping

            # 3. Success bonus: once lifted, give a strong bonus to push towards full grasp
            w_success = 3.0
            r_success = w_success * is_lifted

            # 4. Smoothness / efficiency penalties
            # Joint velocity penalty (to avoid jitter and violent motions)
            vel_norm = torch.norm(joint_vel, dim=-1)
            vel_pen = vel_norm
            w_vel = -0.01
            r_vel = w_vel * vel_pen

            # 5. Linear distance penalty (complements exponential shaping and keeps
            # gradients non-zero when far away)
            w_far = -0.5
            r_far = w_far * dist_to_target

            # --------------------------------------------------------------------------
            # Combine rewards
            # --------------------------------------------------------------------------
            reward = r_dist + r_lift + r_success + r_vel + r_far

            # Sanitize and ensure shape
            reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

            if reward.ndim == 0:
                reward = reward * torch.ones(num_envs, device=device)
            else:
                reward = reward.view(num_envs)

            # Training stability check
            assert torch.isfinite(reward).all(), "Non-finite reward encountered"

            individual_rewards = {
                "r_dist": r_dist,
                "r_lift": r_lift,
                "r_success": r_success,
                "r_vel": r_vel,
                "r_far": r_far,
            }

            return reward, individual_rewards
            """
            reward_strings = [constant_reward_string]
            raw_outputs = reward_strings
        else:
            try:
                responses = self._client.chat.completions.create(
                    model=self._gpt_model,
                    messages=self._prompts,
                    temperature=self._temperature,
                    n=self._num_suggestions,
                )
            except Exception as e:
                raise RuntimeError("An error occurred while prompting the LLM") from e

            raw_outputs = [response.message.content for response in responses.choices]
            reward_strings = [self.extract_code_from_response(raw_output) for raw_output in raw_outputs]

        return {"reward_strings": reward_strings, "raw_outputs": raw_outputs}
