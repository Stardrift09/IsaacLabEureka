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

    def __init__(self, gpt_model: str, num_suggestions: int, temperature: float, system_prompt: str, resume: bool):
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
        self._debug = False
        self.resume = resume

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

    @staticmethod
    def _sanitize(text: str) -> str:
        """Remove characters that make JSON encoding fail (null bytes, lone surrogates)."""
        # Encode to UTF-8 with surrogate replacement, then decode back
        return text.replace('\x00', '').encode('utf-8', errors='replace').decode('utf-8')

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
            self._prompts.append({"role": "assistant", "content": self._sanitize(assistant_prompt)})
        self._prompts.append({"role": "user", "content": self._sanitize(user_prompt)})
        if self.resume:
            print("Resuming......")
            self.resume=False
            try:
                string_to_resume_from = """def _get_rewards_eureka(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    import torch

    eps = 1e-6

    # -----------------------
    # Sanitize environment state
    # -----------------------
    obj_pos = torch.nan_to_num(self.target_object.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
    obj_vel = torch.nan_to_num(self.target_object.data.root_lin_vel_w, nan=0.0, posinf=0.0, neginf=0.0)
    site_pos = torch.nan_to_num(self.target_site.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
    site_vel = torch.nan_to_num(self.target_site.data.root_lin_vel_w, nan=0.0, posinf=0.0, neginf=0.0)

    joint_pos = torch.nan_to_num(self._robot.data.joint_pos, nan=0.0, posinf=0.0, neginf=0.0)
    joint_vel = torch.nan_to_num(self._robot.data.joint_vel, nan=0.0, posinf=0.0, neginf=0.0)
    actions = torch.nan_to_num(self.actions, nan=0.0, posinf=0.0, neginf=0.0)
    prev_actions = torch.nan_to_num(self.prev_actions, nan=0.0, posinf=0.0, neginf=0.0)
    to_desired_rot = torch.nan_to_num(self.to_desired_rot, nan=0.0, posinf=0.0, neginf=0.0)
    stage = torch.nan_to_num(self.stage.float(), nan=0.0, posinf=0.0, neginf=0.0)

    grasped = torch.nan_to_num(self.grasped.float(), nan=0.0, posinf=0.0, neginf=0.0)
    high_enough = torch.nan_to_num(self.high_enough.float(), nan=0.0, posinf=0.0, neginf=0.0)

    # -----------------------
    # Geometry / task state
    # -----------------------
    obj_xy = obj_pos[:, :2]
    site_xy = site_pos[:, :2]
    xy_err = obj_xy - site_xy
    xy_dist = torch.linalg.norm(xy_err, dim=-1)

    site_radius = torch.clamp(
        torch.nan_to_num(self.target_site_radius, nan=0.05, posinf=0.05, neginf=0.05),
        min=eps,
    )

    basket_top_z = torch.nan_to_num(self.target_site_corners_world[1, 2], nan=0.0, posinf=0.0, neginf=0.0)
    basket_bottom_z = torch.nan_to_num(self.target_site_corners_world[0, 2], nan=0.0, posinf=0.0, neginf=0.0)
    obj_z = obj_pos[:, 2]

    low_enough = obj_z < basket_top_z
    inside_site = (xy_dist ** 2) < (site_radius ** 2)
    success = inside_site & low_enough

    # The object should be above the opening before release.
    # Use a moderate release height target above the basket.
    release_target_z = basket_top_z + 0.10
    z_release_err = torch.abs(obj_z - release_target_z)

    # Tight alignment gate for release. We want the object centered over the basket.
    release_xy_threshold = 0.35 * site_radius
    release_height_min = basket_top_z + 0.03
    release_ready = (xy_dist < release_xy_threshold) & (obj_z > release_height_min) & (grasped > 0.5)

    # Gripper semantics: last dim > 0 means opening
    gripper_cmd = torch.clamp(actions[:, -1], -1.0, 1.0)
    open_cmd = torch.clamp(gripper_cmd, min=0.0)
    close_cmd = torch.clamp(-gripper_cmd, min=0.0)

    # Velocities
    rel_vel = obj_vel - site_vel
    rel_speed = torch.linalg.norm(rel_vel, dim=-1)
    obj_speed = torch.linalg.norm(obj_vel, dim=-1)

    # -----------------------
    # Temperatures for shaped rewards
    # -----------------------
    transport_temp = 0.35
    precise_xy_temp = 0.08
    release_height_temp = 0.06
    orientation_temp = 0.12
    settle_temp = 0.20
    centered_drop_temp = 0.05
    limit_temp = 0.08

    # -----------------------
    # Stage-aware reward shaping
    # Main diagnosis from prior run:
    # - agent already reaches stage_4 almost always
    # - but does not open gripper (open_amount ~ 0, release_bonus ~ 0)
    # - reward over-favors holding/alignment hovering
    # So:
    #   1) remove strong keep-holding incentive
    #   2) add explicit open-when-ready reward
    #   3) penalize continuing to hold once release-ready has been reached
    #   4) keep success very dominant
    # -----------------------

    # While grasped: move object over basket in xy.
    transport_xy = torch.exp(-xy_dist / transport_temp) * grasped

    # Precise centering over the basket opening while grasped.
    precise_xy = torch.exp(-xy_dist / precise_xy_temp) * grasped

    # Reach a good release height above the opening while grasped.
    release_height = torch.exp(-z_release_err / release_height_temp) * grasped

    # Keep a reasonable wrist/object orientation while grasped.
    quat_w = torch.clamp(to_desired_rot[:, 0], -1.0, 1.0)
    orientation = torch.exp(-(1.0 - quat_w) / orientation_temp) * grasped

    # Smooth object motion near release and after drop.
    settle = torch.exp(-rel_speed / settle_temp)

    # Strong reward for opening when release-ready.
    open_when_ready = open_cmd * release_ready.float()

    # Penalize not opening after being correctly positioned.
    hold_when_ready_penalty = (1.0 - open_cmd) * release_ready.float() * grasped

    # Penalize opening away from basket / too early.
    premature_open_penalty = open_cmd * (~release_ready).float()

    # After release, reward staying centered over basket while descending.
    descending = ((grasped < 0.5) & (~success) & (obj_z >= basket_bottom_z)).float()
    centered_drop = torch.exp(-xy_dist / centered_drop_temp) * descending

    # Penalty if object has been released but is outside basket footprint.
    released_outside_penalty = ((grasped < 0.5) & (~inside_site) & (~success)).float()

    # Success reward should dominate hovering rewards by a large margin.
    success_reward = success.float()

    # Stage shaping: preserve provided stage semantics, biasing later stages.
    stage_progress = (
        0.10 * stage[:, 2] +   # grasped
        0.20 * stage[:, 3] +   # high enough
        0.35 * stage[:, 4]     # inside_site / over basket region
    )

    # -----------------------
    # Progress memory using helper_variable
    # 0: release-ready has ever been achieved this episode
    # 1: best precise-xy while grasped
    # 2: success latch
    # -----------------------
    prev_release_ready_latch = torch.nan_to_num(self.helper_variable[:, 0], nan=0.0, posinf=1.0, neginf=0.0)
    prev_best_precise = torch.nan_to_num(self.helper_variable[:, 1], nan=0.0, posinf=1.0, neginf=0.0)
    prev_success_latch = torch.nan_to_num(self.helper_variable[:, 2], nan=0.0, posinf=1.0, neginf=0.0)

    release_ready_latch = torch.maximum(prev_release_ready_latch, release_ready.float())
    best_precise = torch.maximum(prev_best_precise, precise_xy.detach())
    precise_improvement = torch.clamp(best_precise - prev_best_precise, min=0.0)
    success_latch = torch.maximum(prev_success_latch, success.float())

    self.helper_variable[:, 0] = torch.nan_to_num(release_ready_latch, nan=0.0, posinf=1.0, neginf=0.0)
    self.helper_variable[:, 1] = torch.nan_to_num(best_precise, nan=0.0, posinf=1.0, neginf=0.0)
    self.helper_variable[:, 2] = torch.nan_to_num(success_latch, nan=0.0, posinf=1.0, neginf=0.0)

    # Extra pressure to actually release after the agent has ever achieved release-ready.
    delayed_release_penalty = release_ready_latch * grasped * (1.0 - open_cmd)

    # -----------------------
    # Regularization
    # -----------------------
    joint_speed_penalty = torch.mean(joint_vel ** 2, dim=-1)
    action_rate_penalty = torch.mean((actions - prev_actions) ** 2, dim=-1)

    dof_range = torch.clamp(self.robot_dof_upper_limits - self.robot_dof_lower_limits, min=eps)
    dist_to_lower = (joint_pos - self.robot_dof_lower_limits) / dof_range
    dist_to_upper = (self.robot_dof_upper_limits - joint_pos) / dof_range
    nearest_limit = torch.minimum(dist_to_lower, dist_to_upper)
    joint_limit_penalty = torch.mean(torch.exp(-nearest_limit / limit_temp), dim=-1)

    # -----------------------
    # Total reward
    # -----------------------
    reward = (
        0.35 * transport_xy
        + 0.55 * precise_xy
        + 0.35 * release_height
        + 0.15 * orientation
        + 0.10 * settle
        + 1.20 * open_when_ready
        + 0.20 * centered_drop
        + 0.10 * stage_progress
        + 0.15 * precise_improvement
        + 5.00 * success_reward
        - 0.80 * hold_when_ready_penalty
        - 0.60 * delayed_release_penalty
        - 0.50 * premature_open_penalty
        - 0.60 * released_outside_penalty
        - 0.02 * joint_speed_penalty
        - 0.04 * action_rate_penalty
        - 0.06 * joint_limit_penalty
    )

    reward = torch.nan_to_num(reward, nan=-1.0, posinf=5.0, neginf=-5.0)
    reward = torch.clamp(reward, -5.0, 5.0)

    assert torch.isfinite(reward).all(), "Non-finite reward detected in _get_rewards_eureka"

    individual_rewards_dict = {
        "transport_xy": transport_xy,
        "precise_xy": precise_xy,
        "release_height": release_height,
        "orientation": orientation,
        "settle": settle,
        "open_when_ready": open_when_ready,
        "hold_when_ready_penalty": -hold_when_ready_penalty,
        "delayed_release_penalty": -delayed_release_penalty,
        "premature_open_penalty": -premature_open_penalty,
        "centered_drop": centered_drop,
        "released_outside_penalty": -released_outside_penalty,
        "success": success_reward,
        "stage_progress": stage_progress,
        "precise_improvement": precise_improvement,
        "joint_speed_penalty": -joint_speed_penalty,
        "action_rate_penalty": -action_rate_penalty,
        "joint_limit_penalty": -joint_limit_penalty,
        "xy_dist": xy_dist,
        "obj_speed": obj_speed,
        "open_amount": open_cmd,
        "release_ready": release_ready.float(),
        "release_ready_latched": release_ready_latch,
        "success_latched": success_latch,
    }

    return reward, individual_rewards_dict
                """
                raw_outputs = [string_to_resume_from for i in range(self._num_suggestions)]
                reward_strings = raw_outputs
                # reward_strings = [string_to_resume_from]
                # raw_outputs = reward_strings            
                return {"reward_strings": reward_strings, "raw_outputs": raw_outputs}
            except:
                pass


        # The official Eureka code only keeps the last round of feedback
        if len(self._prompts) == 6:
            self._prompts.pop(2)
            self._prompts.pop(2)
        if self._debug:
            task= "alphabet_soup"
            if task == "franka":
                constant_reward_string = """def _get_rewards_eureka(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
                # increase rewards for getting the drawer open
                joint_pos_reward = self._cabinet.data.joint_pos[:, 1]
                # penalize for high velocity
                joint_vel_penalty = - torch.abs(self._cabinet.data.joint_vel[:, 1])
                # increase rewards for being closer to the drawer grasp position
                to_target_dist = torch.norm(self.drawer_grasp_pos - self.robot_grasp_pos, dim=-1)
                to_target_reward = - to_target_dist
                # calculate final reward
                reward = joint_pos_reward + joint_vel_penalty + to_target_reward
                # normalize reward with sigmoid function
                reward = torch.sigmoid(reward)
                # scale up reward to match the desired task score
                desired_task_score = 0.9
                reward = reward * desired_task_score
                # prepare individual rewards dict
                individual_rewards_dict = {
                    'joint_pos_reward': joint_pos_reward,
                    'joint_vel_penalty': joint_vel_penalty,
                    'to_target_reward': to_target_reward
                }
                # make sure reward and individual rewards are on the correct device
                reward = reward.to(self.device)
                for key in individual_rewards_dict:
                    individual_rewards_dict[key] = individual_rewards_dict[key].to(self.device)

                return reward, individual_rewards_dict
                """
            elif task == "alphabet_soup":
                constant_reward_string = """def _get_rewards_eureka(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    import torch

    eps = 1e-6

    # -----------------------
    # Sanitize environment state
    # -----------------------
    obj_pos = torch.nan_to_num(self.target_object.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
    obj_vel = torch.nan_to_num(self.target_object.data.root_lin_vel_w, nan=0.0, posinf=0.0, neginf=0.0)
    site_pos = torch.nan_to_num(self.target_site.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
    site_vel = torch.nan_to_num(self.target_site.data.root_lin_vel_w, nan=0.0, posinf=0.0, neginf=0.0)

    joint_pos = torch.nan_to_num(self._robot.data.joint_pos, nan=0.0, posinf=0.0, neginf=0.0)
    joint_vel = torch.nan_to_num(self._robot.data.joint_vel, nan=0.0, posinf=0.0, neginf=0.0)
    actions = torch.nan_to_num(self.actions, nan=0.0, posinf=0.0, neginf=0.0)
    prev_actions = torch.nan_to_num(self.prev_actions, nan=0.0, posinf=0.0, neginf=0.0)
    to_desired_rot = torch.nan_to_num(self.to_desired_rot, nan=0.0, posinf=0.0, neginf=0.0)
    stage = torch.nan_to_num(self.stage.float(), nan=0.0, posinf=0.0, neginf=0.0)

    grasped = torch.nan_to_num(self.grasped.float(), nan=0.0, posinf=0.0, neginf=0.0)
    high_enough = torch.nan_to_num(self.high_enough.float(), nan=0.0, posinf=0.0, neginf=0.0)

    # -----------------------
    # Geometry / task state
    # -----------------------
    obj_xy = obj_pos[:, :2]
    site_xy = site_pos[:, :2]
    xy_err = obj_xy - site_xy
    xy_dist = torch.linalg.norm(xy_err, dim=-1)

    site_radius = torch.clamp(
        torch.nan_to_num(self.target_site_radius, nan=0.05, posinf=0.05, neginf=0.05),
        min=eps,
    )

    basket_top_z = torch.nan_to_num(self.target_site_corners_world[1, 2], nan=0.0, posinf=0.0, neginf=0.0)
    basket_bottom_z = torch.nan_to_num(self.target_site_corners_world[0, 2], nan=0.0, posinf=0.0, neginf=0.0)
    obj_z = obj_pos[:, 2]

    low_enough = obj_z < basket_top_z
    inside_site = (xy_dist ** 2) < (site_radius ** 2)
    success = inside_site & low_enough

    # The object should be above the opening before release.
    # Use a moderate release height target above the basket.
    release_target_z = basket_top_z + 0.10
    z_release_err = torch.abs(obj_z - release_target_z)

    # Tight alignment gate for release. We want the object centered over the basket.
    release_xy_threshold = 0.35 * site_radius
    release_height_min = basket_top_z + 0.03
    release_ready = (xy_dist < release_xy_threshold) & (obj_z > release_height_min) & (grasped > 0.5)

    # Gripper semantics: last dim > 0 means opening
    gripper_cmd = torch.clamp(actions[:, -1], -1.0, 1.0)
    open_cmd = torch.clamp(gripper_cmd, min=0.0)
    close_cmd = torch.clamp(-gripper_cmd, min=0.0)

    # Velocities
    rel_vel = obj_vel - site_vel
    rel_speed = torch.linalg.norm(rel_vel, dim=-1)
    obj_speed = torch.linalg.norm(obj_vel, dim=-1)

    # -----------------------
    # Temperatures for shaped rewards
    # -----------------------
    transport_temp = 0.35
    precise_xy_temp = 0.08
    release_height_temp = 0.06
    orientation_temp = 0.12
    settle_temp = 0.20
    centered_drop_temp = 0.05
    limit_temp = 0.08

    # -----------------------
    # Stage-aware reward shaping
    # Main diagnosis from prior run:
    # - agent already reaches stage_4 almost always
    # - but does not open gripper (open_amount ~ 0, release_bonus ~ 0)
    # - reward over-favors holding/alignment hovering
    # So:
    #   1) remove strong keep-holding incentive
    #   2) add explicit open-when-ready reward
    #   3) penalize continuing to hold once release-ready has been reached
    #   4) keep success very dominant
    # -----------------------

    # While grasped: move object over basket in xy.
    transport_xy = torch.exp(-xy_dist / transport_temp) * grasped

    # Precise centering over the basket opening while grasped.
    precise_xy = torch.exp(-xy_dist / precise_xy_temp) * grasped

    # Reach a good release height above the opening while grasped.
    release_height = torch.exp(-z_release_err / release_height_temp) * grasped

    # Keep a reasonable wrist/object orientation while grasped.
    quat_w = torch.clamp(to_desired_rot[:, 0], -1.0, 1.0)
    orientation = torch.exp(-(1.0 - quat_w) / orientation_temp) * grasped

    # Smooth object motion near release and after drop.
    settle = torch.exp(-rel_speed / settle_temp)

    # Strong reward for opening when release-ready.
    open_when_ready = open_cmd * release_ready.float()

    # Penalize not opening after being correctly positioned.
    hold_when_ready_penalty = (1.0 - open_cmd) * release_ready.float() * grasped

    # Penalize opening away from basket / too early.
    premature_open_penalty = open_cmd * (~release_ready).float()

    # After release, reward staying centered over basket while descending.
    descending = ((grasped < 0.5) & (~success) & (obj_z >= basket_bottom_z)).float()
    centered_drop = torch.exp(-xy_dist / centered_drop_temp) * descending

    # Penalty if object has been released but is outside basket footprint.
    released_outside_penalty = ((grasped < 0.5) & (~inside_site) & (~success)).float()

    # Success reward should dominate hovering rewards by a large margin.
    success_reward = success.float()

    # Stage shaping: preserve provided stage semantics, biasing later stages.
    stage_progress = (
        0.10 * stage[:, 2] +   # grasped
        0.20 * stage[:, 3] +   # high enough
        0.35 * stage[:, 4]     # inside_site / over basket region
    )

    # -----------------------
    # Progress memory using helper_variable
    # 0: release-ready has ever been achieved this episode
    # 1: best precise-xy while grasped
    # 2: success latch
    # -----------------------
    prev_release_ready_latch = torch.nan_to_num(self.helper_variable[:, 0], nan=0.0, posinf=1.0, neginf=0.0)
    prev_best_precise = torch.nan_to_num(self.helper_variable[:, 1], nan=0.0, posinf=1.0, neginf=0.0)
    prev_success_latch = torch.nan_to_num(self.helper_variable[:, 2], nan=0.0, posinf=1.0, neginf=0.0)

    release_ready_latch = torch.maximum(prev_release_ready_latch, release_ready.float())
    best_precise = torch.maximum(prev_best_precise, precise_xy.detach())
    precise_improvement = torch.clamp(best_precise - prev_best_precise, min=0.0)
    success_latch = torch.maximum(prev_success_latch, success.float())

    self.helper_variable[:, 0] = torch.nan_to_num(release_ready_latch, nan=0.0, posinf=1.0, neginf=0.0)
    self.helper_variable[:, 1] = torch.nan_to_num(best_precise, nan=0.0, posinf=1.0, neginf=0.0)
    self.helper_variable[:, 2] = torch.nan_to_num(success_latch, nan=0.0, posinf=1.0, neginf=0.0)

    # Extra pressure to actually release after the agent has ever achieved release-ready.
    delayed_release_penalty = release_ready_latch * grasped * (1.0 - open_cmd)

    # -----------------------
    # Regularization
    # -----------------------
    joint_speed_penalty = torch.mean(joint_vel ** 2, dim=-1)
    action_rate_penalty = torch.mean((actions - prev_actions) ** 2, dim=-1)

    dof_range = torch.clamp(self.robot_dof_upper_limits - self.robot_dof_lower_limits, min=eps)
    dist_to_lower = (joint_pos - self.robot_dof_lower_limits) / dof_range
    dist_to_upper = (self.robot_dof_upper_limits - joint_pos) / dof_range
    nearest_limit = torch.minimum(dist_to_lower, dist_to_upper)
    joint_limit_penalty = torch.mean(torch.exp(-nearest_limit / limit_temp), dim=-1)

    # -----------------------
    # Total reward
    # -----------------------
    reward = (
        0.35 * transport_xy
        + 0.55 * precise_xy
        + 0.35 * release_height
        + 0.15 * orientation
        + 0.10 * settle
        + 1.20 * open_when_ready
        + 0.20 * centered_drop
        + 0.10 * stage_progress
        + 0.15 * precise_improvement
        + 5.00 * success_reward
        - 0.80 * hold_when_ready_penalty
        - 0.60 * delayed_release_penalty
        - 0.50 * premature_open_penalty
        - 0.60 * released_outside_penalty
        - 0.02 * joint_speed_penalty
        - 0.04 * action_rate_penalty
        - 0.06 * joint_limit_penalty
    )

    reward = torch.nan_to_num(reward, nan=-1.0, posinf=5.0, neginf=-5.0)
    reward = torch.clamp(reward, -5.0, 5.0)

    assert torch.isfinite(reward).all(), "Non-finite reward detected in _get_rewards_eureka"

    individual_rewards_dict = {
        "transport_xy": transport_xy,
        "precise_xy": precise_xy,
        "release_height": release_height,
        "orientation": orientation,
        "settle": settle,
        "open_when_ready": open_when_ready,
        "hold_when_ready_penalty": -hold_when_ready_penalty,
        "delayed_release_penalty": -delayed_release_penalty,
        "premature_open_penalty": -premature_open_penalty,
        "centered_drop": centered_drop,
        "released_outside_penalty": -released_outside_penalty,
        "success": success_reward,
        "stage_progress": stage_progress,
        "precise_improvement": precise_improvement,
        "joint_speed_penalty": -joint_speed_penalty,
        "action_rate_penalty": -action_rate_penalty,
        "joint_limit_penalty": -joint_limit_penalty,
        "xy_dist": xy_dist,
        "obj_speed": obj_speed,
        "open_amount": open_cmd,
        "release_ready": release_ready.float(),
        "release_ready_latched": release_ready_latch,
        "success_latched": success_latch,
    }

    return reward, individual_rewards_dict

                """
            else:
                raise NotImplementedError("Wrong task name in llm manager!")
                
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


    