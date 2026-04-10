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
        if self.resume:
            print("Resuming......")
            self.resume=False
            try:
                string_to_resume_from = """def _get_rewards_eureka(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    import torch

    device = self.device

    # -----------------------------
    # Sanitize commonly used states
    # -----------------------------
    target_pos = torch.nan_to_num(self.target_object.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
    site_pos = torch.nan_to_num(self.target_site.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
    grasp_pos = torch.nan_to_num(self.robot_grasp_pos, nan=0.0, posinf=0.0, neginf=0.0)
    target_to_hand = torch.nan_to_num(self.target_to_hand_pos, nan=0.0, posinf=0.0, neginf=0.0)
    site_to_target = torch.nan_to_num(self.site_to_target_pos, nan=0.0, posinf=0.0, neginf=0.0)
    corners_target_obj_to_hand = torch.nan_to_num(self.corners_target_obj_to_hand_pos, nan=0.0, posinf=0.0, neginf=0.0)
    to_desired_rot = torch.nan_to_num(self.to_desired_rot, nan=0.0, posinf=0.0, neginf=0.0)
    stage = torch.nan_to_num(self.stage.float(), nan=0.0, posinf=0.0, neginf=0.0)

    # Velocities for smoothness / drop detection
    obj_lin_vel = torch.nan_to_num(self.target_object.data.root_lin_vel_w, nan=0.0, posinf=0.0, neginf=0.0)
    site_lin_vel = torch.nan_to_num(self.target_site.data.root_lin_vel_w, nan=0.0, posinf=0.0, neginf=0.0)
    tcp_vel = (
        torch.nan_to_num(self._robot.data.body_link_lin_vel_w[:, self.left_finger_body_idx], nan=0.0, posinf=0.0, neginf=0.0)
        + torch.nan_to_num(self._robot.data.body_link_lin_vel_w[:, self.right_finger_body_idx], nan=0.0, posinf=0.0, neginf=0.0)
    ) * 0.5
    rel_vel = torch.nan_to_num(obj_lin_vel - tcp_vel, nan=0.0, posinf=0.0, neginf=0.0)
    site_to_target_vel = torch.nan_to_num(obj_lin_vel - site_lin_vel, nan=0.0, posinf=0.0, neginf=0.0)

    # -----------------------------
    # Geometry / success quantities
    # -----------------------------
    site_height = torch.nan_to_num(
        self.target_site_corners_world[1, 2] - self.target_site_corners_world[0, 2],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    site_height = torch.clamp(site_height, min=0.0)

    obj_xy = target_pos[:, :2]
    site_xy = site_pos[:, :2]
    xy_dist = torch.norm(obj_xy - site_xy, dim=-1)
    xy_dist = torch.nan_to_num(xy_dist, nan=1e3, posinf=1e3, neginf=1e3)

    target_site_radius = float(self.target_site_radius)
    inside_site = xy_dist < target_site_radius
    low_enough = target_pos[:, 2] < site_height
    success = inside_site & low_enough

    # Height margin above basket rim for stage-2/3 gating
    lift_margin = 0.05
    above_site_height = target_pos[:, 2] > (site_height + lift_margin)

    # -----------------------------
    # Progress tracking with helper_variable
    # helper_variable[:, 0] : grasp hold counter
    # helper_variable[:, 1] : stage 2 achieved flag
    # helper_variable[:, 2] : stage 3 achieved flag
    # helper_variable[:, 3] : success latch
    # -----------------------------
    if self.helper_variable.shape[1] >= 4:
        grasp_like = (
            (torch.norm(target_to_hand, dim=-1) < 0.08)
            & (corners_target_obj_to_hand.view(self.num_envs, -1, 3).norm(dim=-1).mean(dim=-1) < 0.12)
        )
        self.helper_variable[:, 0] = torch.where(
            grasp_like,
            torch.clamp(self.helper_variable[:, 0] + 1.0, max=1000.0),
            torch.zeros_like(self.helper_variable[:, 0]),
        )
        self.helper_variable[:, 1] = torch.maximum(
            self.helper_variable[:, 1],
            (stage >= 2).float(),
        )
        self.helper_variable[:, 2] = torch.maximum(
            self.helper_variable[:, 2],
            (stage >= 3).float(),
        )
        self.helper_variable[:, 3] = torch.maximum(
            self.helper_variable[:, 3],
            success.float(),
        )

    # -----------------------------
    # Stage masks:
    # only shape current stage after previous is finished
    # -----------------------------
    stage0_mask = (stage < 1).float()
    stage1_mask = ((stage >= 1) & (stage < 2)).float()
    stage2_mask = ((stage >= 2) & (stage < 3)).float()
    stage3_mask = (stage >= 3).float()

    # -----------------------------
    # Smooth bounded shaping terms
    # -----------------------------
    temp_reach = 0.10
    temp_grasp = 0.08
    temp_rot = 0.20
    temp_lift = 0.10
    temp_align_xy = 0.12
    temp_drop = 0.06
    temp_still = 0.50

    # Stage 0: approach and orient for grasp
    hand_dist = torch.norm(target_to_hand, dim=-1)
    hand_dist = torch.nan_to_num(hand_dist, nan=1e3, posinf=1e3, neginf=1e3)
    reach_raw = torch.exp(-torch.clamp(hand_dist / temp_reach, 0.0, 50.0))

    corner_dist = corners_target_obj_to_hand.view(self.num_envs, -1, 3).norm(dim=-1).mean(dim=-1)
    corner_dist = torch.nan_to_num(corner_dist, nan=1e3, posinf=1e3, neginf=1e3)
    grasp_preshape_raw = torch.exp(-torch.clamp(corner_dist / temp_grasp, 0.0, 50.0))

    rot_w = torch.clamp(to_desired_rot[:, 0], -1.0, 1.0)
    rot_err = 1.0 - rot_w
    rot_raw = torch.exp(-torch.clamp(rot_err / temp_rot, 0.0, 50.0))

    # Stage 1: once grasped, lift above basket height
    lift_gap = torch.clamp((site_height + lift_margin) - target_pos[:, 2], min=0.0)
    lift_raw = torch.exp(-torch.clamp(lift_gap / temp_lift, 0.0, 50.0))
    grasp_stability_raw = torch.exp(-torch.clamp(torch.norm(rel_vel, dim=-1) / temp_still, 0.0, 50.0))

    # Stage 2: move above basket in XY while staying high
    align_xy_raw = torch.exp(-torch.clamp(xy_dist / temp_align_xy, 0.0, 50.0))
    height_hold_raw = above_site_height.float()

    # Stage 3: reward controlled drop into basket only when above basket
    z_to_site_bottom = torch.clamp(target_pos[:, 2] - site_height, min=0.0)
    drop_raw = torch.exp(-torch.clamp(z_to_site_bottom / temp_drop, 0.0, 50.0))
    obj_still_raw = torch.exp(-torch.clamp(torch.norm(site_to_target_vel, dim=-1) / temp_still, 0.0, 50.0))

    # -----------------------------
    # Gate each stage reward strictly
    # -----------------------------
    stage0_quality = 0.55 * reach_raw + 0.30 * grasp_preshape_raw + 0.15 * rot_raw
    stage0_reward = 1.0 * stage0_mask * stage0_quality

    # no stage-1 shaping before stage 1 is entered
    stage1_quality = 0.75 * lift_raw + 0.25 * grasp_stability_raw
    stage1_reward = 1.2 * stage1_mask * stage1_quality

    # no stage-2 shaping before stage 2 is entered
    stage2_quality = align_xy_raw * height_hold_raw
    stage2_reward = 1.5 * stage2_mask * stage2_quality

    # no stage-3 shaping before stage 3 is entered
    ready_to_drop = (xy_dist < target_site_radius * 0.8).float()
    stage3_quality = ready_to_drop * (0.7 * drop_raw + 0.3 * obj_still_raw)
    stage3_reward = 1.8 * stage3_mask * stage3_quality

    # Sparse milestone bonuses to encourage long-horizon completion
    bonus_stage1 = 0.4 * (stage >= 1).float()
    bonus_stage2 = 0.7 * (stage >= 2).float()
    bonus_stage3 = 1.0 * (stage >= 3).float()
    success_bonus = 2.5 * success.float()

    # Mild penalty for moving object far from hand before grasp
    pregrasp_drift_penalty = -0.10 * stage0_mask * torch.clamp(torch.norm(obj_lin_vel, dim=-1), 0.0, 2.0)

    # Total reward
    reward = (
        stage0_reward
        + stage1_reward
        + stage2_reward
        + stage3_reward
        + bonus_stage1
        + bonus_stage2
        + bonus_stage3
        + success_bonus
        + pregrasp_drift_penalty
    )

    reward = torch.nan_to_num(reward, nan=0.0, posinf=5.0, neginf=-5.0)
    reward = torch.clamp(reward, -5.0, 5.0)

    assert torch.isfinite(reward).all(), "Non-finite reward encountered in _get_rewards_eureka."

    individual_rewards_dict = {
        "stage0_reward": torch.nan_to_num(stage0_reward, nan=0.0, posinf=5.0, neginf=-5.0),
        "stage1_reward": torch.nan_to_num(stage1_reward, nan=0.0, posinf=5.0, neginf=-5.0),
        "stage2_reward": torch.nan_to_num(stage2_reward, nan=0.0, posinf=5.0, neginf=-5.0),
        "stage3_reward": torch.nan_to_num(stage3_reward, nan=0.0, posinf=5.0, neginf=-5.0),
        "bonus_stage1": torch.nan_to_num(bonus_stage1, nan=0.0, posinf=5.0, neginf=-5.0),
        "bonus_stage2": torch.nan_to_num(bonus_stage2, nan=0.0, posinf=5.0, neginf=-5.0),
        "bonus_stage3": torch.nan_to_num(bonus_stage3, nan=0.0, posinf=5.0, neginf=-5.0),
        "success_bonus": torch.nan_to_num(success_bonus, nan=0.0, posinf=5.0, neginf=-5.0),
        "pregrasp_drift_penalty": torch.nan_to_num(pregrasp_drift_penalty, nan=0.0, posinf=5.0, neginf=-5.0),
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

    # ------------------------------------------------------------------
    # Sanitize state
    # ------------------------------------------------------------------
    obj_pos = torch.nan_to_num(self.target_object.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
    obj_vel = torch.nan_to_num(self.target_object.data.root_lin_vel_w, nan=0.0, posinf=0.0, neginf=0.0)
    site_pos = torch.nan_to_num(self.target_site.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
    site_vel = torch.nan_to_num(self.target_site.data.root_lin_vel_w, nan=0.0, posinf=0.0, neginf=0.0)
    hand_pos = torch.nan_to_num(self.robot_grasp_pos, nan=0.0, posinf=0.0, neginf=0.0)
    to_desired_rot = torch.nan_to_num(self.to_desired_rot, nan=0.0, posinf=0.0, neginf=0.0)
    stage = torch.nan_to_num(self.stage.float(), nan=0.0, posinf=0.0, neginf=0.0)
    grasped = torch.nan_to_num(self.grasped.float(), nan=0.0, posinf=0.0, neginf=0.0)
    high_enough = torch.nan_to_num(self.high_enough.float(), nan=0.0, posinf=0.0, neginf=0.0)
    joint_pos = torch.nan_to_num(self._robot.data.joint_pos, nan=0.0, posinf=0.0, neginf=0.0)
    joint_vel = torch.nan_to_num(self._robot.data.joint_vel, nan=0.0, posinf=0.0, neginf=0.0)
    actions = torch.nan_to_num(self.actions, nan=0.0, posinf=0.0, neginf=0.0)
    prev_actions = torch.nan_to_num(self.prev_actions, nan=0.0, posinf=0.0, neginf=0.0)
    helper = torch.nan_to_num(self.helper_variable, nan=0.0, posinf=0.0, neginf=0.0)

    tcp_vel = 0.5 * (
        torch.nan_to_num(self._robot.data.body_link_lin_vel_w[:, self.left_finger_body_idx], nan=0.0, posinf=0.0, neginf=0.0)
        + torch.nan_to_num(self._robot.data.body_link_lin_vel_w[:, self.right_finger_body_idx], nan=0.0, posinf=0.0, neginf=0.0)
    )

    # ------------------------------------------------------------------
    # Geometry and task conditions
    # ------------------------------------------------------------------
    target_to_hand = torch.nan_to_num(obj_pos - hand_pos, nan=0.0, posinf=0.0, neginf=0.0)
    hand_obj_dist = torch.linalg.norm(target_to_hand, dim=-1)
    hand_obj_xy_dist = torch.linalg.norm(target_to_hand[:, :2], dim=-1)
    hand_obj_z_dist = torch.abs(target_to_hand[:, 2])

    obj_xy = obj_pos[:, :2]
    site_xy = site_pos[:, :2]
    obj_to_site_xy = obj_xy - site_xy
    obj_site_xy_dist = torch.linalg.norm(obj_to_site_xy, dim=-1)

    basket_top_z = float(torch.nan_to_num(self.target_site_corners_world[1, 2], nan=0.0, posinf=0.0, neginf=0.0).item())
    basket_radius = max(float(self.target_site_radius), eps)

    low_enough = obj_pos[:, 2] < basket_top_z
    inside_site = (obj_site_xy_dist ** 2) < (basket_radius ** 2)
    success = (inside_site & low_enough).float()

    rel_obj_hand_vel = torch.linalg.norm(torch.nan_to_num(obj_vel - tcp_vel, nan=0.0, posinf=0.0, neginf=0.0), dim=-1)
    obj_speed = torch.linalg.norm(obj_vel, dim=-1)
    stage_idx = torch.argmax(stage, dim=-1).float()

    # ------------------------------------------------------------------
    # Helper memory layout
    # 0 prev_stage_idx
    # 1 ever_grasped
    # 2 ever_lifted
    # 3 prev_success
    # 4 prev_grasped
    # 5 best_obj_height
    # 6 best_transport_xy
    # 7 best_pregrasp
    # 8 best_place
    # ------------------------------------------------------------------
    prev_stage_idx = helper[:, 0]
    ever_grasped_prev = helper[:, 1]
    ever_lifted_prev = helper[:, 2]
    prev_success = helper[:, 3]
    prev_grasped = helper[:, 4]
    best_obj_height_prev = helper[:, 5]
    best_transport_xy_prev = helper[:, 6]
    best_pregrasp_prev = helper[:, 7]
    best_place_prev = helper[:, 8]

    # ------------------------------------------------------------------
    # Analysis-driven redesign:
    # - pregrasp_reward dominated and saturated -> reduce weight and make it progress-based too
    # - lift_stage_reward nearly constant -> make lift explicitly depend on object height progress
    # - transport_reward too small / weakly optimized -> strengthen XY-to-basket shaping after grasp/lift
    # - release/success never reached -> add explicit "place while grasped" shaping and larger success bonus
    # - stage_regression penalty small but unnecessary for exploration -> remove from total
    # - demonstrations show success_bonus is the main discriminator; keep it strong
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Stage 1: Reach + orient for grasp
    # ------------------------------------------------------------------
    temp_reach = 0.12
    reach_reward = torch.exp(-hand_obj_dist / temp_reach)

    temp_xy = 0.06
    xy_align_reward = torch.exp(-hand_obj_xy_dist / temp_xy)

    temp_z = 0.05
    z_align_reward = torch.exp(-hand_obj_z_dist / temp_z)

    temp_rot = 0.20
    rot_w = torch.clamp(to_desired_rot[:, 0], -1.0, 1.0)
    rot_reward = torch.exp(-(1.0 - rot_w) / temp_rot)

    temp_rel_vel = 0.35
    still_reward = torch.exp(-rel_obj_hand_vel / temp_rel_vel)

    pregrasp_score = (
        0.30 * reach_reward
        + 0.30 * xy_align_reward
        + 0.15 * z_align_reward
        + 0.20 * rot_reward
        + 0.05 * still_reward
    )
    best_pregrasp = torch.maximum(best_pregrasp_prev, pregrasp_score)
    pregrasp_progress = torch.clamp(best_pregrasp - best_pregrasp_prev, min=0.0, max=1.0)

    # ------------------------------------------------------------------
    # Stage 2: Grasp + lift
    # Use object height progress directly because previous lift reward saturated.
    # ------------------------------------------------------------------
    lift_clearance = basket_top_z + 0.08
    obj_height_above_top = torch.clamp(obj_pos[:, 2] - basket_top_z, min=0.0)
    lift_goal_gap = torch.clamp(lift_clearance - obj_pos[:, 2], min=0.0)

    temp_lift_gap = 0.08
    lift_height_reward = torch.exp(-lift_goal_gap / temp_lift_gap)

    best_obj_height = torch.maximum(best_obj_height_prev, obj_height_above_top)
    height_progress = torch.clamp(best_obj_height - best_obj_height_prev, min=0.0, max=0.05) / 0.05

    temp_obj_stable = 0.50
    obj_stable_reward = torch.exp(-obj_speed / temp_obj_stable)

    lift_stage_reward = (
        0.35 * grasped
        + 0.35 * lift_height_reward
        + 0.20 * height_progress
        + 0.10 * obj_stable_reward
    )

    # ------------------------------------------------------------------
    # Stage 3: Move object above basket while keeping grasp
    # Strengthen transport because previous term was too weak and flat.
    # ------------------------------------------------------------------
    temp_transport_xy = 0.10
    basket_xy_reward = torch.exp(-obj_site_xy_dist / temp_transport_xy)

    above_basket_target_z = basket_top_z + 0.10
    above_basket_z_err = torch.abs(obj_pos[:, 2] - above_basket_target_z)

    temp_transport_z = 0.08
    basket_z_reward = torch.exp(-above_basket_z_err / temp_transport_z)

    transport_xy_score = basket_xy_reward * grasped * (0.3 + 0.7 * high_enough)
    best_transport_xy = torch.maximum(best_transport_xy_prev, transport_xy_score)
    transport_progress = torch.clamp(best_transport_xy - best_transport_xy_prev, min=0.0, max=1.0)

    transport_reward = (
        0.55 * basket_xy_reward * grasped
        + 0.20 * basket_z_reward * grasped
        + 0.25 * transport_progress
    )

    # ------------------------------------------------------------------
    # Stage 4: Place into basket
    # Reward being centered over basket and descending below rim.
    # This is active even before release, to create a bridge to success.
    # ------------------------------------------------------------------
    temp_place_xy = 0.045
    place_xy_reward = torch.exp(-obj_site_xy_dist / temp_place_xy)

    depth_inside = torch.clamp(basket_top_z - obj_pos[:, 2], min=0.0)
    temp_depth = 0.04
    place_depth_reward = 1.0 - torch.exp(-depth_inside / temp_depth)

    place_score = place_xy_reward * (0.4 + 0.6 * place_depth_reward)
    best_place = torch.maximum(best_place_prev, place_score)
    place_progress = torch.clamp(best_place - best_place_prev, min=0.0, max=1.0)

    # Encourage opening only when object is well positioned over basket
    open_cmd = torch.clamp(actions[:, -1], -1.0, 1.0)
    open_amount = torch.clamp(open_cmd, min=0.0)
    release_after_transport_reward = open_amount * place_xy_reward * (0.2 + 0.8 * place_depth_reward)

    # Placement shaping while still grasped or just released near correct pose
    place_reward = (
        0.45 * place_xy_reward
        + 0.35 * place_depth_reward
        + 0.20 * place_progress
    )

    # ------------------------------------------------------------------
    # Sparse event bonuses
    # ------------------------------------------------------------------
    ever_grasped = torch.maximum(ever_grasped_prev, grasped)
    ever_lifted = torch.maximum(ever_lifted_prev, grasped * high_enough)

    first_grasp_bonus = torch.clamp(grasped - ever_grasped_prev, min=0.0, max=1.0)
    first_lift_bonus = torch.clamp(grasped * high_enough - ever_lifted_prev, min=0.0, max=1.0)
    newly_successful = torch.clamp(success - prev_success, min=0.0, max=1.0)

    # Stronger success signal based on demonstration statistics
    success_bonus = 3.5 * success + 2.5 * newly_successful

    # ------------------------------------------------------------------
    # Penalties
    # ------------------------------------------------------------------
    dropped_now = ((prev_grasped > 0.5) & (grasped < 0.5)).float()
    bad_drop = dropped_now * (1.0 - success) * (1.0 - place_xy_reward)
    early_drop_penalty = 0.8 * bad_drop

    premature_open_penalty = 0.12 * open_amount * (1.0 - place_xy_reward * (0.4 + 0.6 * high_enough))

    action_delta = actions - prev_actions
    action_smooth_penalty = 0.0015 * torch.sum(action_delta * action_delta, dim=-1)

    joint_vel_penalty = 0.0008 * torch.sum(joint_vel * joint_vel, dim=-1)

    dof_range = torch.clamp(self.robot_dof_upper_limits - self.robot_dof_lower_limits, min=eps)
    dist_to_lower = (joint_pos - self.robot_dof_lower_limits) / dof_range
    dist_to_upper = (self.robot_dof_upper_limits - joint_pos) / dof_range
    min_limit_dist = torch.minimum(dist_to_lower, dist_to_upper)
    limit_margin = 0.12
    joint_limit_frac = torch.clamp((limit_margin - min_limit_dist) / limit_margin, min=0.0, max=1.0)
    joint_limit_penalty = 0.02 * torch.sum(joint_limit_frac * joint_limit_frac, dim=-1)

    object_motion_penalty = 0.003 * obj_speed

    # Keep as diagnostic only; do not penalize exploration with it
    stage_regression_penalty = 0.05 * torch.clamp(prev_stage_idx - stage_idx, min=0.0, max=4.0)

    # ------------------------------------------------------------------
    # Total reward
    # Per-step reward kept in [-5, 5].
    # ------------------------------------------------------------------
    reward = (
        0.45 * pregrasp_score
        + 0.35 * pregrasp_progress
        + 0.70 * lift_stage_reward
        + 0.90 * transport_reward
        + 0.85 * place_reward
        + 0.30 * release_after_transport_reward
        + 0.80 * first_grasp_bonus
        + 1.00 * first_lift_bonus
        + success_bonus
        - early_drop_penalty
        - premature_open_penalty
        - action_smooth_penalty
        - joint_vel_penalty
        - joint_limit_penalty
        - object_motion_penalty
    )

    # ------------------------------------------------------------------
    # Update helper memory
    # ------------------------------------------------------------------
    self.helper_variable[:, 0] = stage_idx
    self.helper_variable[:, 1] = ever_grasped
    self.helper_variable[:, 2] = ever_lifted
    self.helper_variable[:, 3] = success
    self.helper_variable[:, 4] = grasped
    self.helper_variable[:, 5] = best_obj_height
    self.helper_variable[:, 6] = best_transport_xy
    self.helper_variable[:, 7] = best_pregrasp
    self.helper_variable[:, 8] = best_place

    reward = torch.nan_to_num(reward, nan=0.0, posinf=5.0, neginf=-5.0)
    reward = torch.clamp(reward, -5.0, 5.0)

    assert torch.isfinite(reward).all(), "Non-finite reward detected in _get_rewards_eureka."

    individual_rewards_dict = {
        "pregrasp_reward": torch.nan_to_num(0.45 * pregrasp_score + 0.35 * pregrasp_progress, nan=0.0, posinf=0.0, neginf=0.0),
        "lift_stage_reward": torch.nan_to_num(0.70 * lift_stage_reward + 0.80 * first_grasp_bonus + 1.00 * first_lift_bonus, nan=0.0, posinf=0.0, neginf=0.0),
        "transport_reward": torch.nan_to_num(0.90 * transport_reward, nan=0.0, posinf=0.0, neginf=0.0),
        "release_after_transport_reward": torch.nan_to_num(0.30 * release_after_transport_reward + 0.85 * place_reward, nan=0.0, posinf=0.0, neginf=0.0),
        "success_bonus": torch.nan_to_num(success_bonus, nan=0.0, posinf=0.0, neginf=0.0),
        "early_drop_penalty": torch.nan_to_num(-early_drop_penalty, nan=0.0, posinf=0.0, neginf=0.0),
        "action_smooth_penalty": torch.nan_to_num(-action_smooth_penalty, nan=0.0, posinf=0.0, neginf=0.0),
        "joint_vel_penalty": torch.nan_to_num(-joint_vel_penalty, nan=0.0, posinf=0.0, neginf=0.0),
        "joint_limit_penalty": torch.nan_to_num(-joint_limit_penalty, nan=0.0, posinf=0.0, neginf=0.0),
        "object_motion_penalty": torch.nan_to_num(-object_motion_penalty, nan=0.0, posinf=0.0, neginf=0.0),
        "stage_regression_penalty": torch.nan_to_num(-stage_regression_penalty, nan=0.0, posinf=0.0, neginf=0.0),
        "success_metric": torch.nan_to_num(success, nan=0.0, posinf=0.0, neginf=0.0),
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


    