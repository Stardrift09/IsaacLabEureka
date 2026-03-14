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

    device = self.device
    eps: float = 1e-6

    # ----------------------------
    # Fetch + sanitize key signals
    # ----------------------------
    obj_pos_w = torch.nan_to_num(self.target_object.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
    site_pos_w = torch.nan_to_num(self.target_site.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
    hand_pos_w = torch.nan_to_num(self.robot_grasp_pos, nan=0.0, posinf=0.0, neginf=0.0)

    obj_xy = obj_pos_w[:, :2]
    site_xy = site_pos_w[:, :2]
    hand_xy = hand_pos_w[:, :2]

    obj_z = obj_pos_w[:, 2]
    hand_z = hand_pos_w[:, 2]

    # Relative vectors (already computed in obs path, but re-sanitize for safety)
    t2h = torch.nan_to_num(self.target_to_hand_pos, nan=0.0, posinf=0.0, neginf=0.0)
    s2t = torch.nan_to_num(self.site_to_target_pos, nan=0.0, posinf=0.0, neginf=0.0)

    # Distances
    d_hand = torch.linalg.norm(t2h, dim=-1)
    d_hand_xy = torch.linalg.norm(t2h[:, :2], dim=-1)
    d_site_xy = torch.linalg.norm(s2t[:, :2], dim=-1)

    # Orientation "pointing down" proxy using provided self.quat (quat = z_quat * q_hand_inv)
    # Desired: identity -> quat close to [0,0,0,1]
    quat_err = torch.nan_to_num(self.quat, nan=0.0, posinf=0.0, neginf=0.0)
    # Scalar part close to 1 means close to desired
    quat_w = torch.clamp(quat_err[:, 3], -1.0, 1.0)
    quat_v_norm = torch.linalg.norm(quat_err[:, :3], dim=-1)

    # ----------------------------
    # Success metric (as provided)
    # ----------------------------
    low_enough = obj_z < 0.1
    dist2 = ((obj_xy - site_xy) ** 2).sum(dim=-1)
    inside_site = dist2 < (float(self.target_site_radius) ** 2)

    # ----------------------------
    # Shaping components
    # ----------------------------
    # 1) Approach object with hand (coarse + fine)
    temp_hand_coarse: float = 0.25
    temp_hand_fine: float = 0.07
    r_hand_coarse = torch.exp(-torch.clamp(d_hand / temp_hand_coarse, 0.0, 10.0))
    r_hand_fine = torch.exp(-torch.clamp(d_hand / temp_hand_fine, 0.0, 10.0))

    # 2) Encourage approaching from above (reduces pushing): hand above object and small XY offset
    #    Smooth gating: prefer hand_z >= obj_z + margin, while still allowing recovery.
    above_margin: float = 0.03
    temp_above: float = 0.03
    above_score = torch.sigmoid(torch.clamp((hand_z - (obj_z + above_margin)) / (temp_above + eps), -10.0, 10.0))

    temp_xy_align: float = 0.06
    r_xy_align = torch.exp(-torch.clamp(d_hand_xy / temp_xy_align, 0.0, 10.0))

    r_approach_from_above = above_score * r_xy_align

    # 3) Orientation alignment (point down) via quat closeness to identity
    temp_quat_w: float = 0.15
    temp_quat_v: float = 0.30
    r_quat_w = torch.exp(-torch.clamp((1.0 - quat_w) / (temp_quat_w + eps), 0.0, 10.0))
    r_quat_v = torch.exp(-torch.clamp(quat_v_norm / (temp_quat_v + eps), 0.0, 10.0))
    r_orientation = 0.5 * (r_quat_w + r_quat_v)

    # 4) "Don't push away" proxy: reward keeping object close to its initial basket line? (not available)
    #    Instead: reward minimizing object XY speed if available, else use gentle penalty for being far from site
    #    (this discourages pushing it away from the basket area during manipulation).
    if hasattr(self.target_object.data, "root_lin_vel_w"):
        obj_lin_vel_w = torch.nan_to_num(self.target_object.data.root_lin_vel_w, nan=0.0, posinf=0.0, neginf=0.0)
        obj_speed_xy = torch.linalg.norm(obj_lin_vel_w[:, :2], dim=-1)
        temp_speed_xy: float = 0.6
        r_low_obj_xy_speed = torch.exp(-torch.clamp(obj_speed_xy / (temp_speed_xy + eps), 0.0, 10.0))
    else:
        r_low_obj_xy_speed = torch.ones((self.num_envs,), device=device)

    # 5) Bring object over basket in XY (for dropping)
    temp_site_xy: float = 0.10
    r_site_xy = torch.exp(-torch.clamp(d_site_xy / temp_site_xy, 0.0, 10.0))

    # 6) Progression: prefer first grasp/close, then move towards site, then drop low inside.
    #    Use smooth gates based on hand-object distance.
    grasp_close_thresh: float = 0.06
    temp_grasp_gate: float = 0.02
    grasp_gate = torch.sigmoid(torch.clamp((grasp_close_thresh - d_hand) / (temp_grasp_gate + eps), -10.0, 10.0))

    # While not yet "grasped": focus on approach + orientation + no-push
    r_pregrasp = 0.55 * r_hand_coarse + 0.25 * r_approach_from_above + 0.20 * r_orientation
    r_pregrasp = r_pregrasp * (0.7 + 0.3 * r_low_obj_xy_speed)

    # After close: focus on moving object to site XY, still keep hand near to maintain control
    r_transport = 0.55 * r_site_xy + 0.25 * r_hand_fine + 0.20 * r_low_obj_xy_speed

    # Combine via gate
    r_shaped = (1.0 - grasp_gate) * r_pregrasp + grasp_gate * r_transport

    # 7) Terminal-like bonus when success condition met (dense but strong)
    #    Success is: inside site AND low enough (<0.1)
    success = inside_site & low_enough
    r_success = success.float()

    # Additional "near success": inside site and getting low (smooth)
    temp_low: float = 0.05
    low_score = torch.exp(-torch.clamp(torch.relu(obj_z - 0.1) / (temp_low + eps), 0.0, 10.0))
    r_near_success = r_site_xy * low_score

    # ----------------------------
    # Weights + total reward
    # ----------------------------
    w_shaped: float = 1.0
    w_near_success: float = 1.0
    w_success: float = 6.0

    reward = w_shaped * r_shaped + w_near_success * r_near_success + w_success * r_success

    # Mild regularization: penalize extreme joint velocities if available
    if hasattr(self, "_robot") and hasattr(self._robot, "data") and hasattr(self._robot.data, "joint_vel"):
        jvel = torch.nan_to_num(self._robot.data.joint_vel, nan=0.0, posinf=0.0, neginf=0.0)
        # keep small to avoid destabilizing exploration
        vel_pen = torch.mean(torch.clamp(jvel * jvel, 0.0, 100.0), dim=-1)
        reward = reward - 0.01 * vel_pen
    else:
        vel_pen = torch.zeros((self.num_envs,), device=device)

    # Final safety clamps
    reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
    reward = torch.clamp(reward, -10.0, 10.0)

    # Assert finite for training stability
    if not torch.isfinite(reward).all():
        reward = torch.where(torch.isfinite(reward), reward, torch.zeros_like(reward))

    # Assert finite for training stability
    if not torch.isfinite(reward).all():
        reward = torch.where(torch.isfinite(reward), reward, torch.zeros_like(reward))

    # reward = r_orientation
    # individual_rewards = {
    #     # "hand_coarse": r_hand_coarse,
    #     # "hand_fine": r_hand_fine,
    #     # "approach_from_above": r_approach_from_above,
    #     "orientation": r_orientation,
    #     # "low_obj_xy_speed": r_low_obj_xy_speed,
    #     # "site_xy": r_site_xy,
    #     # "grasp_gate": grasp_gate,
    #     # "shaped": r_shaped,
    #     # "near_success": r_near_success,
    #     # "success": r_success,
    #     # "vel_pen": vel_pen,
    # }

    # hand_quat = self._robot.data.body_quat_w[:, self.hand_link_idx]
    # q_hand_inv = quat_inv(hand_quat)
    # quat_1=torch.tensor([0, 0, 0, 1],device = self.device)
    # z_quat = quat_1.repeat(self.num_envs,1)
    # quat = quat_mul(z_quat,q_hand_inv)
    # reward = hand_quat[:,1]

    # z_local = torch.tensor([0, 0, 1], device=self.device,dtype=torch.float)
    # z_local = z_local.repeat(self.num_envs, 1)
    # z_world = quat_apply(hand_quat, z_local)
    # reward = -z_world[:,-1]

    individual_rewards = {
        "quat": reward
        "reach": torch.nan_to_num(r_reach),
        "upright": torch.nan_to_num(r_upright),
        "lift": torch.nan_to_num(r_lift),
        "site_xy": torch.nan_to_num(r_site_xy),
        "place_shaped": torch.nan_to_num(r_place_shaped),
        "cautious": torch.nan_to_num(r_cautious),
        "stage_mix": torch.nan_to_num(r_staged),
        "success_bonus": torch.nan_to_num(r_success),
        "tilt_pen": torch.nan_to_num(r_tilt_pen),
        "success_metric": success,  # per-env success indicator
    }
    return reward, individual_rewards

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


    