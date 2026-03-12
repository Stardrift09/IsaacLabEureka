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

    eps = 1e-6
    device = self.device

    # ------------------------------------------------------------------
    # Sanitize environment tensors
    # ------------------------------------------------------------------
    robot_grasp_pos = torch.nan_to_num(self.robot_grasp_pos)
    target_pos = torch.nan_to_num(self.target_object.data.root_pos_w)
    target_vel = torch.nan_to_num(self.target_object.data.root_lin_vel_w)
    target_quat = torch.nan_to_num(self.target_object.data.root_quat_w)
    default_root_state = torch.nan_to_num(self.target_object.data.default_root_state)
    desired_quat = torch.nan_to_num(default_root_state[:, 3:7])

    target_to_hand_pos = torch.nan_to_num(self.target_to_hand_pos)
    manipulability = torch.nan_to_num(self.manipulability).clamp(min=0.0)
    grasped = torch.nan_to_num(self.grasped.float().squeeze(-1)).clamp(0.0, 1.0)

    joint_pos = torch.nan_to_num(self._robot.data.joint_pos)
    joint_vel = torch.nan_to_num(self._robot.data.joint_vel)
    body_lin_vel = torch.nan_to_num(self._robot.data.body_link_lin_vel_w)
    left_finger_vel = body_lin_vel[:, self.left_finger_body_idx]
    right_finger_vel = body_lin_vel[:, self.right_finger_body_idx]
    tcp_vel = 0.5 * (left_finger_vel + right_finger_vel)

    corners_flat = torch.nan_to_num(self.corners_target_obj_to_hand_pos)
    corners = corners_flat.view(self.num_envs, -1, 3)
    corner_dists = torch.linalg.norm(corners, dim=-1)
    min_corner_dist = torch.nan_to_num(corner_dists.min(dim=-1).values)

    to_desired_rot = torch.nan_to_num(self.to_desired_rot)

    # ------------------------------------------------------------------
    # Derived geometry
    # self.target_to_hand_pos = object_center - tcp
    # ------------------------------------------------------------------
    dist_3d = torch.linalg.norm(target_to_hand_pos, dim=-1)
    dist_xy = torch.linalg.norm(target_to_hand_pos[:, :2], dim=-1)
    z_rel = robot_grasp_pos[:, 2] - target_pos[:, 2]  # tcp_z - obj_z

    pregrasp_height = 0.10
    pregrasp_target = target_pos + torch.tensor([0.0, 0.0, pregrasp_height], device=device).unsqueeze(0)
    pregrasp_dist = torch.linalg.norm(robot_grasp_pos - pregrasp_target, dim=-1)

    tcp_speed = torch.linalg.norm(tcp_vel, dim=-1)
    tcp_xy_speed = torch.linalg.norm(tcp_vel[:, :2], dim=-1)
    tcp_down_speed = (-tcp_vel[:, 2]).clamp(min=0.0)
    obj_speed = torch.linalg.norm(target_vel, dim=-1)
    joint_speed = torch.linalg.norm(joint_vel, dim=-1)

    finger_pos = joint_pos[:, -2:]
    gripper_opening = torch.nan_to_num(finger_pos.sum(dim=-1))

    # ------------------------------------------------------------------
    # Orientation error
    # ------------------------------------------------------------------
    rot_w = torch.clamp(to_desired_rot[:, 0], -1.0, 1.0)
    rot_w01 = 0.5 * (rot_w + 1.0)

    target_quat = target_quat / torch.linalg.norm(target_quat, dim=-1, keepdim=True).clamp_min(eps)
    desired_quat = desired_quat / torch.linalg.norm(desired_quat, dim=-1, keepdim=True).clamp_min(eps)
    target_quat_conj = target_quat.clone()
    target_quat_conj[:, 1:] = -target_quat_conj[:, 1:]

    w1, x1, y1, z1 = desired_quat[:, 0], desired_quat[:, 1], desired_quat[:, 2], desired_quat[:, 3]
    w2, x2, y2, z2 = target_quat_conj[:, 0], target_quat_conj[:, 1], target_quat_conj[:, 2], target_quat_conj[:, 3]
    q_err = torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )
    q_err = q_err / torch.linalg.norm(q_err, dim=-1, keepdim=True).clamp_min(eps)
    q_err = torch.where(q_err[:, 0:1] < 0, -q_err, q_err)
    angle_error = 2.0 * torch.acos(torch.clamp(q_err[:, 0], -1.0, 1.0))
    angle_error = torch.nan_to_num(angle_error)

    # ------------------------------------------------------------------
    # Lift / success-aligned quantities
    # ------------------------------------------------------------------
    object_default_height = default_root_state[:, self.input_direction]
    lift_height = (target_pos[:, 2] - object_default_height).clamp(min=0.0)
    high_enough = (target_pos[:, 2] > object_default_height + 0.10).float()
    small_rotation = (angle_error < 0.8).float()
    success = grasped * high_enough * small_rotation

    # ------------------------------------------------------------------
    # Training-feedback-informed redesign:
    # 1) Existing policy exploits large easy rewards: orientation, xy_align,
    #    hover_height, descend_slow, open_approach.
    # 2) Contact and closing rewards are ~0, so the policy never learns grasp.
    # 3) Several penalties are almost constant and do not guide behavior.
    # 4) Demonstrations show successful trajectories still have noticeable
    #    object motion / close timing penalties, so avoid over-penalizing them.
    #
    # New strategy:
    # - Reduce all easy pre-contact rewards.
    # - Add progress reward from hover -> descend -> near-contact.
    # - Reward "close only when centered and low".
    # - Strongly reward grasp and post-grasp lift.
    # - Penalize object motion mainly when far from a valid grasp region.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Temperatures
    # ------------------------------------------------------------------
    temp_pregrasp = 0.10
    temp_xy = 0.05
    temp_hover_z = 0.05
    temp_orient = 0.30
    temp_center = 0.025
    temp_contact_z = 0.020
    temp_corner = 0.020
    temp_open = 0.030
    temp_close = 0.025
    temp_slow_tcp = 0.10
    temp_slow_obj = 0.10
    temp_slow_joint = 2.0
    temp_lift = 0.06

    # ------------------------------------------------------------------
    # Core smooth features
    # ------------------------------------------------------------------
    pregrasp_raw = torch.exp(-pregrasp_dist / temp_pregrasp)
    xy_raw = torch.exp(-dist_xy / temp_xy)
    hover_z_raw = torch.exp(-torch.abs(z_rel - pregrasp_height) / temp_hover_z)
    orient_raw = 0.35 * rot_w01 + 0.65 * torch.exp(-angle_error / temp_orient)

    center_raw = torch.exp(-dist_3d / temp_center)
    low_z_raw = torch.exp(-torch.abs(z_rel - 0.015) / temp_contact_z)
    corner_raw = torch.exp(-min_corner_dist / temp_corner)

    open_ref = 0.075
    close_ref = 0.010
    open_raw = torch.exp(-torch.abs(gripper_opening - open_ref) / temp_open)
    close_raw = torch.exp(-torch.abs(gripper_opening - close_ref) / temp_close)

    slow_tcp_raw = torch.exp(-tcp_speed / temp_slow_tcp)
    slow_obj_raw = torch.exp(-obj_speed / temp_slow_obj)
    slow_joint_raw = torch.exp(-joint_speed / temp_slow_joint)

    lift_progress_raw = (lift_height / 0.12).clamp(0.0, 1.0)
    lift_target_raw = torch.exp(-torch.abs(lift_height - 0.12) / temp_lift)

    # ------------------------------------------------------------------
    # Stage gates
    # ------------------------------------------------------------------
    align_gate = xy_raw * orient_raw
    hover_gate = align_gate * hover_z_raw
    near_contact_gate = xy_raw * orient_raw * low_z_raw
    very_near_contact = ((dist_xy < 0.025) & (torch.abs(z_rel - 0.015) < 0.020)).float()

    # Progressive descend reward:
    # encourage being lower only when xy/orientation are already correct.
    descend_progress = torch.clamp((pregrasp_height - z_rel) / pregrasp_height, 0.0, 1.0)
    desired_down_speed = 0.025
    descend_speed_raw = torch.exp(-torch.abs(tcp_down_speed - desired_down_speed) / 0.025)

    # Close readiness should not require exact contact, otherwise sparse.
    close_region_raw = 0.6 * center_raw + 0.4 * low_z_raw

    # ------------------------------------------------------------------
    # Positive terms: reduced easy rewards, increased grasp-linked rewards
    # ------------------------------------------------------------------
    r_pregrasp = 0.08 * pregrasp_raw * (1.0 - grasped)
    r_xy_align = 0.12 * xy_raw * (1.0 - grasped)
    r_hover_height = 0.08 * hover_z_raw * (1.0 - grasped)
    r_orientation = 0.18 * orient_raw * (1.0 - 0.3 * grasped)
    r_manipulability = 0.04 * torch.tanh(manipulability)

    r_open_approach = 0.08 * open_raw * (1.0 - very_near_contact) * (1.0 - grasped)
    r_hover_stable = 0.16 * hover_gate * slow_tcp_raw * open_raw * (1.0 - grasped)
    r_descend_progress = 0.55 * align_gate * descend_progress * open_raw * (1.0 - grasped)
    r_descend_slow = 0.30 * align_gate * descend_speed_raw * open_raw * (1.0 - grasped)

    # Main bridge to grasp: easier to optimize than previous contact terms
    r_near_contact = 1.40 * near_contact_gate * open_raw * (1.0 - grasped)
    r_centering = 1.10 * close_region_raw * xy_raw * orient_raw * (1.0 - grasped)
    r_corner_clearance = 0.35 * corner_raw * near_contact_gate * (1.0 - grasped)

    # Explicit close timing
    r_close_ready = 2.20 * close_raw * close_region_raw * xy_raw * orient_raw * (1.0 - grasped)
    r_close_bonus = 1.20 * very_near_contact * close_raw * (1.0 - grasped)

    # Make actual grasp decisively better than hovering
    r_grasp = 12.0 * grasped
    r_hold_still = 0.80 * grasped * close_raw * slow_obj_raw
    r_lift_progress = 8.0 * grasped * lift_progress_raw
    r_lift_target = 8.0 * grasped * lift_target_raw * orient_raw
    r_gentle_lift = 0.80 * grasped * slow_obj_raw * slow_tcp_raw

    r_slow_joint = 0.03 * slow_joint_raw
    r_success_bonus = 30.0 * success

    # ------------------------------------------------------------------
    # Penalties: lighter and more targeted than before
    # ------------------------------------------------------------------
    # Penalize object motion most when not yet in valid close region
    p_push_obj = -0.60 * obj_speed * (1.0 - near_contact_gate) * (1.0 - grasped)
    p_lateral = -0.18 * tcp_xy_speed * torch.exp(-dist_3d / 0.06) * (1.0 - grasped)
    p_premature_descend = -0.15 * tcp_down_speed * (1.0 - align_gate) * (1.0 - grasped)

    # Penalize closing while still clearly not ready, but less aggressively
    not_ready_close_gate = (1.0 - 0.7 * close_region_raw) * (1.0 - 0.5 * xy_raw)
    not_ready_close_gate = torch.clamp(not_ready_close_gate, 0.0, 1.0)
    p_early_close = -0.25 * close_raw * not_ready_close_gate * (1.0 - grasped)

    # At valid close region, being open forever is bad
    p_stay_open_at_contact = -0.45 * very_near_contact * open_raw * (1.0 - grasped)

    # Closing at bad pose only matters near the object
    p_bad_close_pose = -0.20 * close_raw * torch.exp(-dist_3d / 0.05) * (1.0 - xy_raw * low_z_raw) * (1.0 - grasped)

    p_object_motion = -0.12 * obj_speed * torch.exp(-dist_3d / 0.05) * (1.0 - grasped)
    p_under_object = -0.15 * (-z_rel).clamp(min=0.0) * (1.0 - grasped)
    p_fast_near = -0.10 * tcp_speed * near_contact_gate * (1.0 - grasped)
    p_drop = -1.50 * grasped * torch.clamp(-target_vel[:, 2], min=0.0)

    individual_rewards_dict = {
        "pregrasp": r_pregrasp,
        "xy_align": r_xy_align,
        "hover_height": r_hover_height,
        "orientation": r_orientation,
        "manipulability": r_manipulability,
        "open_approach": r_open_approach,
        "hover_stable": r_hover_stable,
        "descend_progress": r_descend_progress,
        "descend_slow": r_descend_slow,
        "near_contact": r_near_contact,
        "centering": r_centering,
        "corner_clearance": r_corner_clearance,
        "close_ready": r_close_ready,
        "close_bonus": r_close_bonus,
        "grasp": r_grasp,
        "hold_still": r_hold_still,
        "lift_progress": r_lift_progress,
        "lift_target": r_lift_target,
        "gentle_lift": r_gentle_lift,
        "slow_joint": r_slow_joint,
        "push_obj_penalty": p_push_obj,
        "lateral_penalty": p_lateral,
        "premature_descend_penalty": p_premature_descend,
        "early_close_penalty": p_early_close,
        "stay_open_at_contact_penalty": p_stay_open_at_contact,
        "bad_close_pose_penalty": p_bad_close_pose,
        "object_motion_penalty": p_object_motion,
        "under_object_penalty": p_under_object,
        "fast_near_penalty": p_fast_near,
        "drop_penalty": p_drop,
        "success_bonus": r_success_bonus,
    }

    reward = torch.zeros(self.num_envs, device=device)
    for value in individual_rewards_dict.values():
        reward = reward + torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

    reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
    assert torch.isfinite(reward).all(), "Non-finite reward detected in _get_rewards_eureka"

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


    