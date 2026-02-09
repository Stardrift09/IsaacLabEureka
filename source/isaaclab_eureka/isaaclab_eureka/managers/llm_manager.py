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
        self._debug = False

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


test_reward_string = """def _get_rewards_eureka(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    import torch

    device = self.device
    eps: float = 1e-6

    # -------------------------
    # Fetch & sanitize signals
    # -------------------------
    obj_pos = torch.nan_to_num(self.target_object.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
    obj_quat = torch.nan_to_num(self.target_object.data.root_quat_w, nan=0.0, posinf=0.0, neginf=0.0)
    site_pos = torch.nan_to_num(self.target_site.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)

    hand_pos = torch.nan_to_num(self.robot_grasp_pos, nan=0.0, posinf=0.0, neginf=0.0)
    # hand_rot available in env: self.robot_grasp_rot (used in obs)
    hand_rot = torch.nan_to_num(self.robot_grasp_rot, nan=0.0, posinf=0.0, neginf=0.0)

    # Distances
    to_hand = obj_pos - hand_pos
    dist_hand = torch.linalg.norm(to_hand, dim=-1)

    obj_xy = obj_pos[:, :2]
    site_xy = site_pos[:, :2]
    dist_site_xy = torch.linalg.norm(obj_xy - site_xy, dim=-1)
    obj_z = obj_pos[:, 2]

    # Success metric components (match description)
    low_enough = obj_z < 0.1
    inside_site = dist_site_xy < float(self.target_site_radius)

    # -------------------------
    # Orientation: keep upright
    # -------------------------
    # For upright, reward alignment of object local z-axis with world z-axis.
    # Convert quat (w,x,y,z) to z-axis in world:
    # z_axis_world = R(q) * [0,0,1]
    # For quaternion q = (w, x, y, z):
    # z_axis = -[2(xz + wy), 2(yz - wx), 1 - 2(x^2 + y^2)] # my change for correct
    w = obj_quat[:, 0]
    x = obj_quat[:, 1]
    y = obj_quat[:, 2]
    z = obj_quat[:, 3]
    z_axis_world = torch.stack(
        (2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)),
        dim=-1,
    )
    z_axis_world = torch.nan_to_num(z_axis_world, nan=0.0, posinf=0.0, neginf=0.0)
    upright_cos = torch.clamp(z_axis_world[:, 2], -1.0, 1.0)  # dot with world z
    # map [-1,1] -> [0,1]
    upright_score = 0.5 * (upright_cos + 1.0)

    # -------------------------
    # Stage detection (simple)
    # -------------------------
    # Stage 0: reach object
    # Stage 1: lift and keep upright
    # Stage 2: move above basket/site while keeping upright
    # Stage 3: place into basket (inside_site & low_enough)
    reach_thresh: float = 0.06
    lift_z: float = 0.14  # above table-ish; encourages picking up before moving
    above_site_xy_thresh: float = float(self.target_site_radius) * 1.5 + 0.02

    reached = dist_hand < reach_thresh
    lifted = obj_z > lift_z
    above_site_xy = dist_site_xy < above_site_xy_thresh
    placed = inside_site & low_enough

    # Use floats for blending/gating
    reached_f = reached.float()
    lifted_f = lifted.float()
    above_site_xy_f = above_site_xy.float()
    placed_f = placed.float()

    # Stage weights (soft step-by-step curriculum)
    # - always some reach reward, but later stages dominate once conditions satisfied
    w_reach = 1.0 - reached_f
    w_lift = reached_f * (1.0 - lifted_f)
    w_move = reached_f * lifted_f * (1.0 - above_site_xy_f)
    w_place = reached_f * lifted_f * above_site_xy_f

    # Normalize weights to avoid dead zones
    w_sum = w_reach + w_lift + w_move + w_place + eps
    w_reach = w_reach / w_sum
    w_lift = w_lift / w_sum
    w_move = w_move / w_sum
    w_place = w_place / w_sum

    # -------------------------
    # Dense reward components
    # -------------------------
    # Temperatures (required for transformed components)
    temp_reach: float = 0.08
    temp_site_xy: float = 0.12
    temp_upright: float = 0.25
    temp_height: float = 0.06
    temp_cautious: float = 0.10

    # Reach: encourage small hand-object distance
    r_reach = torch.exp(-dist_hand / temp_reach)

    # Upright: strongly encourage uprightness during lift/move/place
    # Use exponential on (1 - upright_score) to heavily penalize tilt
    r_upright = torch.exp(-(1.0 - upright_score) / temp_upright)

    # Lift: encourage object height (only meaningful after reaching)
    # Target height is lift_z; saturate above it.
    height_err = torch.clamp(lift_z - obj_z, min=0.0)
    r_lift = torch.exp(-height_err / temp_height)

    # Move to site in XY (while lifted)
    r_site_xy = torch.exp(-dist_site_xy / temp_site_xy)

    # Place: bonus for being inside and low enough (shaped)
    # Provide smooth shaping even before boolean success.
    # Use a soft "inside" score based on radius.
    radius = float(self.target_site_radius)
    inside_soft = torch.exp(-torch.clamp(dist_site_xy - radius, min=0.0) / (radius + 0.02))
    # low_enough shaping: encourage z down to 0.1 once near site
    z_target: float = 0.1
    z_err_place = torch.clamp(obj_z - z_target, min=0.0)
    r_low = torch.exp(-z_err_place / 0.04)
    r_place_shaped = inside_soft * r_low

    # Cautiousness: discourage very fast object motion (proxy: hand-object relative speed not available;
    # use joint velocities as a weak regularizer if present).
    if hasattr(self, "_robot") and hasattr(self._robot, "data") and hasattr(self._robot.data, "joint_vel"):
        jv = torch.nan_to_num(self._robot.data.joint_vel, nan=0.0, posinf=0.0, neginf=0.0)
        speed = torch.linalg.norm(jv, dim=-1)
    else:
        speed = torch.zeros((self.num_envs,), device=device)
    r_cautious = torch.exp(-speed / temp_cautious)

    # -------------------------
    # Compose staged reward
    # -------------------------
    # Make later stages include upright + cautious consistently.
    stage_reach = r_reach * (0.3 + 0.7 * r_cautious)
    stage_lift = r_lift * r_upright * (0.4 + 0.6 * r_cautious)
    stage_move = r_site_xy * r_upright * (0.4 + 0.6 * r_cautious)
    stage_place = r_place_shaped * r_upright * (0.5 + 0.5 * r_cautious)

    # Weighted staged sum
    r_staged = (
        w_reach * stage_reach
        + w_lift * stage_lift
        + w_move * stage_move
        + w_place * stage_place
    )

    # Sparse success bonus (kept bounded)
    success = (inside_site & low_enough).float()
    r_success = 1.0 * success

    # Small penalty if object is very tilted (avoid catastrophic behavior)
    tilt_pen = torch.clamp(0.7 - upright_score, min=0.0)  # only when quite non-upright
    r_tilt_pen = -0.2 * tilt_pen

    # Final reward (bounded-ish)
    reward = 1.5 * r_staged + 1.0 * r_success + r_tilt_pen

    # Safety: finite
    reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
    assert torch.isfinite(reward).all()

    

    hand_quat = self._robot.data.body_quat_w[:, self.hand_link_idx]
    q_hand_inv = quat_inv(hand_quat)
    quat_1=torch.tensor([0, 0, 0, 1],device = self.device)
    z_quat = quat_1.repeat(self.num_envs,1)
    quat = quat_mul(z_quat,q_hand_inv)
    reward = quat[:,1]
    individual_rewards = {
        "quat": reward
        # "reach": torch.nan_to_num(r_reach),
        # "upright": torch.nan_to_num(r_upright),
        # "lift": torch.nan_to_num(r_lift),
        # "site_xy": torch.nan_to_num(r_site_xy),
        # "place_shaped": torch.nan_to_num(r_place_shaped),
        # "cautious": torch.nan_to_num(r_cautious),
        # "stage_mix": torch.nan_to_num(r_staged),
        # "success_bonus": torch.nan_to_num(r_success),
        # "tilt_pen": torch.nan_to_num(r_tilt_pen),
        # "success_metric": success,  # per-env success indicator
    }
    return reward, individual_rewards
                """