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
    # VERY IMPORTANT!!!!!!!!!! This reward is able to lift the object already, but the object rotates more or less, improve the new reward function based on this!
    import torch

    device = self.device
    n = self.num_envs
    eps = 1e-6

    def _safe(x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # -------------------------------------------------------------------------
    # Core state (sanitized)
    # -------------------------------------------------------------------------
    rel_pos = _safe(self.target_to_hand_pos)  # object_center - tcp, [n,3]
    dx, dy, dz = rel_pos[:, 0], rel_pos[:, 1], rel_pos[:, 2]
    xy_dist = torch.sqrt(dx * dx + dy * dy + eps)
    dist = torch.sqrt(xy_dist * xy_dist + dz * dz + eps)

    # "10cm above object center" waypoint: tcp_z = obj_z + 0.10 -> dz = -0.10
    dz_above = dz + 0.10
    dist_above = torch.sqrt(dx * dx + dy * dy + dz_above * dz_above + eps)

    # Orientation: reward delta-quat w -> 1
    wq = _safe(self.to_desired_rot[:, 0]).clamp(-1.0, 1.0)
    ori_align = ((wq + 1.0) * 0.5).clamp(0.0, 1.0)
    ori_err = (1.0 - ori_align).clamp(0.0, 1.0)

    # Velocities (tcp approximated by finger avg)
    tcp_vel = _safe(
        0.5
        * (
            self._robot.data.body_link_lin_vel_w[:, self.left_finger_body_idx]
            + self._robot.data.body_link_lin_vel_w[:, self.right_finger_body_idx]
        )
    )
    tcp_speed = torch.linalg.norm(tcp_vel, dim=-1)
    tcp_vxy = torch.linalg.norm(tcp_vel[:, :2], dim=-1)
    tcp_vz = tcp_vel[:, 2]

    obj_vel = _safe(self.target_object.data.root_lin_vel_w)
    obj_speed = torch.linalg.norm(obj_vel, dim=-1)

    rel_vel = _safe(obj_vel - tcp_vel)
    rel_speed = torch.linalg.norm(rel_vel, dim=-1)

    # Gripper openness proxy from finger joints (last 2 dofs)
    joint_pos = _safe(self._robot.data.joint_pos)
    if joint_pos.shape[-1] >= 2:
        finger_sum = joint_pos[:, -2:].sum(dim=-1)
    else:
        finger_sum = torch.zeros((n,), device=device)

    # Success signals
    grasped = _safe(self.grasped.float()).view(-1)

    object_default_state = _safe(self.target_object.data.default_root_state)
    obj_z_default = _safe(object_default_state[:, self.input_direction])
    obj_z = _safe(self.target_object.data.root_pos_w[:, 2])
    lift_height = torch.clamp(obj_z - (obj_z_default + 0.05), 0.0, 0.35)

    manipulability = torch.clamp(_safe(self.manipulability).view(-1), 0.0, 10.0)

    # -------------------------------------------------------------------------
    # Soft stage gates (avoid "always-on" terms that caused reward hacking)
    # -------------------------------------------------------------------------
    k = 60.0
    gate_near = torch.sigmoid(-k * (dist - 0.07))         # within ~7cm
    gate_very_near = torch.sigmoid(-k * (dist - 0.04))    # within ~4cm
    gate_xy_good = torch.sigmoid(-k * (xy_dist - 0.020))  # within ~2cm
    gate_xy_tight = torch.sigmoid(-k * (xy_dist - 0.012)) # within ~1.2cm

    # tcp above object center => dz = obj - tcp < 0
    gate_tcp_above = torch.sigmoid(-35.0 * (dz + 0.003))  # dz < -3mm

    # Prefer approaching from above waypoint region (10cm above)
    gate_above_wp = torch.sigmoid(-k * (dist_above - 0.05))  # within ~5cm of above waypoint

    gate_ori_good = torch.sigmoid(40.0 * (0.06 - ori_err))  # ori_err < 0.06

    pre = (1.0 - grasped) * (1.0 - gate_near)
    approach = (1.0 - grasped) * gate_near
    pocket = (1.0 - grasped) * gate_very_near * gate_xy_tight * gate_ori_good * gate_tcp_above
    post = grasped

    # -------------------------------------------------------------------------
    # Bounded shaping rewards (temperatures)
    # -------------------------------------------------------------------------
    t_dist = 0.06
    t_xy = 0.03
    t_above = 0.05
    t_ori = 0.12
    t_speed = 0.30
    t_rel = 0.22
    t_lift = 0.08

    r_close = torch.exp(-dist / t_dist).clamp(0.0, 1.0)
    r_xy = torch.exp(-xy_dist / t_xy).clamp(0.0, 1.0)
    r_above = torch.exp(-dist_above / t_above).clamp(0.0, 1.0)
    r_ori = torch.exp(-ori_err / t_ori).clamp(0.0, 1.0)

    r_slow_tcp = torch.exp(-tcp_speed / t_speed).clamp(0.0, 1.0)
    r_slow_obj = torch.exp(-obj_speed / t_speed).clamp(0.0, 1.0)
    r_slow = (0.8 * r_slow_tcp + 0.2 * r_slow_obj).clamp(0.0, 1.0)

    r_gentle = torch.exp(-rel_speed / t_rel).clamp(0.0, 1.0)

    r_lift = (1.0 - torch.exp(-lift_height / t_lift)).clamp(0.0, 1.0)

    r_manip = torch.tanh(manipulability / 2.0).clamp(0.0, 1.0)

    # Gripper open/close (kept but heavily gated to avoid constant reward)
    r_open = torch.sigmoid(14.0 * (finger_sum - 0.02)).clamp(0.0, 1.0)
    r_closed = torch.sigmoid(14.0 * (0.03 - finger_sum)).clamp(0.0, 1.0)

    # Pocket alignment metric (sharper than before but still smooth)
    # Encourage: very small xy_dist and dz close to 0 (tcp at object center height), while still approaching from above.
    t_pocket = 0.55
    pocket_metric = torch.sqrt((xy_dist / 0.012) ** 2 + (torch.abs(dz) / 0.025) ** 2 + eps)
    r_pocket = torch.exp(-pocket_metric / t_pocket).clamp(0.0, 1.0)

    # Descent control: small downward speed only when aligned/above/near
    # Track a gentle downward speed band.
    t_vz = 0.05
    vz_des = -0.02
    r_vz = torch.exp(-torch.abs(tcp_vz - vz_des) / t_vz).clamp(0.0, 1.0)
    r_vxy = torch.exp(-tcp_vxy / 0.12).clamp(0.0, 1.0)
    r_desc_ctrl = (0.6 * r_vz + 0.4 * r_vxy).clamp(0.0, 1.0)

    # -------------------------------------------------------------------------
    # Key fix vs bad runs:
    # - In bad training, the policy got huge reward from always-on dense terms
    #   and tolerated huge penalties; success stayed ~0.
    # - Make dense terms small; make "pocket->close->grasp->lift" decisive.
    # - Penalties must be mild and only near; otherwise agent avoids the object.
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Mild penalties (bounded; near-only)
    # -------------------------------------------------------------------------
    pen_push = gate_very_near * (1.0 - grasped) * torch.sigmoid(30.0 * (rel_speed - 0.10))
    pen_sweep = gate_very_near * (1.0 - grasped) * torch.sigmoid(30.0 * (tcp_vxy - 0.12))
    pen_fast_down = gate_very_near * (1.0 - grasped) * torch.sigmoid(40.0 * ((-tcp_vz) - 0.15))
    pen_close_early = gate_near * (1.0 - grasped) * r_closed * (1.0 - pocket)

    # -------------------------------------------------------------------------
    # Weights (scaled so typical per-episode totals don't explode)
    # -------------------------------------------------------------------------
    w_above = 0.6
    w_xy = 0.8
    w_close = 0.8
    w_ori = 0.25

    # Keep gripper open only while not too near; otherwise don't shape much
    w_open = 0.10

    # Critical funnel to success:
    w_pocket = 3.0
    w_desc = 0.8
    w_close_cmd = 6.0

    # Sparse objectives:
    w_grasp = 25.0
    w_lift = 15.0

    # Regularizer:
    w_slow = 0.25
    w_gentle = 0.30
    w_manip = 0.15

    # Penalties (mild):
    w_pen_push = 0.6
    w_pen_sweep = 0.3
    w_pen_fast_down = 0.3
    w_pen_close_early = 0.4

    # -------------------------------------------------------------------------
    # Weighted reward terms
    # -------------------------------------------------------------------------
    # Waypoint: go above first (pre only)
    term_above = w_above * pre * r_above

    # General approach (small, cannot dominate)
    approach_from_above = (0.35 + 0.65 * gate_tcp_above).clamp(0.0, 1.0)
    term_xy = w_xy * (pre + approach) * approach_from_above * r_xy
    term_close = w_close * (pre + approach) * approach_from_above * r_close

    # Orientation only matters meaningfully when approaching/near
    term_ori = w_ori * (pre + approach + 0.5 * pocket + 0.2 * post) * r_ori

    # Open gripper until very near
    term_open = w_open * (pre + approach) * (1.0 - gate_very_near) * r_open

    # Pocket alignment + controlled descent (near + above + oriented)
    term_pocket = w_pocket * (approach + pocket) * gate_xy_good * gate_ori_good * r_pocket
    term_desc = w_desc * (approach + pocket) * gate_above_wp * gate_xy_good * gate_ori_good * gate_tcp_above * r_desc_ctrl

    # Close command: only in pocket and when slow/gentle to avoid pushing
    term_close_cmd = w_close_cmd * pocket * r_closed * (0.6 * r_slow + 0.4 * r_gentle)

    # Mild global motion quality (prevents jerky exploration, but low weight)
    term_slow = w_slow * (0.4 * pre + 0.7 * approach + 0.9 * pocket + 1.2 * post) * r_slow
    term_gentle = w_gentle * (0.4 * pre + 0.8 * approach + 1.0 * pocket + 1.0 * post) * r_gentle
    term_manip = w_manip * r_manip

    # Sparse task outcomes
    term_grasp = w_grasp * grasped
    term_lift = w_lift * post * r_lift

    # Penalties
    term_pen_push = -w_pen_push * pen_push
    term_pen_sweep = -w_pen_sweep * pen_sweep
    term_pen_fast_down = -w_pen_fast_down * pen_fast_down
    term_pen_close_early = -w_pen_close_early * pen_close_early

    reward = (
        term_above
        + term_xy
        + term_close
        + term_ori
        + term_open
        + term_pocket
        + term_desc
        + term_close_cmd
        + term_slow
        + term_gentle
        + term_manip
        + term_grasp
        + term_lift
        + term_pen_push
        + term_pen_sweep
        + term_pen_fast_down
        + term_pen_close_early
    )

    reward = _safe(reward).view(-1)
    assert reward.shape == (n,)
    assert torch.isfinite(reward).all()

    individual = {
        "above_pos": _safe(term_above),
        "xy_align": _safe(term_xy),
        "close_pos": _safe(term_close),
        "orientation": _safe(term_ori),
        "open_gripper": _safe(term_open),
        "pocket": _safe(term_pocket),
        "descend_ctrl": _safe(term_desc),
        "close_gripper_cmd": _safe(term_close_cmd),
        "slow": _safe(term_slow),
        "gentle": _safe(term_gentle),
        "manipulability": _safe(term_manip),
        "grasp": _safe(term_grasp),
        "lift": _safe(term_lift),
        "push_penalty": _safe(term_pen_push),
        "sweep_penalty": _safe(term_pen_sweep),
        "fast_down_penalty": _safe(term_pen_fast_down),
        "close_early_penalty": _safe(term_pen_close_early),
    }
    return reward, individual
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


    