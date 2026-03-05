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
                    eps: float = 1e-6

                    def _safe(x: torch.Tensor) -> torch.Tensor:
                        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

                    # ------------------------------------------------------------
                    # Fetch & sanitize (all tensors on self.device)
                    # ------------------------------------------------------------
                    obj_pos = _safe(self.target_object.data.root_pos_w)  # (N,3)
                    obj_vel = _safe(self.target_object.data.root_lin_vel_w)  # (N,3)
                    site_pos = _safe(self.target_site.data.root_pos_w)  # (N,3)
                    site_vel = _safe(self.target_site.data.root_lin_vel_w)  # (N,3)

                    robot_q = _safe(self._robot.data.joint_pos)  # (N,dof)
                    robot_qd = _safe(self._robot.data.joint_vel)  # (N,dof)

                    target_to_hand = _safe(self.target_to_hand_pos)  # (N,3)
                    site_to_target = _safe(self.site_to_target_pos)  # (N,3)
                    hand_quat = _safe(self.hand_quat)  # (N,4)

                    grasped = _safe(self._grasp_detection().float())
                    if grasped.ndim == 2 and grasped.shape[-1] == 1:
                        grasped = grasped.squeeze(-1)
                    grasped = torch.clamp(grasped, 0.0, 1.0)

                    manipulability = torch.clamp(_safe(self.manipulability), 0.0, 1e3)

                    # TCP vel (slip proxy)
                    left_v = _safe(self._robot.data.body_link_lin_vel_w[:, self.left_finger_body_idx])
                    right_v = _safe(self._robot.data.body_link_lin_vel_w[:, self.right_finger_body_idx])
                    tcp_vel = 0.5 * (left_v + right_v)
                    rel_vel = torch.linalg.norm(obj_vel - tcp_vel, dim=-1)

                    # Optional action regularization (if available)
                    act = getattr(self, "actions", None)
                    if act is None:
                        act_pen = torch.zeros((self.num_envs,), device=device)
                    else:
                        act = _safe(act)
                        act_pen = torch.sum(torch.clamp(act, -1.0, 1.0) ** 2, dim=-1)

                    # ------------------------------------------------------------
                    # Geometry / success metric
                    # ------------------------------------------------------------
                    rad: float = float(self.target_site_radius)
                    d_xy = torch.linalg.norm(site_to_target[:, :2], dim=-1)
                    obj_z = obj_pos[:, 2]
                    site_z = site_pos[:, 2]
                    h = obj_z - site_z  # height above basket bottom plane

                    inside_site = (d_xy * d_xy) < (rad * rad)
                    low_enough = obj_z < 0.1
                    success = (inside_site & low_enough).float()

                    hand_obj_dist = torch.linalg.norm(target_to_hand, dim=-1)

                    # ------------------------------------------------------------
                    # helper_variable progress / memory (robust stage machine)
                    # hv[0]=stage (0..5)
                    # hv[1]=stable_cnt
                    # hv[2]=prev_dxy
                    # hv[3]=prev_h
                    # hv[4]=prev_grasp
                    # hv[5]=armed_cnt
                    # hv[6]=tsr (time since proper release)
                    # hv[7]=had_proper_release
                    # hv[8]=success_latch
                    # hv[9]=unused
                    # ------------------------------------------------------------
                    hv = self.helper_variable
                    if hv is None or hv.shape[0] != self.num_envs or hv.shape[1] < 10:
                        hv = torch.zeros((self.num_envs, 10), device=device)

                    stage = _safe(hv[:, 0])
                    stable_cnt = _safe(hv[:, 1])
                    prev_dxy = _safe(hv[:, 2])
                    prev_h = _safe(hv[:, 3])
                    prev_grasp = torch.clamp(_safe(hv[:, 4]), 0.0, 1.0)
                    armed_cnt = _safe(hv[:, 5])
                    tsr = _safe(hv[:, 6])
                    had_release = torch.clamp(_safe(hv[:, 7]), 0.0, 1.0)
                    success_latch = torch.clamp(_safe(hv[:, 8]), 0.0, 1.0)

                    prev_dxy = torch.where(prev_dxy > 0.0, prev_dxy, d_xy.detach())
                    prev_h = torch.where(prev_h != 0.0, prev_h, h.detach())

                    # Stable grasp proxy (slightly easier than previous; avoid blocking exploration)
                    stable_grasp = (grasped > 0.5) & (rel_vel < 0.60) & (hand_obj_dist < 0.17)
                    stable_cnt = torch.clamp(stable_cnt + stable_grasp.float(), 0.0, 300.0)
                    stable_enough = stable_cnt >= 3.0

                    # Stages / thresholds (make "place above basket" crisper to improve conversion to success)
                    lift_h: float = 0.16
                    place_h: float = 0.15  # height to be "above opening" before release

                    lifted_now = (h > lift_h) & (grasped > 0.5) & stable_enough
                    align_xy: float = max(0.028, 0.65 * rad)  # tighter than before to increase inside_site probability
                    aligned_now = (d_xy < align_xy) & (h > place_h) & (grasped > 0.5) & stable_enough

                    # Require being above opening AND near center for a couple steps before "armed"
                    in_band_lo: float = 0.14
                    in_band_hi: float = 0.24
                    in_release_band = (h > in_band_lo) & (h < in_band_hi)
                    arm_ready = aligned_now & in_release_band
                    armed_cnt = torch.clamp(armed_cnt + arm_ready.float(), 0.0, 20.0)
                    armed_enough = armed_cnt >= 3.0

                    # Release event and proper release detection
                    release_event = ((prev_grasp > 0.5) & (grasped < 0.5)).float()
                    proper_release = (release_event > 0.5) & armed_enough & (d_xy < max(0.05, 0.95 * rad)) & in_release_band

                    had_release = torch.maximum(had_release, proper_release.float())

                    # Time since proper release (for limited post-shaping)
                    tsr = torch.where(proper_release, torch.zeros_like(tsr), tsr + 1.0)
                    tsr = torch.where(had_release > 0.5, tsr, torch.zeros_like(tsr))
                    tsr = torch.clamp(tsr, 0.0, 80.0)

                    # Success latch to give a terminal-ish bonus without relying on env termination
                    success_latch = torch.maximum(success_latch, success)

                    # Monotonic stage
                    stage = torch.maximum(stage, stable_enough.float() * 1.0)
                    stage = torch.maximum(stage, lifted_now.float() * 2.0)
                    stage = torch.maximum(stage, aligned_now.float() * 3.0)
                    stage = torch.maximum(stage, had_release * 4.0)
                    stage = torch.maximum(stage, success_latch * 5.0)

                    # Reset arming if grasp lost before proper release; reset after success
                    armed_cnt = torch.where((grasped < 0.5) & (had_release < 0.5), torch.zeros_like(armed_cnt), armed_cnt)
                    armed_cnt = torch.where(success_latch > 0.5, torch.zeros_like(armed_cnt), armed_cnt)

                    # Write back memory
                    hv[:, 0] = stage.detach()
                    hv[:, 1] = stable_cnt.detach()
                    hv[:, 2] = d_xy.detach()
                    hv[:, 3] = h.detach()
                    hv[:, 4] = grasped.detach()
                    hv[:, 5] = armed_cnt.detach()
                    hv[:, 6] = tsr.detach()
                    hv[:, 7] = had_release.detach()
                    hv[:, 8] = success_latch.detach()
                    self.helper_variable = hv

                    g_stable = (stage >= 1.0).float()
                    g_lifted = (stage >= 2.0).float()
                    g_aligned = (stage >= 3.0).float()
                    g_released = (stage >= 4.0).float()

                    # ------------------------------------------------------------
                    # Reward terms (changes from best-so-far)
                    # 1) Reduce "release_event" dominance by: smaller weight + shaped "armed staying" reward.
                    # 2) Increase conversion to actual success by: stronger centering while aligned, and post-release inside_xy.
                    # 3) Replace/soften progress deltas with bounded potential differences (less noisy).
                    # 4) Add explicit action penalty.
                    # ------------------------------------------------------------

                    # HOLD: stable grasp + low slip
                    temp_slip: float = 0.30
                    r_low_slip = torch.exp(-torch.clamp(rel_vel, 0.0, 8.0) / temp_slip)
                    temp_hand: float = 0.14
                    r_hand_close = torch.exp(-torch.clamp(hand_obj_dist, 0.0, 0.9) / temp_hand)
                    r_hold = (0.65 * r_low_slip + 0.35 * r_hand_close) * (grasped > 0.5).float()

                    # LIFT: be above lift_h
                    temp_lift: float = 0.07
                    lift_err = torch.clamp(torch.relu(lift_h - h), 0.0, 2.0)
                    r_lift = torch.exp(-lift_err / temp_lift) * (grasped > 0.5).float() * g_stable

                    # MOVE XY while lifted
                    temp_xy: float = 0.10
                    r_xy = torch.exp(-torch.clamp(d_xy, 0.0, 3.0) / temp_xy) * (grasped > 0.5).float() * g_lifted

                    # Potential-based progress (less spiky than raw delta; also works near goal)
                    temp_phi: float = 0.16
                    phi = torch.exp(-torch.clamp(d_xy, 0.0, 3.0) / temp_phi)
                    phi_prev = torch.exp(-torch.clamp(prev_dxy, 0.0, 3.0) / temp_phi)
                    r_phi_prog = torch.clamp(phi - phi_prev, -0.10, 0.10) * (grasped > 0.5).float() * g_lifted

                    # PRE-PLACE: tight center + height band, only when grasped
                    temp_band: float = 0.06
                    band_err = torch.clamp(torch.relu(in_band_lo - h) + torch.relu(h - in_band_hi), 0.0, 3.0)
                    r_band = torch.exp(-band_err / temp_band)

                    # Stronger centering when approaching place stage (helps actual inside_site)
                    temp_center: float = max(0.020, 0.45 * rad)
                    r_center = torch.exp(-torch.clamp(d_xy, 0.0, 3.0) / temp_center)
                    r_pre = r_center * r_band * (grasped > 0.5).float() * g_lifted

                    # "Stay armed" reward: encourages hovering aligned+inband before release (replaces over-reliance on release bonus)
                    r_armed_stay = arm_ready.float() * (grasped > 0.5).float() * g_lifted

                    # RELEASE: reward only for proper release; penalize premature releases
                    temp_rel_xy: float = 0.06
                    r_rel_xy = torch.exp(-torch.clamp(d_xy, 0.0, 3.0) / temp_rel_xy)
                    r_release_good = release_event * proper_release.float() * r_rel_xy * r_band
                    r_release_bad = -release_event * (1.0 - proper_release.float())  # includes unarmed + wrong position

                    # POST-RELEASE window
                    post_window = torch.exp(-torch.clamp(tsr, 0.0, 80.0) / 14.0)  # in (0,1]
                    post_mask = (1.0 - grasped) * g_released * post_window

                    # Keep object inside in XY during fall (important for conversion)
                    temp_inside: float = max(0.020, 0.55 * rad)
                    r_inside_xy = torch.exp(-torch.clamp(d_xy, 0.0, 3.0) / temp_inside) * post_mask

                    # Encourage it to go down after release (bounded)
                    dh = torch.clamp(prev_h - h, -0.12, 0.12)
                    r_down = torch.relu(dh) * post_mask

                    # Low + settle only if inside_site (avoid rewarding dropping outside)
                    temp_low: float = 0.055
                    z_err = torch.clamp(torch.relu(obj_z - 0.1), 0.0, 2.0)
                    r_low = torch.exp(-z_err / temp_low)
                    temp_settle: float = 1.05
                    settle_speed = torch.linalg.norm(obj_vel - site_vel, dim=-1)
                    r_settle = torch.exp(-torch.clamp(settle_speed, 0.0, 12.0) / temp_settle)

                    r_drop = post_mask * inside_site.float() * r_low * r_settle

                    # Success bonus (latching gives at least one-step big reward)
                    r_success = success_latch

                    # ------------------------------------------------------------
                    # Regularization / safety (smooth motion, avoid singularities)
                    # ------------------------------------------------------------
                    qd_l2 = torch.sum(torch.clamp(robot_qd, -60.0, 60.0) ** 2, dim=-1)

                    q_min = self.robot_dof_lower_limits
                    q_max = self.robot_dof_upper_limits
                    q_center = 0.5 * (q_min + q_max)
                    q_half = torch.clamp(0.5 * (q_max - q_min), min=eps)
                    q_norm = _safe(torch.abs((robot_q - q_center) / q_half))
                    jl_margin: float = 0.88
                    jl_pen = torch.sum(torch.relu(q_norm - jl_margin) ** 2, dim=-1)

                    # manipulability reward (bounded [0,1))
                    temp_manip: float = 0.55
                    r_manip = 1.0 - torch.exp(-torch.clamp(manipulability, 0.0, 12.0) / temp_manip)

                    # mild orientation prior: keep near "pointing down" (qx,qw near 0)
                    qx = hand_quat[:, 0]
                    qw = hand_quat[:, 3]
                    temp_ori: float = 0.40
                    ori_err = torch.clamp(torch.abs(qx) + torch.abs(qw), 0.0, 2.0)
                    r_ori = torch.exp(-ori_err / temp_ori)

                    # keep speeds controlled
                    obj_speed = torch.linalg.norm(obj_vel, dim=-1)
                    tcp_speed = torch.linalg.norm(tcp_vel, dim=-1)
                    temp_speed: float = 1.35
                    r_speed = torch.exp(-torch.clamp(obj_speed + tcp_speed, 0.0, 25.0) / temp_speed)

                    # ------------------------------------------------------------
                    # Weights (rebalance: reduce release-event farming; increase centering & post-release inside_xy)
                    # ------------------------------------------------------------
                    w_hold = 0.18
                    w_lift = 0.30
                    w_xy = 0.18
                    w_phi_prog = 7.0
                    w_pre = 0.95
                    w_armed_stay = 0.45

                    w_release_good = 5.0
                    w_release_bad = 4.0

                    w_inside_xy = 4.5
                    w_down = 7.0
                    w_drop = 5.0

                    w_success = 95.0

                    w_manip = 0.06
                    w_ori = 0.02
                    w_speed = 0.10

                    w_qd = 0.0017
                    w_jl = 0.06
                    w_act = 0.02

                    rew_hold = w_hold * r_hold
                    rew_lift = w_lift * r_lift
                    rew_xy = w_xy * r_xy
                    rew_phi_prog = w_phi_prog * r_phi_prog
                    rew_pre = w_pre * r_pre
                    rew_armed_stay = w_armed_stay * r_armed_stay

                    rew_release = w_release_good * r_release_good + w_release_bad * r_release_bad

                    rew_inside_xy = w_inside_xy * r_inside_xy
                    rew_down = w_down * r_down
                    rew_drop = w_drop * r_drop

                    rew_success = w_success * r_success

                    rew_manip = w_manip * r_manip
                    rew_ori = w_ori * r_ori
                    rew_speed = w_speed * r_speed

                    rew_qd = -w_qd * qd_l2
                    rew_jl = -w_jl * jl_pen
                    rew_act = -w_act * act_pen

                    reward = (
                        rew_hold
                        + rew_lift
                        + rew_xy
                        + rew_phi_prog
                        + rew_pre
                        + rew_armed_stay
                        + rew_release
                        + rew_inside_xy
                        + rew_down
                        + rew_drop
                        + rew_success
                        + rew_manip
                        + rew_ori
                        + rew_speed
                        + rew_qd
                        + rew_jl
                        + rew_act
                    )

                    reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
                    assert torch.isfinite(reward).all(), "Non-finite reward detected."

                    individual_rewards = {
                        "hold": rew_hold,
                        "lift": rew_lift,
                        "move_xy": rew_xy,
                        "phi_xy_progress": rew_phi_prog,
                        "pre_place": rew_pre,
                        "armed_stay": rew_armed_stay,
                        "release_event": rew_release,
                        "inside_xy_post": rew_inside_xy,
                        "down_progress": rew_down,
                        "drop_in": rew_drop,
                        "success": rew_success,
                        "manipulability": rew_manip,
                        "hand_ori": rew_ori,
                        "controlled_speed": rew_speed,
                        "joint_vel_reg": rew_qd,
                        "joint_limit_reg": rew_jl,
                        "action_reg": rew_act,
                    }
                    return reward, individual_rewards
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


    