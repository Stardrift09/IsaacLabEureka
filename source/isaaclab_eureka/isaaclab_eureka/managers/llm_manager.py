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

                    # ---------------------------------------------------------------------
                    # Core issue from logs:
                    # - Components that should be in [0,1] are being tracked at ~30-50, which
                    #   indicates you are logging *episode sums* (not per-step means). So we
                    #   must keep per-step rewards small and very well-shaped.
                    # - dist in training hovers ~3-6m equivalent (very far) while r_reach etc.
                    #   still accumulate large episode sums -> agent can get reward without
                    #   actually reaching and grasping.
                    # - Absolute heights appear badly offset (obj_h_env ~2-3, grip_h_env ~4-5).
                    #   Using absolute height thresholds is brittle. Use *height change* (delta)
                    #   as primary lift signal, plus a weak absolute clamp.
                    # - No gripper state available -> infer "grasp/hold" by: near + object
                    #   moving up while end-effector remains near (proxy).
                    # ---------------------------------------------------------------------

                    # ----------------------------
                    # Sanitize required state
                    # ----------------------------
                    grasp_pos_w = torch.nan_to_num(self.robot_grasp_pos, nan=0.0, posinf=0.0, neginf=0.0)
                    obj_pos_w = torch.nan_to_num(self.target_object.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
                    env_origins = torch.nan_to_num(self.scene.env_origins, nan=0.0, posinf=0.0, neginf=0.0)

                    grasp_pos_env = grasp_pos_w - env_origins
                    obj_pos_env = obj_pos_w - env_origins

                    to_obj_w = torch.nan_to_num(obj_pos_w - grasp_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
                    dist = torch.linalg.norm(to_obj_w, dim=-1)
                    dist = torch.nan_to_num(dist, nan=10.0, posinf=10.0, neginf=10.0)
                    dist = torch.clamp(dist, 0.0, 5.0)

                    obj_h = torch.nan_to_num(obj_pos_env[:, 2], nan=0.0, posinf=0.0, neginf=0.0)
                    grip_h = torch.nan_to_num(grasp_pos_env[:, 2], nan=0.0, posinf=0.0, neginf=0.0)
                    obj_h = torch.clamp(obj_h, -2.0, 20.0)
                    grip_h = torch.clamp(grip_h, -2.0, 20.0)

                    # Proxy for motion smoothness
                    joint_vel = torch.nan_to_num(self._robot.data.joint_vel, nan=0.0, posinf=0.0, neginf=0.0)
                    vel_l2 = torch.linalg.norm(joint_vel, dim=-1)
                    vel_l2 = torch.nan_to_num(vel_l2, nan=0.0, posinf=100.0, neginf=0.0)
                    vel_l2 = torch.clamp(vel_l2, 0.0, 100.0)

                    # ----------------------------
                    # Height delta (robust lift signal)
                    # ----------------------------
                    # Maintain an EMA-like "initial/table" reference without assuming true table height.
                    # We update a per-env baseline that slowly follows the object's height when the gripper is far
                    # (i.e., likely not grasped), so lifting while near produces a positive delta.
                    if not hasattr(self, "_obj_h_ref_eureka"):
                        self._obj_h_ref_eureka = obj_h.clone()
                    else:
                        self._obj_h_ref_eureka = torch.nan_to_num(self._obj_h_ref_eureka, nan=0.0, posinf=0.0, neginf=0.0)
                        self._obj_h_ref_eureka = torch.clamp(self._obj_h_ref_eureka, -2.0, 20.0)

                    # Far gate: when far, allow reference to track object (prevents drift)
                    far_for_ref: float = 0.20
                    ref_track_rate: float = 0.02  # slow tracking for stability
                    far_mask = (dist > far_for_ref).to(dtype=torch.float32, device=device)
                    self._obj_h_ref_eureka = self._obj_h_ref_eureka + ref_track_rate * far_mask * (obj_h - self._obj_h_ref_eureka)

                    # On reset-like situations (object height NaN/inf already handled), clamp again
                    self._obj_h_ref_eureka = torch.clamp(torch.nan_to_num(self._obj_h_ref_eureka, nan=0.0), -2.0, 20.0)

                    lift_delta = obj_h - self._obj_h_ref_eureka
                    lift_delta = torch.nan_to_num(lift_delta, nan=0.0, posinf=0.0, neginf=0.0)
                    lift_delta = torch.clamp(lift_delta, -1.0, 5.0)

                    # ----------------------------
                    # Reward components (all per-step bounded)
                    # ----------------------------

                    # (1) Reach: dense shaping over a wider range (training dist ~3-6)
                    reach_temp: float = 0.60
                    r_reach = torch.exp(-dist / (reach_temp + eps))
                    r_reach = torch.clamp(r_reach, 0.0, 1.0)

                    # (2) Very near gate (for grasp/lift shaping)
                    near_temp: float = 0.08
                    near_gate = torch.exp(-0.5 * (dist / (near_temp + eps)) ** 2)
                    near_gate = torch.clamp(near_gate, 0.0, 1.0)

                    # (3) Encourage gripper to be slightly above object when approaching (reduces premature closing)
                    dz = torch.abs(grip_h - obj_h)
                    dz = torch.nan_to_num(dz, nan=5.0, posinf=5.0, neginf=5.0)
                    dz = torch.clamp(dz, 0.0, 5.0)
                    align_temp: float = 0.15
                    r_align_z = torch.exp(-0.5 * (dz / (align_temp + eps)) ** 2)
                    r_align_z = torch.clamp(r_align_z, 0.0, 1.0)

                    # (4) Close-hold proxy: only reward being extremely close when already near
                    close_temp: float = 0.03
                    r_close = torch.exp(-0.5 * (dist / (close_temp + eps)) ** 2)
                    r_close = torch.clamp(r_close, 0.0, 1.0)
                    r_close_when_near = near_gate * r_close

                    # (5) Lift via delta-height: only matters if near (proxy for grasping)
                    # Scale: 0 at <=0, saturate around ~8cm lift.
                    lift_scale: float = 0.08
                    lift_prog = torch.clamp(lift_delta / (lift_scale + eps), 0.0, 2.0)
                    lift_shape_temp: float = 0.50
                    r_lift = 1.0 - torch.exp(-lift_prog / lift_shape_temp)
                    r_lift = torch.clamp(r_lift, 0.0, 1.0)
                    r_lift = near_gate * r_lift

                    # (6) Hold: reward keeping near while lifted (stabilizes grasp rather than throwing)
                    hold_temp: float = 0.10
                    hold_gate = torch.exp(-0.5 * (dist / (hold_temp + eps)) ** 2)
                    hold_gate = torch.clamp(hold_gate, 0.0, 1.0)
                    # "Lifted" boolean proxy from delta height (>=5cm)
                    lifted = (lift_delta > 0.05).to(dtype=torch.float32, device=device)
                    r_hold = lifted * hold_gate

                    # (7) Success (sparse): require near + sufficient lift_delta.
                    # Keep sparse bonus moderate to avoid reward hacking via constant terms.
                    success = (dist < 0.05) & (lift_delta > 0.10)
                    r_success = success.to(dtype=torch.float32, device=device)
                    success_1 = (dist < 0.05) & (lift_delta > 0.2)
                    r_success_1 = success_1.to(dtype=torch.float32, device=device)
                    success_2 = (dist < 0.05) & (lift_delta > 0.3)
                    r_success_2 = success_2.to(dtype=torch.float32, device=device)
                    success_3 = (dist < 0.05) & (lift_delta > 0.4)
                    r_success_3 = success_3.to(dtype=torch.float32, device=device)
                    # ----------------------------
                    # Penalties (small, bounded)
                    # ----------------------------
                    vel_pen_temp: float = 6.0
                    p_vel = 1.0 - torch.exp(-vel_l2 / (vel_pen_temp + eps))
                    p_vel = torch.clamp(p_vel, 0.0, 1.0)

                    # Penalize being far to prevent policies that ignore the object and still get align reward
                    far_margin: float = 0.25
                    far_pen_temp: float = 0.50
                    p_far = 1.0 - torch.exp(-torch.clamp(dist - far_margin, min=0.0) / (far_pen_temp + eps))
                    p_far = torch.clamp(p_far, 0.0, 1.0)

                    # ----------------------------
                    # Combine: emphasize actual progress (reach -> close -> lift -> hold -> success)
                    # Keep per-step total typically in ~[0, 3].
                    # ----------------------------
                    w_reach: float = 0.35
                    w_align: float = 0.15
                    w_close: float = 0.25
                    w_lift: float = 0.80
                    w_hold: float = 0.30
                    w_success: float = 2.00
                    w_vel: float = 0.01
                    w_far: float = 0.05

                    reward = (
                        w_reach * r_reach
                        + w_align * (r_align_z * (0.2 + 0.8 * r_reach))  # alignment matters more when reaching
                        + w_close * r_close_when_near
                        + w_lift * r_lift
                        + w_hold * r_hold
                        + w_success * r_success
                        + w_success * r_success_1
                        + w_success * r_success_2
                        + w_success * r_success_3
                        - w_vel * p_vel
                        - w_far * p_far
                    )

                    reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
                    reward = torch.clamp(reward, -1.0, 4.0)
                    assert torch.isfinite(reward).all(), "Non-finite reward encountered."

                    individual = {
                        "r_reach": r_reach,
                        "r_align_z": r_align_z,
                        "near_gate": near_gate,
                        "r_close_when_near": r_close_when_near,
                        "r_lift": r_lift,
                        "r_hold": r_hold,
                        "r_success": r_success,
                        "p_vel": p_vel,
                        "p_far": p_far,
                        "dist": dist,
                        "obj_h_env": obj_h,
                        "grip_h_env": grip_h,
                        "dz": dz,
                        "obj_h_ref": self._obj_h_ref_eureka,
                        "lift_delta": lift_delta,
                    }
                    return reward, individual
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
