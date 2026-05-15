# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

"""Template strings used for prompting in Isaac Lab Eureka."""


DIRECT_WORKFLOW_REWARD_FORMATTING_INSTRUCTIONS = """
Your reward function should use useful variables from the environment as inputs.
It must comply to the following signature exactly:

def _get_rewards_eureka(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ...
    return reward, individual_rewards_dict

Your individual_rewards_dict should keys not start with Eureka/ as that will be handled by code.
Make sure any new tensor or variable you introduce is on the same device as self.device.
The output of the reward function should consist of two items:
    (1) the total reward, which has a dimension of (self.num_envs,) and is a torch.Tensor,
    (2) a dictionary of each individual, unweighted reward component. The reward components should be designed to be informative, enabling identification of where the policy fails and providing actionable insights for improving the reward function.
The code output should be formatted as a python code string: "```python ... ```" and contain only the get_rewards_eureka function.

Some helpful tips for writing the reward function code:
    (1) You are using Isaaclab 2.3. Design the reward to be GPU-safe and training-stable: avoid unguarded division, ensure all exponentials and normalizations are bounded, sanitize all environment state inputs with torch.nan_to_num, and assert the reward remains finite for every timestep, including resets and failure states.
    (2) You may find it helpful to normalize the reward to a fixed range by applying transformations like torch.exp to the overall reward or its components. Per step reward should be in [-5, 5].
    (3) If you choose to transform a reward component, then you must also introduce a temperature parameter inside the transformation function; this parameter must be a named variable in the reward function and it must not be an input variable. Each transformed reward component should have its own temperature variable
    (4) Make sure the type of each input variable is correctly specified; a float input variable should not be specified as torch.Tensor
    (5) Most importantly, the reward code's input variables must contain only attributes of the provided environment class definition (namely, variables that have prefix self.). Under no circumstance can you introduce new input variables.
    (6) Robot joint limits are give as self.robot_dof_upper_limits and self.robot_dof_lower_limits, 1D tensor vector with 9 elements. Have correct penalty to avoid being close to joint limits!
    (7) For long-horizon tasks, when stage information is available, leave the previous stages unchanged unless intervention is necessary.
    (8) Gamma is 0.9 so ensure that success has enough reward to be favored over hovering. (Should be 10 times larger than the best non-success reward)
    (9) For staged tasks, later stages should provide higher value than earlier stages, but avoid creating persistent rewards that make the agent prefer waiting inside an intermediate stage. Stage bonuses should primarily reward reaching a stage, and progress terms should encourage continued advancement.

        If using persistent stage bonuses, ensure that the discounted cumulative value of staying in any non-terminal stage is lower than the value of progressing to the next stage or completing the task. With gamma = 0.9, a persistent per-step reward h can be worth up to h / (1 - gamma), so terminal or success rewards must dominate that value.

        Prefer combining:
            (1) bounded dense shaping within each stage,
            (2) a small persistent stage indicator bonus,
            (3) progress/improvement rewards toward the next stage,
            (4) a time penalty or stagnation penalty,
            (5) a large success reward or terminal success bonus.

    """


DIRECT_WORKFLOW_INITIAL_PROMPT = """
You are a reward engineer trying to write reward functions to solve reinforcement learning tasks as effective as possible.
Your goal is to write a reward function for the environment that will help the agent learn the task described in text.
""" + DIRECT_WORKFLOW_REWARD_FORMATTING_INSTRUCTIONS


DIRECT_WORKFLOW_INITIAL_PROMPT_CURRICULUM = """
You are a reward engineer designing reward functions for curriculum learning.

The provided checkpoint already enables the agent to successfully hover above the target object(stage 4). Your task is to design a reward function that builds on top of this capability and do the last stage task: dropping the object and make sure it falls in the basket.

Focus on shaping rewards that progressively encourage the next stages of the behavior, ensuring smooth learning transitions from the current capability to the final objective.
""" + DIRECT_WORKFLOW_REWARD_FORMATTING_INSTRUCTIONS


TASK_FAILURE_FEEDBACK_PROMPT = """
Executing the reward function code above has the following error: {traceback_msg}.
Please fix the bug and provide a new, improved reward function!
""" + DIRECT_WORKFLOW_REWARD_FORMATTING_INSTRUCTIONS


TASK_SUCCESS_PRE_FEEDBACK_PROMPT = """
We trained a RL policy using the provided reward function code and tracked the values of the individual components in the reward function as well as global policy metrics such as success rates and episode lengths after every {feedback_subsampling} epochs and the maximum, mean, minimum values encountered:
"""
LLM_TASK_FEEDBACK_PROMPT = """In addition, we also have the following feedback string from a vision-language model that analyzes the policy's output videos:
{llm_task_feedback}
"""

TASK_SUCCESS_POST_FEEDBACK_PROMPT = """
Please carefully analyze the feedback and provide a new, improved reward function that can better solve the task.

Important context:
- The reported statistics are aggregated over ALL parallel environments.
- Values correspond to environments that have finished episodes at that iteration.

When analyzing:
- It is acceptable to focus on the dominant (majority) behavior reflected in the statistics.
- You do NOT need to infer rare edge cases or hidden distributions.
- However, your interpretation of the statistics must be logically correct and consistent with the observed values.

Core objective:
- The goal is to make the agent successfully complete the task.
- Improving reward components is a means to this goal, NOT the goal itself.

Guidelines for analysis:

(1) Task-first reasoning:
    - Always prioritize whether the current policy is progressing toward solving the task.
    - Identify what is preventing task completion (e.g., not lifting high enough, not moving to target, unstable grasp).
    - Focus on fixing these bottlenecks.

(2) Reward components:
    - Only modify reward components if they are clearly:
        * Misaligned with the task
        * Not providing useful gradients
        * Causing incorrect behavior
    - Avoid unnecessary tuning if the reward already supports correct behavior.

(3) Long-horizon tasks:
    - Low success rate early is expected.
    - Focus on whether meaningful stage progress is happening.
    - If the agent is stuck, redesign rewards to better guide the next stage.

(4) Flat or ineffective rewards:
    - If a reward component stays nearly constant, it is not being optimized.
    - Consider:
        (a) Rescaling
        (b) Reformulating
        (c) Removing it

(5) Reward scaling:
    - Ensure no single reward dominates excessively.
    - Maintain a balance so all relevant signals contribute to learning.

(6) Stage-based reasoning:
    - Identify which stage the agent is currently stuck in.
    - Improve rewards that directly help transition to the next stage.
    - Avoid over-tuning earlier stages unless they block progress.

Instructions:
- First, analyze the current behavior and identify the main failure point in solving the task.
- Then, evaluate which reward components are responsible for this failure (if any).
- Finally, propose a revised reward function that improves task completion.

Focus on practical improvements that help the agent succeed, rather than purely optimizing reward metrics.
""" + DIRECT_WORKFLOW_REWARD_FORMATTING_INSTRUCTIONS


DIRECT_WORKFLOW_TASK_PROMPT = """
Write a reward function for the following task: {task_description}
The success metric is: {success_metric}
The desired task score is: {success_metric_to_win}
Here is how we get the observations and update important intermediate values from the environment:
{get_observations_method_as_string}
"""


REPLAY_FEEDBACK_PROMPT = """We provide you the output of the reward function on some successful demonstrations as follows, and you can utilize it for better reward generation:
{replay_feedback_string}"""


BEST_ITERATION_FEEDBACK_PROMPT = """We also keep track of the best reward function in last iterations. The best reward function we have so far has the following feedback string: 
task score: {success_metric}
reward function: {gpt_reward_method}
Task feedback: {task_feedback}
"""