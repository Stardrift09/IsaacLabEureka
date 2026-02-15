# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

TASKS_CFG = {
    "Isaac-Cartpole-Direct-v0": {
        "description": "balance a pole on a cart so that the pole stays upright",
        "success_metric": "extras['Eureka/success_metric'] = self.episode_length_buf[env_ids].float().mean() / self.max_episode_length",
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.01,
    },
    "Isaac-Quadcopter-Direct-v0": {
        "description": (
            "bring the quadcopter to the target position: self._desired_pos_w, while making sure it flies smoothly"
        ),
        "success_metric": (
            "extras['Eureka/success_metric'] = torch.linalg.norm(self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1).mean()"
        ),
        "success_metric_to_win": 0.0,
        "success_metric_tolerance": 0.2,
    },


    # "LivingRoomScene1PickUpTheCreamCheeseAndPutItInTheBasket-v0": {
    #     "description": "control the franka arm to pick up the cream_cheese, neglecting basket for now. Grasp from above to avoid pushing the object away",
    #     "success_metric": (
    #         "torch.exp(-((0.6 - self._cream_cheese.data.root_pos_w[env_ids, 2]) ** 2) / (2 * 0.01**2)).mean()"
    #     ),
    #     "success_metric_to_win": 0.9,
    #     "success_metric_tolerance": 0.02,
    # },
    
    # This one worked in grasping and lifting object
    "LivingRoomScene1PickUpTheAlphabetSoupAndPutItInTheBasket": {
        "description": "control the franka arm to pick up the target object without pushing it away, lift it up and drop it in the basket. This is a long horizon task so use the self.helper_variable for keeping current stage of the task.",
        "success_metric": (
         """
    extras['Eureka/success_metric'] = (self.target_object.data.root_pos_w[env_ids, 2] > 0.4).float().mean()"""
        ),
        "success_metric_to_win": 0.95,
        "success_metric_tolerance": 0.02,
    },

    # "LivingRoomScene1PickUpTheAlphabetSoupAndPutItInTheBasket": {
    #     "description": "control the franka arm to pick up the target object without pushing it away, lift it up and drop it in the basket. This is a long horizon task so use the self.helper_variable for keeping current stage of the task.",
    #     "success_metric": (
    #      """low_enough = self.target_object.data.root_pos_w[env_ids, 2] <0.1
    # obj_xy = self.target_object.data.root_pos_w[env_ids, :2]
    # site_pos = self.target_site.data.root_pos_w[env_ids, :2]
    # dist2 = ((obj_xy - site_pos)**2).sum(dim=-1)
    # inside_site = dist2 < self.target_site_radius**2
    # extras['Eureka/success_metric'] = (inside_site & low_enough).float().mean()"""
    #     ),
    #     "success_metric_to_win": 0.95,
    #     "success_metric_tolerance": 0.02,
    # },




    # "LivingRoomScene1PickUpTheAlphabetSoupAndPutItInTheBasket": {
    #     "description": "control the franka arm to pick up the alphabet soup(target object)",
    #     "success_metric": (
    #         "extras['Eureka/success_metric'] = torch.exp(-((0.4 - self.target_object.data.root_pos_w[env_ids, 2]) ** 2) / (2 * 0.01**2)).mean()"
    #     ),
    #     "success_metric_to_win": 0.9,
    #     "success_metric_tolerance": 0.02,
    # },

    # "ReplayLivingRoomScene1PickUpTheAlphabetSoupAndPutItInTheBasket": {
    #     "description": "control the franka arm to pick up the cream_cheese, neglecting basket for now. Grasp from above to avoid pushing the object away",
    #     "success_metric": (
    #         "torch.exp(-((0.6 - self.target_object.data.root_pos_w[env_ids, 2]) ** 2) / (2 * 0.01**2)).mean()"
    #     ),
    #     "success_metric_to_win": 0.9,
    #     "success_metric_tolerance": 0.02,
    # },


    "Isaac-Franka-Cabinet-Direct-v0": {
        "description": "control the franka arm to open the cabinet drawer",
        "success_metric": (
            "torch.exp(-((0.39 - self._cabinet.data.joint_pos[env_ids, 1]) ** 2) / (2 * 0.01**2)).mean()"
        ),
        "success_metric_to_win": 0.8,
        "success_metric_tolerance": 0.02,
    },

    # this one is too loose

    # "Isaac-Franka-Cabinet-Direct-v0": {
    #     "description": "control the franka arm to open the cabinet drawer",
    #     "success_metric": (
    #         "torch.clamp("
    #         "self._cabinet.data.joint_pos[env_ids, 1] / 0.39, "
    #         "0.0, 1.0"
    #         ").mean()"
    #     ),
    #     "success_metric_to_win": 1.0,
    #     "success_metric_tolerance": 0.05,
    # },
}
"""Configuration for the tasks supported by Isaac Lab Eureka.

`TASKS_CFG` is a dictionary that maps task names to their configuration. Each task configuration
is a dictionary that contains the following keys:

- `description`: A description of the task.
- `success_metric`: A Python expression that computes the success metric for the task.
- `success_metric_to_win`: The threshold for the success metric to win the task and stop.
- `success_metric_tolerance`: The tolerance for the success metric to consider the task successful.
"""
