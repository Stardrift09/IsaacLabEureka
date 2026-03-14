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

    "AToB": {
        "description": "Control the eef to move to the desired position, shape the reward well to avoid being stuck at local minimum, add reward on manipulability",
        "success_metric": """obj_xyz = self.target_pos[env_ids]
    grasp_pos = self.robot_grasp_pos[env_ids]
    dist2 = ((obj_xyz - grasp_pos)**2).sum(dim=-1)
    tolerance=torch.tensor(0.1,device=self.device)
    inside_site = dist2 < tolerance**2
    extras['Eureka/success_metric'] =inside_site.float().mean()""",
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
    
    # # This one worked in grasping and lifting object
    # "LivingRoomScene1PickUpTheAlphabetSoupAndPutItInTheBasket": {
    #     "description": "control the franka arm to pick up the target object without pushing it away, lift it up and drop it in the basket. This is a long horizon task so use the self.helper_variable for keeping current stage of the task.",
    #     "success_metric": (
    #      """
    # extras['Eureka/success_metric'] = (self.target_object.data.root_pos_w[env_ids, 2] > 0.4).float().mean()"""
    #     ),
    #     "success_metric_to_win": 0.95,
    #     "success_metric_tolerance": 0.02,
    # },



    # THE VERSION THAT IS ABLE TO GRASP THE OBJECT UPSTRAIGHT

    # "TestPickItUp": {
    #     "description": "Control the Franka arm to first move to a point 10 cm above the target object position while aligning precisely with the target orientation. Then descend slowly while keeping the gripper open to avoid collision.  When the object center is close to the gripper tcp, close the gripper to grasp the object, avoiding any pushing. After securing the grasp, lift the object gently. Ensure the entire motion is slow, stable, and well-controlled. Important: avoid being stuck by local minima by carefully designing reward terms.",
    #     "success_metric": (
    #      """grasped = self._grasp_detection(env_ids) # [num_envs, 1] 1 means two fingers have contact force against object, thereby grasping
    # object_default_state = self.target_object.data.default_root_state
    # high_enough = self.target_object.data.root_pos_w[env_ids, 2] > object_default_state[env_ids,self.input_direction] + 0.1
    # target_object_current_pose = self.target_object.data.root_quat_w[env_ids]        # shape (N, 4)
    # target_object_desired_pose = object_default_state[env_ids,3:7]         # shape (N, 4)
    # target_object_current_pose_inv = quat_conjugate(target_object_current_pose)
    # # relative rotation
    # q_error = quat_mul(target_object_desired_pose, target_object_current_pose_inv)
    # q_error = q_error / torch.norm(q_error, dim=-1, keepdim=True).clamp_min(1e-9)
    # q_error = torch.where(q_error[:, 0:1] < 0, -q_error, q_error)
    # angle_error = 2 * torch.acos(torch.clamp(q_error[:, 0], -1.0, 1.0))
    # small_rotation = angle_error < 0.8
    # terminated = small_rotation & high_enough & grasped.bool().squeeze()
    # extras['Eureka/success_metric'] = terminated.float().mean()"""
    #     ),
    #     "success_metric_to_win": 1.0,
    #     "success_metric_tolerance": 0.05,
    # },


    "TestPickItUp": {
        "description": """Pick up the object, move to above the basket, and drop it inside the basket. This is a multi-stage, long-horizon task. Use `self.helper_variable` to track task progress if necessary.
        Do not shape reward for next stage before the previous stage is well finished!
        """,
        "success_metric": (
         """site_height = self.target_site_corners_world[1,2] - self.target_site_corners_world[0,2]
    low_enough = self.target_object.data.root_pos_w[env_ids, 2] <site_height
    obj_xy = self.target_object.data.root_pos_w[env_ids, :2]
    site_pos = self.target_site.data.root_pos_w[env_ids, :2]
    dist2 = ((obj_xy - site_pos)**2).sum(dim=-1)
    inside_site = dist2 < self.target_site_radius**2
    extras['Eureka/success_metric'] = (inside_site & low_enough).float().mean()"""
        ),
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.05,
    },



    "TestPutItInTheBasket": {
        "description": """**Initial condition:**  
        The robot gripper rigidly grasps the target object with a stable, non-slipping grasp. The basket (target receptacle) pose is known.

**Objective:**  
Lift it safely, and place it inside the basket. This is a multi-stage, long-horizon task. Use `self.helper_variable` to track task progress. Add regularization on the robot action, avoid singularity and weird motion.

## Task sequence and constraints

### 1) Hold
- Make sure a stable grasp before lifting.

### 2) Lift
- Move the grasped object horizontally toward the basket.
- Keep the motion smooth and controlled.

### 4) Place
- Position the object above the basket opening.
- Release the object so that it falls inside the basket.
        """,
        "success_metric": (
         """site_height = self.target_site_corners_world[1,2] - self.target_site_corners_world[0,2]
    low_enough = self.target_object.data.root_pos_w[env_ids, 2] <site_height
    obj_xy = self.target_object.data.root_pos_w[env_ids, :2]
    site_pos = self.target_site.data.root_pos_w[env_ids, :2]
    dist2 = ((obj_xy - site_pos)**2).sum(dim=-1)
    inside_site = dist2 < self.target_site_radius**2
    extras['Eureka/success_metric'] = (inside_site & low_enough).float().mean()"""
        ),
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.05,
    },


# renaming LivingRoomScene1PickUpTheAlphabetSoupAndPutItInTheBasket to 

    "LivingRoomScene1PickUpTheAlphabetSoupAndPutItInTheBasket": {
        "description": """**Initial condition:**  
The robot starts with its gripper free. The target object is placed on a table. The basket position is known. The robot must avoid pushing the object away before grasping.

**Objective:**  
Pick up the target object in a controlled manner, lift it safely, and place it inside the basket. This is a multi-stage, long-horizon task. Use `self.helper_variable` to track task progress. Add regularization on the robot action, avoid singularity and weird motion.

## Task sequence and constraints

### 1) Approach and grasp
- Move the end-effector toward the object without pushing or sliding it.
- Establish a stable grasp before lifting.

### 2) Lift
- Move the grasped object horizontally toward the basket.
- Keep the motion smooth and controlled.

### 4) Place
- Position the object above the basket opening.
- Release the object so that it falls inside the basket.""",
        "success_metric": (
         """site_height = self.target_site_corners_world[1,2] - self.target_site_corners_world[0,2]
    low_enough = self.target_object.data.root_pos_w[env_ids, 2] <site_height
    obj_xy = self.target_object.data.root_pos_w[env_ids, :2]
    site_pos = self.target_site.data.root_pos_w[env_ids, :2]
    dist2 = ((obj_xy - site_pos)**2).sum(dim=-1)
    inside_site = dist2 < self.target_site_radius**2
    extras['Eureka/success_metric'] = (inside_site & low_enough).float().mean()"""
        ),
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.05,
    },




    "TestStageAsFeedback": {
        "description": """**Initial condition:**  
        The robot gripper rigidly grasps the target object with a stable, non-slipping grasp. The basket (target receptacle) pose is known.

**Objective:**  
Lift it safely, and place it inside the basket. This is a multi-stage, long-horizon task. Use `self.helper_variable` to track task progress. Add regularization on the robot action, avoid singularity and weird motion.

## Task sequence and constraints

### 1) Hold
- Make sure a stable grasp before lifting.

### 2) Lift
- Move the grasped object horizontally toward the basket.
- Keep the motion smooth and controlled.

### 4) Place
- Position the object above the basket opening.
- Release the object so that it falls inside the basket.
        """,
        "success_metric": (
         """site_height = self.target_site_corners_world[1,2] - self.target_site_corners_world[0,2]
    low_enough = self.target_object.data.root_pos_w[env_ids, 2] <site_height
    obj_xy = self.target_object.data.root_pos_w[env_ids, :2]
    site_pos = self.target_site.data.root_pos_w[env_ids, :2]
    dist2 = ((obj_xy - site_pos)**2).sum(dim=-1)
    inside_site = dist2 < self.target_site_radius**2
    extras['Eureka/success_metric'] = (inside_site & low_enough).float().mean()"""
        ),
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.05,
    },


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
            "extras['Eureka/success_metric'] = (0.39 < self._cabinet.data.joint_pos[env_ids, 1]).float().mean()"
        ),
        "success_metric_to_win": 0.9,
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
