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

    "TurnOnTheStove": {
        "description": """**Objective:** Turn on the stove by rotating the button knob.

The Franka arm must approach the stove button and apply force to rotate the button_joint
from 0 to > 1.5 radians (out of a 2.1 rad limit).

## Key attributes
- `self.robot_grasp_pos`: TCP world position [num_envs, 3]
- `self.hand_to_button_pos`: vector from TCP to button center [num_envs, 3]
- `self.button_pos_w`: button world position [num_envs, 3]
- `self._stove.data.joint_pos[:, self.button_joint_idx]`: current button angle [num_envs]
- `self.cfg.button_success_threshold`: 1.5 rad

## Reward hints
1. Approach: reward decreasing distance between TCP and button.
2. Press: reward increasing button_joint angle.
3. Add action regularization for smooth motion.
""",
        "success_metric": (
            """button_angle = self._stove.data.joint_pos[env_ids, self.button_joint_idx]
    extras['Eureka/success_metric'] = (button_angle > self.cfg.button_success_threshold).float().mean()"""
        ),
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.05,
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
    #      """grasped = self._grasp_detection(env_ids) # [num_envs] 1 means two fingers have contact force against object, thereby grasping
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
    # terminated = small_rotation & high_enough & grasped
    # extras['Eureka/success_metric'] = terminated.float().mean()"""
    #     ),
    #     "success_metric_to_win": 1.0,
    #     "success_metric_tolerance": 0.05,
    # },


    # "TestPickItUp": {
    #     "description": """Pick up the object, move to above the basket, and drop it inside the basket. This is a multi-stage, long-horizon task. Use `self.helper_variable` to track task progress if necessary.
    #     """,
    #     "success_metric": (
    #      """low_enough = self.target_object.data.root_pos_w[env_ids, 2] <self.target_site_corners_world[1,2]
    # obj_xy = self.target_object.data.root_pos_w[env_ids, :2]
    # site_pos = self.target_site.data.root_pos_w[env_ids, :2]
    # dist2 = ((obj_xy - site_pos)**2).sum(dim=-1)
    # inside_site = dist2 < self.target_site_radius**2

    # success = inside_site & low_enough
    # stage_masked = self.stage[env_ids].clone()
    # stage_masked = stage_masked * (~success).unsqueeze(-1)
    # for i in range(1, self.num_stages): # Log stages except for stage 0
    #     extras[f'Eureka/stage_{i}'] = (stage_masked[:,i]).float().mean()

    # extras['Eureka/success_metric'] = (success).float().mean()
    # """
    #     ),
    #     "success_metric_to_win": 1.0,
    #     "success_metric_tolerance": 0.05,
    # },


    "TestCollideAndPlace": {
        "description": """**Objective:**
Pick up the target object (alphabet_soup), collide it with the tomato_sauce can, then place the target object inside the basket. The basket must remain untouched throughout.

## Task sequence and constraints

### 1) Approach and grasp
- Move the end-effector to the target object and grasp it cleanly without pushing it.

### 2) Collide with tomato_sauce
- Move the grasped object so it physically contacts and displaces the tomato_sauce can by at least 5 cm from its initial position.
- `self.collision_triggered[env_ids]`: bool tensor, True once collision threshold exceeded.
- `self.collision_obj_pos`: tomato_sauce world position [num_envs, 3].

### 3) Place in basket
- After collision, transport the object above the basket and release it inside.
- The basket must not drift more than 3 cm from its spawn position.

## Key observations / attributes
- `self.robot_grasp_pos`: TCP world position [num_envs, 3]
- `self.target_object.data.root_pos_w`: alphabet_soup world pos [num_envs, 3]
- `self.collision_obj_pos`: tomato_sauce world pos [num_envs, 3]
- `self.collision_triggered`: bool [num_envs], True after collision
- `self.basket_corners_world`: basket corners [num_envs, 8, 3]
- `self.target_site.data.root_pos_w`: basket center [num_envs, 3]
- `self.target_site_radius`: scalar, basket XY acceptance radius
- `self.basket_init_pos`: basket initial world pos [num_envs, 3]
- `self.cfg.basket_drift_threshold`: 0.03 m

Add regularisation on joint speed and action rate for smooth motion.
""",
        "success_metric": (
            """obj_z = self.target_object.data.root_pos_w[env_ids, 2]
    basket_z = self.basket_corners_world[env_ids, :, 2]
    basket_bottom_z = basket_z.min(dim=1).values
    basket_top_z = basket_z.max(dim=1).values
    low_enough = obj_z < basket_top_z
    high_enough_for_basket = obj_z > basket_bottom_z
    obj_xy = self.target_object.data.root_pos_w[env_ids, :2]
    site_pos = self.target_site.data.root_pos_w[env_ids, :2]
    dist2 = ((obj_xy - site_pos) ** 2).sum(dim=-1)
    inside_site = dist2 < self.target_site_radius ** 2
    basket_displacement = torch.norm(self.target_site.data.root_pos_w[env_ids] - self.basket_init_pos[env_ids], dim=-1)
    basket_untouched = basket_displacement < self.cfg.basket_drift_threshold
    success = self.collision_triggered[env_ids] & inside_site & low_enough & high_enough_for_basket & basket_untouched
    extras['Eureka/success_metric'] = success.float().mean()"""
        ),
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.05,
    },

    "TestSlightCollideAndPlace": {
        "description": """**Objective:**
Pick up the target object (alphabet_soup), lightly collide it with the tomato_sauce can, then place the target object inside the basket. The collision must be gentle: the tomato_sauce may be displaced at most 0.2 m from its initial position. The basket must remain untouched throughout.

## Task sequence and constraints

### 1) Approach and grasp
- Move the end-effector to the target object and grasp it cleanly without pushing it.

### 2) Light collision with tomato_sauce
- Move the grasped object so it contacts and slightly displaces the tomato_sauce (≥ 5 cm, ≤ 20 cm from its initial position).
- `self.collision_triggered[env_ids]`: True once displacement ≥ 5 cm.
- `self.cfg.collision_max_displacement`: 0.2 m upper bound.
- `self.collision_obj_pos`: tomato_sauce world position [num_envs, 3].

### 3) Place in basket
- After the gentle collision, transport the object above the basket and release it inside.
- The basket must not drift more than 3 cm from its spawn position.

## Key attributes
- `self.robot_grasp_pos`: TCP world position [num_envs, 3]
- `self.target_object.data.root_pos_w`: alphabet_soup world pos [num_envs, 3]
- `self.collision_obj_pos`: tomato_sauce world pos [num_envs, 3]
- `self.collision_object_init_pos`: tomato_sauce spawn pos [num_envs, 3]
- `self.collision_triggered`: bool [num_envs]
- `self.basket_corners_world`: basket corners [num_envs, 8, 3]
- `self.target_site.data.root_pos_w`: basket center [num_envs, 3]
- `self.target_site_radius`: basket XY acceptance radius
- `self.basket_init_pos`: basket initial world pos [num_envs, 3]
- `self.cfg.basket_drift_threshold`: 0.03 m
- `self.cfg.collision_max_displacement`: 0.2 m

Add regularisation on joint speed and action rate for smooth motion.
""",
        "success_metric": (
            """obj_z = self.target_object.data.root_pos_w[env_ids, 2]
    basket_z = self.basket_corners_world[env_ids, :, 2]
    basket_bottom_z = basket_z.min(dim=1).values
    basket_top_z = basket_z.max(dim=1).values
    low_enough = obj_z < basket_top_z
    high_enough_for_basket = obj_z > basket_bottom_z
    obj_xy = self.target_object.data.root_pos_w[env_ids, :2]
    site_pos = self.target_site.data.root_pos_w[env_ids, :2]
    dist2 = ((obj_xy - site_pos) ** 2).sum(dim=-1)
    inside_site = dist2 < self.target_site_radius ** 2
    basket_displacement = torch.norm(self.target_site.data.root_pos_w[env_ids] - self.basket_init_pos[env_ids], dim=-1)
    basket_untouched = basket_displacement < self.cfg.basket_drift_threshold
    collision_displacement = torch.norm(self.collision_obj_pos[env_ids] - self.collision_object_init_pos[env_ids], dim=-1)
    collision_not_too_hard = collision_displacement <= self.cfg.collision_max_displacement
    success = self.collision_triggered[env_ids] & collision_not_too_hard & inside_site & low_enough & high_enough_for_basket & basket_untouched
    extras['Eureka/success_metric'] = success.float().mean()"""
        ),
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.05,
    },

    "TestSlightCollide": {
        "description": """**Objective:**
Stage 1 of a two-stage curriculum. Grasp the target object (alphabet_soup) and make a *gentle* collision with the tomato_sauce can. The episode terminates as soon as the tomato_sauce is displaced between 5 cm and 20 cm from its initial position. Stage 2 (placing in basket) is handled by TestCollideAndPlace.

## Task sequence

### 1) Approach and grasp
- Move the end-effector to the target object and grasp it.

### 2) Light collision with tomato_sauce
- Move the grasped object so it contacts and gently displaces the tomato_sauce.
- `self.collision_triggered` becomes True once displacement ≥ 5 cm.
- `self.cfg.collision_max_displacement` = 0.2 m (upper bound for "gentle").

## Key attributes
- `self.robot_grasp_pos`: TCP world position [num_envs, 3]
- `self.target_object.data.root_pos_w`: alphabet_soup world pos [num_envs, 3]
- `self.collision_obj_pos`: tomato_sauce world pos [num_envs, 3]
- `self.collision_object_init_pos`: tomato_sauce spawn pos [num_envs, 3]
- `self.collision_triggered`: bool [num_envs], True once displacement ≥ 5 cm
- `self.cfg.collision_max_displacement`: 0.2 m

Add regularisation on joint speed and action rate for smooth motion.
""",
        "success_metric": (
            """collision_displacement = torch.norm(self.collision_obj_pos[env_ids] - self.collision_object_init_pos[env_ids], dim=-1)
    collision_not_too_hard = collision_displacement <= self.cfg.collision_max_displacement
    collision_obj_vel = torch.norm(self.collision_object.data.root_lin_vel_w[env_ids], dim=-1)
    collision_obj_stopped = collision_obj_vel < self.cfg.collision_stop_vel_threshold
    success = self.collision_triggered[env_ids] & collision_not_too_hard & collision_obj_stopped
    extras['Eureka/success_metric'] = success.float().mean()"""
        ),
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.05,
    },

    # This one is for curriculum learning.
    "TestPickItUp": {
        "description": """This is a curriculum learning task where the policy is already able to hover over the basket with object grasp. Now, you only formulate the last stage reward to slightly change the policy: let the object be correctly dropped after the condition in the last stage is met. You can gate the other stages and make the policy untouched or something
        """,
        "consider_stage_in_success_metric": False,
        "success_metric": (
         """low_enough = self.target_object.data.root_pos_w[env_ids, 2] <self.target_site_corners_world[1,2]
    obj_xy = self.target_object.data.root_pos_w[env_ids, :2]
    site_pos = self.target_site.data.root_pos_w[env_ids, :2]
    dist2 = ((obj_xy - site_pos)**2).sum(dim=-1)
    inside_site = dist2 < self.target_site_radius**2

    success = inside_site & low_enough
    stage_masked = self.stage[env_ids].clone()
    stage_masked = stage_masked * (~success).unsqueeze(-1)
    for i in range(1, self.num_stages): # Log stages except for stage 0
        extras[f'Eureka/stage_{i}'] = (stage_masked[:,i]).float().mean()

    extras['Eureka/success_metric'] = (success).float().mean()
    """
        ),
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.05,
    },


    "PickItUp": {
        "description": """**Stage 1 of 2 — Pick up the object.**

**Initial condition:**
The robot gripper starts open, near its home pose. The target object is resting on the table.

**Objective:**
Grasp the target object cleanly from above without pushing it aside, then lift it until
the lowest point of the object is above the top rim of the basket. Keep the grasp stable
throughout the motion.

## Task sequence and constraints
### 1) Approach
- Move the end-effector to ~10 cm above the object, aligning with the grasp orientation.
- Keep the gripper open during approach to avoid collision.

### 2) Descend and grasp
- Descend slowly while keeping alignment.
- Close the gripper only when the object center is close to the TCP.

### 3) Lift
- Lift the grasped object until it clears the basket rim.
- Keep motion slow and stable. Avoid dropping.

Add regularisation on joint speed and action rate to ensure smooth motion.
""",
        "success_metric": (
            """grasped = self._grasp_detection(env_ids)
    lowest_z_object = self._get_target_object_lowest_points()[env_ids]
    high_enough = lowest_z_object > self.target_site_corners_world[1, 2] + 0.02
    terminated = high_enough & grasped
    extras['Eureka/success_metric'] = terminated.float().mean()"""
        ),
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.05,
    },

    "PickItUpNoGrasp": {
        "description": """**Stage 1 of 2 — Pick up the object (ablation: no grasp detector, no stage tracker).**

**Initial condition:**
The robot gripper starts open, near its home pose. The target object is resting on the table.

**Objective:**
Grasp the target object cleanly from above without pushing it aside, then lift it until
the lowest point of the object is above the top rim of the basket. The object must retain
its original upright orientation throughout (rotation error < 0.8 rad). Keep the grasp
stable throughout the motion.

**Success condition:** object lowest point above basket rim AND rotation error < 0.8 rad.
Pushing or tipping the object does NOT count as success.

## Available observations (no _grasp_detection or self.stage — those are removed)
- `self.robot_grasp_pos`: TCP position [num_envs, 3]
- `self.target_to_hand_pos`: vector from object center to TCP [num_envs, 3]
- `self.target_object.data.root_pos_w`: object world position [num_envs, 3]
- `self.target_object.data.root_quat_w`: object orientation [num_envs, 4]
- `self.to_desired_rot`: quaternion error to desired grasp orientation (reward w→1) [num_envs, 4]
- `self.target_site_corners_world`: basket min/max corners in env-local frame [2, 3]
- `self._robot.data.joint_pos`: joint positions [num_envs, 9]
- Finger contact forces (raw): `self.scene['left_contact_sensor'].data.force_matrix_w` and `self.scene['right_contact_sensor'].data.force_matrix_w`
- Finger joint widths: `self._robot.data.joint_pos[:, self.left_finger_joint_idx]` and `self._robot.data.joint_pos[:, self.right_finger_joint_idx]`

## Task sequence and constraints
### 1) Approach
- Move the end-effector to ~10 cm above the object, aligning with the grasp orientation.
- Keep the gripper open during approach to avoid collision.

### 2) Descend and grasp
- Descend slowly while keeping alignment.
- Close the gripper only when the object center is close to the TCP.

### 3) Lift
- Lift the grasped object until it clears the basket rim while keeping its orientation upright.
- Keep motion slow and stable.

Add regularisation on joint speed and action rate to ensure smooth motion.
Since there is no _grasp_detection helper, infer grasp from raw contact forces and finger width.
Add an orientation-keeping reward using `self.to_desired_rot` (reward when w component → 1).
""",
        "success_metric": (
            """lowest_z_object = self._get_target_object_lowest_points()[env_ids]
    high_enough = lowest_z_object > self.target_site_corners_world[1, 2] + 0.02
    success = high_enough & self.small_rotation[env_ids]
    extras['Eureka/success_metric'] = success.float().mean()"""
        ),
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.05,
    },

    "PickItUpNoGraspGamma99": {
        "description": """**Stage 1 of 2 — Pick up the object (ablation: no grasp detector, no stage tracker).**

**Initial condition:**
The robot gripper starts open, near its home pose. The target object is resting on the table.

**Objective:**
Grasp the target object cleanly from above without pushing it aside, then lift it until
the lowest point of the object is above the top rim of the basket. The object must retain
its original upright orientation throughout (rotation error < 0.8 rad). Keep the grasp
stable throughout the motion.

**Success condition:** object lowest point above basket rim AND rotation error < 0.8 rad.
Pushing or tipping the object does NOT count as success.

## Available observations (no _grasp_detection or self.stage — those are removed)
- `self.robot_grasp_pos`: TCP position [num_envs, 3]
- `self.target_to_hand_pos`: vector from object center to TCP [num_envs, 3]
- `self.target_object.data.root_pos_w`: object world position [num_envs, 3]
- `self.target_object.data.root_quat_w`: object orientation [num_envs, 4]
- `self.to_desired_rot`: quaternion error to desired grasp orientation (reward w→1) [num_envs, 4]
- `self.target_site_corners_world`: basket min/max corners in env-local frame [2, 3]
- `self._robot.data.joint_pos`: joint positions [num_envs, 9]
- Finger contact forces (raw): `self.scene['left_contact_sensor'].data.force_matrix_w` and `self.scene['right_contact_sensor'].data.force_matrix_w`
- Finger joint widths: `self._robot.data.joint_pos[:, self.left_finger_joint_idx]` and `self._robot.data.joint_pos[:, self.right_finger_joint_idx]`

## Task sequence and constraints
### 1) Approach
- Move the end-effector to ~10 cm above the object, aligning with the grasp orientation.
- Keep the gripper open during approach to avoid collision.

### 2) Descend and grasp
- Descend slowly while keeping alignment.
- Close the gripper only when the object center is close to the TCP.

### 3) Lift
- Lift the grasped object until it clears the basket rim while keeping its orientation upright.
- Keep motion slow and stable.

Add regularisation on joint speed and action rate to ensure smooth motion.
Since there is no _grasp_detection helper, infer grasp from raw contact forces and finger width.
Add an orientation-keeping reward using `self.to_desired_rot` (reward when w component → 1).
""",
        "success_metric": (
            """lowest_z_object = self._get_target_object_lowest_points()[env_ids]
    high_enough = lowest_z_object > self.target_site_corners_world[1, 2] + 0.02
    success = high_enough & self.small_rotation[env_ids]
    extras['Eureka/success_metric'] = success.float().mean()"""
        ),
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.05,
    },

    "PlaceInBasket": {
        "consider_stage_in_success_metric": False,
        "description": """**Stage 2 of 2 — Place the grasped object inside the basket.**

**Initial condition:**
The robot gripper already holds the target object with a stable grasp. The object starts
above the table, mid-trajectory. The basket (target receptacle) pose is known.

**Objective:**
Move the grasped object above the basket opening, then release it so it falls inside.

## Task sequence and constraints
### 1) Transport
- Move the object horizontally toward the basket while keeping the grasp.
- Keep the motion smooth and controlled.

### 2) Position above basket
- Align the object with the basket opening.
- Object center-of-mass should be directly above the basket.

### 3) Release
- Open the gripper when aligned above the basket.
- The object should fall and land inside (low enough and within the basket radius).

Add regularisation on joint speed and action rate. Use `self.helper_variable` to
track task progress if needed.
""",
        "success_metric": (
            """low_enough = self.target_object.data.root_pos_w[env_ids, 2] < self.target_site_corners_world[1, 2]
    obj_xy = self.target_object.data.root_pos_w[env_ids, :2]
    site_pos = self.target_site.data.root_pos_w[env_ids, :2]
    dist2 = ((obj_xy - site_pos) ** 2).sum(dim=-1)
    inside_site = dist2 < self.target_site_radius ** 2
    success = inside_site & low_enough
    stage_masked = self.stage[env_ids].clone()
    stage_masked = stage_masked * (~success).unsqueeze(-1)
    for i in range(1, self.num_stages):
        extras[f'Eureka/stage_{i}'] = stage_masked[:, i].float().mean()
    extras['Eureka/success_metric'] = success.float().mean()"""
        ),
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.05,
    },

    "TestPutItInTheBasket": {
        "consider_stage_in_success_metric": False,
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

    success = inside_site & low_enough
    stage_masked = self.stage[env_ids].clone()
    stage_masked = stage_masked * (~success).unsqueeze(-1)
    for i in range(1, self.num_stages): # Log stages except for stage 0
        extras[f'Eureka/stage_{i}'] = (stage_masked[:,i]).float().mean()

    extras['Eureka/success_metric'] = success.float().mean()"""
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
        "consider_stage_in_success_metric": False,
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
