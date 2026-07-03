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

    "OpenTheMicrowave": {
        "description": """**Objective:** Open the microwave by pulling the door handle.

The Franka arm must approach the microwave door handle and apply force to rotate the
microjoint from 0 to < -1.222 radians (70 degrees open; fully open is -2.094 rad / ~120 degrees).

## Key attributes
- `self.robot_grasp_pos`: TCP world position [num_envs, 3]
- `self.hand_to_handle_pos`: vector from TCP to door handle center [num_envs, 3]
- `self.handle_pos_w`: handle world position [num_envs, 3]
- `self._microwave.data.joint_pos[:, self.door_joint_idx]`: current door angle [num_envs] (0=closed, negative=open)
- `self.cfg.door_success_threshold`: -1.222 rad (70 degrees)

## Reward hints
1. Approach: reward decreasing distance between TCP and handle.
2. Pull: reward decreasing (more negative) door angle.
3. Add action regularization for smooth motion.
""",
        "success_metric": (
            """door_angle = self._microwave.data.joint_pos[env_ids, self.door_joint_idx]
    extras['Eureka/success_metric'] = (door_angle < self.cfg.door_success_threshold).float().mean()"""
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


    "TestPickItUp": {
        "description": """Pick up the target object (alphabet_soup) and drop it inside the basket. This is a multi-stage, long-horizon task.

## Task stages (reflected in self.stage one-hot [num_envs, 5])
- Stage 0: default — EEF not yet positioned above object
- Stage 1: pregrasp — EEF is close in XY and positioned above the object
- Stage 2: grasped — stable bilateral contact detected, EEF close to object
- Stage 3: lifted — object grasped AND lifted above basket rim
- Stage 4: over basket — object inside basket XY footprint (ready to release)
- Success: object inside basket volume (low_enough & high_enough_for_basket & inside_site)

## Key attributes
- `self.robot_grasp_pos`: EEF TCP world pos [num_envs, 3]
- `self.robot_grasp_rot`: EEF TCP world quat [num_envs, 4]
- `self.to_desired_rot`: quat from current→desired grasp orientation [num_envs, 4]; reward w→1 for correct approach angle
- `self.target_object.data.root_pos_w`: object center world pos [num_envs, 3]
- `self.target_object.data.root_quat_w`: object world quat [num_envs, 4]
- `self.corners_target_obj`: object 8 corners in world frame [num_envs, 8, 3]
- `self.target_site.data.root_pos_w`: basket center world pos [num_envs, 3]
- `self.target_site_radius`: basket XY acceptance radius (scalar)
- `self.basket_corners_world`: basket 8 corners in world frame [num_envs, 8, 3]
- `self.grasped`: bool [num_envs], True if object is stably grasped this step
- `self.high_enough`: bool [num_envs], True if object lifted above basket rim
- `self.stage`: one-hot stage [num_envs, 5] — use for stage-conditioned rewards
- `self.site_to_target_pos`: basket_center − object_center [num_envs, 3]

Design multi-stage rewards following the stage structure above. Use `self.stage` to condition rewards on the current stage. The reward must be monotonically improving across stages (see reward formatting instructions).
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
    dist2 = ((obj_xy - site_pos)**2).sum(dim=-1)
    inside_site = dist2 < self.target_site_radius**2

    success = inside_site & low_enough & high_enough_for_basket
    stage_masked = self.stage[env_ids].clone()
    stage_masked = stage_masked * (~success).unsqueeze(-1)
    for i in range(1, self.num_stages):  # Log stages 1-4; skip stage 0 (default)
        extras[f'Eureka/stage_{i}'] = stage_masked[:, i].float().mean()

    extras['Eureka/success_metric'] = success.float().mean()
    """
        ),
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.05,
        "consider_stage_in_success_metric": True,
    },

    "PickItUpCollideClean": {
        "description": """Starting with the target object (alphabet_soup) ALREADY GRASPED in the air, knock the tomato_sauce can by displacing it far enough. This is the ONLY goal — there is no basket placement.

## Initial condition
The episode STARTS with the object already grasped mid-air. The reset samples the initial robot joint state + held-object pose + basket pose from a file of recorded grasp-moment states, so the object is held from t=0 and the policy must keep holding it. There is NO approach/grasp-from-table sub-task. The tomato_sauce can sits on the table at a fixed nominal spawn.

## Objective (the only goal)
While keeping the object grasped, steer the held object so it CONTACTS and DISPLACES the tomato_sauce can by MORE than `self.cfg.collision_displacement_threshold` (0.1 m) from its initial position. The episode SUCCEEDS (terminates) when the can has moved past that threshold AND the object is still grasped (`self.grasped`) — the knock must be made with the held object, not by dropping or throwing it.

## Key attributes
- `self.robot_grasp_pos`: EEF TCP world pos [num_envs, 3]
- `self.robot_grasp_rot`: EEF TCP world quat [num_envs, 4]
- `self.target_object.data.root_pos_w`: held object center world pos [num_envs, 3]
- `self.target_object.data.root_quat_w`: held object world quat [num_envs, 4]
- `self.corners_target_obj`: held object 8 corners in world frame [num_envs, 8, 3]
- `self.collision_object.data.root_pos_w`: tomato_sauce can world pos [num_envs, 3]
- `self.collision_obj_pos`: tomato_sauce can world pos [num_envs, 3]
- `self.collision_object_init_pos`: tomato_sauce can initial (spawn) pos [num_envs, 3]
- `self.collision_triggered`: bool [num_envs], latched True once the can is displaced past the threshold
- `self.cfg.collision_displacement_threshold`: 0.1 m
- `self.grasped`: bool [num_envs], True if the object is stably grasped this step

Design rewards to: keep the object grasped (gripper closed), steer the held object toward the can (e.g. reward decreasing distance from the held object to `self.collision_obj_pos`), and strongly reward displacing the can past 0.1 m (reward `self.collision_triggered` becoming True, and/or the can's displacement from `self.collision_object_init_pos`). Add regularization on joint speed and action rate for smooth motion.
        """,
        "consider_stage_in_success_metric": False,
        "success_metric": (
            """success = self.collision_triggered[env_ids] & self.grasped[env_ids]
    extras['Eureka/success_metric'] = success.float().mean()"""
        ),
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.05,
    },

    "PlaceAfterCollideClean": {
        "description": """Starting with the target object (alphabet_soup) ALREADY GRASPED in the air AND MOVING (post-collision momentum), place it inside the basket. This is the place phase that follows the collide phase.

## Initial condition
The episode STARTS from a recorded POST-COLLISION state: the object is grasped, lifted off the table, and the arm + object carry velocity from the moment a prior collision was accomplished (the reset replays robot joint pos+vel and object/basket pose+vel from a file). There is NO grasp-from-table sub-task; keep the object held and bring it to the basket. The collision object (tomato_sauce) is NOT present in this phase.

## Objective
Carry the grasped, moving object over the basket and release it so it ends up inside the basket volume. Damp out the initial momentum smoothly, keep the grasp until over the basket, then lower/release.

## Key attributes
- `self.robot_grasp_pos`: EEF TCP world pos [num_envs, 3]
- `self.robot_grasp_rot`: EEF TCP world quat [num_envs, 4]
- `self.to_desired_rot`: quat from current→desired grasp orientation [num_envs, 4]; reward w→1 for correct orientation
- `self.target_object.data.root_pos_w`: object center world pos [num_envs, 3]
- `self.target_object.data.root_quat_w`: object world quat [num_envs, 4]
- `self.target_object.data.root_lin_vel_w`: object linear velocity [num_envs, 3] (nonzero at reset)
- `self.corners_target_obj`: object 8 corners in world frame [num_envs, 8, 3]
- `self.target_site.data.root_pos_w`: basket center world pos [num_envs, 3]
- `self.target_site_radius`: basket XY acceptance radius (scalar)
- `self.basket_corners_world`: basket 8 corners in world frame [num_envs, 8, 3]
- `self.grasped`: bool [num_envs], True if object is stably grasped this step
- `self.high_enough`: bool [num_envs], True if object lifted above basket rim
- `self.stage`: one-hot stage [num_envs, 5] — use for stage-conditioned rewards

Design rewards to: settle the initial velocity, carry the held object over the basket, and place/release it inside. Reward decreasing object→basket distance and the in-basket condition; add regularization on joint speed and action rate.
        """,
        "consider_stage_in_success_metric": False,
        "success_metric": (
            """obj_z = self.target_object.data.root_pos_w[env_ids, 2]
    basket_z = self.basket_corners_world[env_ids, :, 2]
    basket_bottom_z = basket_z.min(dim=1).values
    basket_top_z = basket_z.max(dim=1).values
    low_enough = obj_z < basket_top_z
    high_enough_for_basket = obj_z > basket_bottom_z
    obj_xy = self.target_object.data.root_pos_w[env_ids, :2]
    site_pos = self.target_site.data.root_pos_w[env_ids, :2]
    dist2 = ((obj_xy - site_pos)**2).sum(dim=-1)
    inside_site = dist2 < self.target_site_radius**2
    success = inside_site & low_enough & high_enough_for_basket
    extras['Eureka/success_metric'] = success.float().mean()"""
        ),
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.05,
    },

    "TestPickItUpKetchup": {
        "description": """Pick up the target object (ketchup) and drop it inside the basket. This is a multi-stage, long-horizon task.

## Task stages (reflected in self.stage one-hot [num_envs, 5])
- Stage 0: default — EEF not yet positioned above object
- Stage 1: pregrasp — EEF is close in XY and positioned above the object
- Stage 2: grasped — stable bilateral contact detected, EEF close to object
- Stage 3: lifted — object grasped AND lifted above basket rim
- Stage 4: over basket — object inside basket XY footprint (ready to release)
- Success: object inside basket volume (low_enough & high_enough_for_basket & inside_site)

## Key attributes
- `self.robot_grasp_pos`: EEF TCP world pos [num_envs, 3]
- `self.robot_grasp_rot`: EEF TCP world quat [num_envs, 4]
- `self.to_desired_rot`: quat from current→desired grasp orientation [num_envs, 4]; reward w→1 for correct approach angle
- `self.target_object.data.root_pos_w`: object center world pos [num_envs, 3]
- `self.target_object.data.root_quat_w`: object world quat [num_envs, 4]
- `self.corners_target_obj`: object 8 corners in world frame [num_envs, 8, 3]
- `self.target_site.data.root_pos_w`: basket center world pos [num_envs, 3]
- `self.target_site_radius`: basket XY acceptance radius (scalar)
- `self.basket_corners_world`: basket 8 corners in world frame [num_envs, 8, 3]
- `self.grasped`: bool [num_envs], True if object is stably grasped this step
- `self.high_enough`: bool [num_envs], True if object lifted above basket rim
- `self.stage`: one-hot stage [num_envs, 5] — use for stage-conditioned rewards
- `self.site_to_target_pos`: basket_center − object_center [num_envs, 3]

Design multi-stage rewards following the stage structure above. Use `self.stage` to condition rewards on the current stage. The reward must be monotonically improving across stages (see reward formatting instructions).
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
    dist2 = ((obj_xy - site_pos)**2).sum(dim=-1)
    inside_site = dist2 < self.target_site_radius**2

    success = inside_site & low_enough & high_enough_for_basket
    stage_masked = self.stage[env_ids].clone()
    stage_masked = stage_masked * (~success).unsqueeze(-1)
    for i in range(1, self.num_stages):  # Log stages 1-4; skip stage 0 (default)
        extras[f'Eureka/stage_{i}'] = stage_masked[:, i].float().mean()

    extras['Eureka/success_metric'] = success.float().mean()
    """
        ),
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.05,
        "consider_stage_in_success_metric": True,
    },

    "TestPickItUpCreamCheese": {
        "description": """Pick up the target object (cream_cheese) and drop it inside the basket. This is a multi-stage, long-horizon task.

## Task stages (reflected in self.stage one-hot [num_envs, 5])
- Stage 0: default — EEF not yet positioned above object
- Stage 1: pregrasp — EEF is close in XY and positioned above the object
- Stage 2: grasped — stable bilateral contact detected, EEF close to object
- Stage 3: lifted — object grasped AND lifted above basket rim
- Stage 4: over basket — object inside basket XY footprint (ready to release)
- Success: object inside basket volume (low_enough & high_enough_for_basket & inside_site)

## Key attributes
- `self.robot_grasp_pos`: EEF TCP world pos [num_envs, 3]
- `self.robot_grasp_rot`: EEF TCP world quat [num_envs, 4]
- `self.to_desired_rot`: quat from current→desired grasp orientation [num_envs, 4]; reward w→1 for correct approach angle
- `self.target_object.data.root_pos_w`: object center world pos [num_envs, 3]
- `self.target_object.data.root_quat_w`: object world quat [num_envs, 4]
- `self.corners_target_obj`: object 8 corners in world frame [num_envs, 8, 3]
- `self.target_site.data.root_pos_w`: basket center world pos [num_envs, 3]
- `self.target_site_radius`: basket XY acceptance radius (scalar)
- `self.basket_corners_world`: basket 8 corners in world frame [num_envs, 8, 3]
- `self.grasped`: bool [num_envs], True if object is stably grasped this step
- `self.high_enough`: bool [num_envs], True if object lifted above basket rim
- `self.stage`: one-hot stage [num_envs, 5] — use for stage-conditioned rewards
- `self.site_to_target_pos`: basket_center − object_center [num_envs, 3]

Design multi-stage rewards following the stage structure above. Use `self.stage` to condition rewards on the current stage. The reward must be monotonically improving across stages (see reward formatting instructions).
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
    dist2 = ((obj_xy - site_pos)**2).sum(dim=-1)
    inside_site = dist2 < self.target_site_radius**2

    success = inside_site & low_enough & high_enough_for_basket
    stage_masked = self.stage[env_ids].clone()
    stage_masked = stage_masked * (~success).unsqueeze(-1)
    for i in range(1, self.num_stages):  # Log stages 1-4; skip stage 0 (default)
        extras[f'Eureka/stage_{i}'] = stage_masked[:, i].float().mean()

    extras['Eureka/success_metric'] = success.float().mean()
    """
        ),
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.05,
        "consider_stage_in_success_metric": True,
    },

    "TestPickItUpTomatoSauce": {
        "description": """Pick up the target object (tomato_sauce) and drop it inside the basket. This is a multi-stage, long-horizon task.

## Task stages (reflected in self.stage one-hot [num_envs, 5])
- Stage 0: default — EEF not yet positioned above object
- Stage 1: pregrasp — EEF is close in XY and positioned above the object
- Stage 2: grasped — stable bilateral contact detected, EEF close to object
- Stage 3: lifted — object grasped AND lifted above basket rim
- Stage 4: over basket — object inside basket XY footprint (ready to release)
- Success: object inside basket volume (low_enough & high_enough_for_basket & inside_site)

## Key attributes
- `self.robot_grasp_pos`: EEF TCP world pos [num_envs, 3]
- `self.robot_grasp_rot`: EEF TCP world quat [num_envs, 4]
- `self.to_desired_rot`: quat from current→desired grasp orientation [num_envs, 4]; reward w→1 for correct approach angle
- `self.target_object.data.root_pos_w`: object center world pos [num_envs, 3]
- `self.target_object.data.root_quat_w`: object world quat [num_envs, 4]
- `self.corners_target_obj`: object 8 corners in world frame [num_envs, 8, 3]
- `self.target_site.data.root_pos_w`: basket center world pos [num_envs, 3]
- `self.target_site_radius`: basket XY acceptance radius (scalar)
- `self.basket_corners_world`: basket 8 corners in world frame [num_envs, 8, 3]
- `self.grasped`: bool [num_envs], True if object is stably grasped this step
- `self.high_enough`: bool [num_envs], True if object lifted above basket rim
- `self.stage`: one-hot stage [num_envs, 5] — use for stage-conditioned rewards
- `self.site_to_target_pos`: basket_center − object_center [num_envs, 3]

Design multi-stage rewards following the stage structure above. Use `self.stage` to condition rewards on the current stage. The reward must be monotonically improving across stages (see reward formatting instructions).
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
    dist2 = ((obj_xy - site_pos)**2).sum(dim=-1)
    inside_site = dist2 < self.target_site_radius**2

    success = inside_site & low_enough & high_enough_for_basket
    stage_masked = self.stage[env_ids].clone()
    stage_masked = stage_masked * (~success).unsqueeze(-1)
    for i in range(1, self.num_stages):  # Log stages 1-4; skip stage 0 (default)
        extras[f'Eureka/stage_{i}'] = stage_masked[:, i].float().mean()

    extras['Eureka/success_metric'] = success.float().mean()
    """
        ),
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.05,
        "consider_stage_in_success_metric": True,
    },


    "TestPickItUpMilk": {
        "description": """Pick up the target object (milk) and drop it inside the basket. This is a multi-stage, long-horizon task.

## Task stages (reflected in self.stage one-hot [num_envs, 5])
- Stage 0: default — EEF not yet positioned above object
- Stage 1: pregrasp — EEF is close in XY and positioned above the object
- Stage 2: grasped — stable bilateral contact detected, EEF close to object
- Stage 3: lifted — object grasped AND lifted above basket rim
- Stage 4: over basket — object inside basket XY footprint (ready to release)
- Success: object inside basket volume (low_enough & high_enough_for_basket & inside_site)

## Key attributes
- `self.robot_grasp_pos`: EEF TCP world pos [num_envs, 3]
- `self.robot_grasp_rot`: EEF TCP world quat [num_envs, 4]
- `self.to_desired_rot`: quat from current→desired grasp orientation [num_envs, 4]; reward w→1 for correct approach angle
- `self.target_object.data.root_pos_w`: object center world pos [num_envs, 3]
- `self.target_object.data.root_quat_w`: object world quat [num_envs, 4]
- `self.corners_target_obj`: object 8 corners in world frame [num_envs, 8, 3]
- `self.target_site.data.root_pos_w`: basket center world pos [num_envs, 3]
- `self.target_site_radius`: basket XY acceptance radius (scalar)
- `self.basket_corners_world`: basket 8 corners in world frame [num_envs, 8, 3]
- `self.grasped`: bool [num_envs], True if object is stably grasped this step
- `self.high_enough`: bool [num_envs], True if object lifted above basket rim
- `self.stage`: one-hot stage [num_envs, 5] — use for stage-conditioned rewards
- `self.site_to_target_pos`: basket_center − object_center [num_envs, 3]

Design multi-stage rewards following the stage structure above. Use `self.stage` to condition rewards on the current stage. The reward must be monotonically improving across stages (see reward formatting instructions).
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
    dist2 = ((obj_xy - site_pos)**2).sum(dim=-1)
    inside_site = dist2 < self.target_site_radius**2

    success = inside_site & low_enough & high_enough_for_basket
    stage_masked = self.stage[env_ids].clone()
    stage_masked = stage_masked * (~success).unsqueeze(-1)
    for i in range(1, self.num_stages):  # Log stages 1-4; skip stage 0 (default)
        extras[f'Eureka/stage_{i}'] = stage_masked[:, i].float().mean()

    extras['Eureka/success_metric'] = success.float().mean()
    """
        ),
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.05,
        "consider_stage_in_success_metric": True,
    },



    "TestPickItUpOrangeJuice": {
        "description": """Pick up the target object (orange_juice) and drop it inside the basket. This is a multi-stage, long-horizon task.

## Task stages (reflected in self.stage one-hot [num_envs, 5])
- Stage 0: default — EEF not yet positioned above object
- Stage 1: pregrasp — EEF is close in XY and positioned above the object
- Stage 2: grasped — stable bilateral contact detected, EEF close to object
- Stage 3: lifted — object grasped AND lifted above basket rim
- Stage 4: over basket — object inside basket XY footprint (ready to release)
- Success: object inside basket volume (low_enough & high_enough_for_basket & inside_site)

## Key attributes
- `self.robot_grasp_pos`: EEF TCP world pos [num_envs, 3]
- `self.robot_grasp_rot`: EEF TCP world quat [num_envs, 4]
- `self.to_desired_rot`: quat from current→desired grasp orientation [num_envs, 4]; reward w→1 for correct approach angle
- `self.target_object.data.root_pos_w`: object center world pos [num_envs, 3]
- `self.target_object.data.root_quat_w`: object world quat [num_envs, 4]
- `self.corners_target_obj`: object 8 corners in world frame [num_envs, 8, 3]
- `self.target_site.data.root_pos_w`: basket center world pos [num_envs, 3]
- `self.target_site_radius`: basket XY acceptance radius (scalar)
- `self.basket_corners_world`: basket 8 corners in world frame [num_envs, 8, 3]
- `self.grasped`: bool [num_envs], True if object is stably grasped this step
- `self.high_enough`: bool [num_envs], True if object lifted above basket rim
- `self.stage`: one-hot stage [num_envs, 5] — use for stage-conditioned rewards
- `self.site_to_target_pos`: basket_center − object_center [num_envs, 3]

Design multi-stage rewards following the stage structure above. Use `self.stage` to condition rewards on the current stage. The reward must be monotonically improving across stages (see reward formatting instructions).
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
    dist2 = ((obj_xy - site_pos)**2).sum(dim=-1)
    inside_site = dist2 < self.target_site_radius**2

    success = inside_site & low_enough & high_enough_for_basket
    stage_masked = self.stage[env_ids].clone()
    stage_masked = stage_masked * (~success).unsqueeze(-1)
    for i in range(1, self.num_stages):  # Log stages 1-4; skip stage 0 (default)
        extras[f'Eureka/stage_{i}'] = stage_masked[:, i].float().mean()

    extras['Eureka/success_metric'] = success.float().mean()
    """
        ),
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.05,
        "consider_stage_in_success_metric": True,
    },



    "TestPickItUpButter": {
        "description": """Pick up the target object (butter) and drop it inside the basket. This is a multi-stage, long-horizon task.

## Task stages (reflected in self.stage one-hot [num_envs, 5])
- Stage 0: default — EEF not yet positioned above object
- Stage 1: pregrasp — EEF is close in XY and positioned above the object
- Stage 2: grasped — stable bilateral contact detected, EEF close to object
- Stage 3: lifted — object grasped AND lifted above basket rim
- Stage 4: over basket — object inside basket XY footprint (ready to release)
- Success: object inside basket volume (low_enough & high_enough_for_basket & inside_site)

## Key attributes
- `self.robot_grasp_pos`: EEF TCP world pos [num_envs, 3]
- `self.robot_grasp_rot`: EEF TCP world quat [num_envs, 4]
- `self.to_desired_rot`: quat from current→desired grasp orientation [num_envs, 4]; reward w→1 for correct approach angle
- `self.target_object.data.root_pos_w`: object center world pos [num_envs, 3]
- `self.target_object.data.root_quat_w`: object world quat [num_envs, 4]
- `self.corners_target_obj`: object 8 corners in world frame [num_envs, 8, 3]
- `self.target_site.data.root_pos_w`: basket center world pos [num_envs, 3]
- `self.target_site_radius`: basket XY acceptance radius (scalar)
- `self.basket_corners_world`: basket 8 corners in world frame [num_envs, 8, 3]
- `self.grasped`: bool [num_envs], True if object is stably grasped this step
- `self.high_enough`: bool [num_envs], True if object lifted above basket rim
- `self.stage`: one-hot stage [num_envs, 5] — use for stage-conditioned rewards
- `self.site_to_target_pos`: basket_center − object_center [num_envs, 3]

Design multi-stage rewards following the stage structure above. Use `self.stage` to condition rewards on the current stage. The reward must be monotonically improving across stages (see reward formatting instructions).
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
    dist2 = ((obj_xy - site_pos)**2).sum(dim=-1)
    inside_site = dist2 < self.target_site_radius**2

    success = inside_site & low_enough & high_enough_for_basket
    stage_masked = self.stage[env_ids].clone()
    stage_masked = stage_masked * (~success).unsqueeze(-1)
    for i in range(1, self.num_stages):  # Log stages 1-4; skip stage 0 (default)
        extras[f'Eureka/stage_{i}'] = stage_masked[:, i].float().mean()

    extras['Eureka/success_metric'] = success.float().mean()
    """
        ),
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.05,
        "consider_stage_in_success_metric": True,
    },



    "PlaceInBasketNoGrasp": {
        "consider_stage_in_success_metric": False,
        "description": """**Stage 2 (no-grasp ablation) — Place the object inside the basket without grasp/stage detection.**

**Initial condition:**
The robot gripper already holds the target object mid-air. The basket (target receptacle) pose is known.

**Objective:**
Move the object above the basket opening and release it so it falls inside.

## Key attributes
- `self.robot_grasp_pos`: TCP world position [num_envs, 3]
- `self.target_object.data.root_pos_w`: object center in world [num_envs, 3]
- `self.target_site.data.root_pos_w`: basket bottom-center in world [num_envs, 3]
- `self.site_to_target_pos`: vector from basket site to object [num_envs, 3]
- `self.target_site_radius`: radius of placement zone (meters)
- `self.target_site_corners_world`: static [2, 3] — min/max corners of basket
- `self.basket_corners_world`: dynamic [num_envs, 8, 3] — 8 basket corners world frame
- `self.target_to_hand_pos`: vector from object center to TCP [num_envs, 3]
- Action last dim > 0 opens gripper

## Reward hints
1. Approach basket: reward decreasing `self.site_to_target_pos` magnitude.
2. Align: reward XY distance from object to basket center.
3. Release: reward opening gripper once aligned over basket.
4. Add regularization on joint speed and action rate.

NOTE: _grasp_detection(), _current_stage_detection(), and self.stage are NOT available.
Shape rewards using raw geometry only.
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
    success = inside_site & low_enough & high_enough_for_basket
    extras['Eureka/success_metric'] = success.float().mean()"""
        ),
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.05,
    },

    "PlaceInBasketDropNoGrasp": {
        "consider_stage_in_success_metric": False,
        "description": """**Stage 2 (drop + no-grasp ablation) — Object inside basket AND EEF outside exclusion zone at success.**

**Initial condition:**
The robot gripper already holds the target object mid-air. The basket (target receptacle) pose is known.

**Objective:**
Transport the object and release it so it lands inside the basket. The episode only terminates
with success when the object is inside the basket **and** the EEF is currently outside
the exclusion zone (`drop_eef_exclusion_factor * target_site_radius` from basket center).
This forces the robot to release from outside and not hover directly above.

## Task sequence and constraints
### 1) Transport
- Move toward the basket while keeping the object near the TCP.

### 2) Release outside basket
- Open the gripper **before** EEF XY is within the exclusion zone.
- Success only counts when EEF is outside the zone at the moment the object lands inside.

## Key attributes
- `self.robot_grasp_pos`: EEF TCP world pos [num_envs, 3]
- `self.target_site.data.root_pos_w`: basket center world pos [num_envs, 3]
- `self.target_site_radius`: basket XY clearance radius (scalar)
- `self.cfg.drop_eef_exclusion_factor`: 2.0 — exclusion zone = 2× target_site_radius
- `self.basket_corners_world`: basket 8 corners in world frame [num_envs, 8, 3]
- `self.site_to_target_pos`: vector from basket site to object [num_envs, 3]
- `self.target_to_hand_pos`: vector from object center to TCP [num_envs, 3]
- Action last dim > 0 opens gripper

NOTE: _grasp_detection(), _current_stage_detection(), and self.stage are NOT available.
Shape rewards using raw geometry only.
Add regularisation on joint speed and action rate.
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
    eef_xy = self.robot_grasp_pos[env_ids, :2]
    exclusion_radius = self.cfg.drop_eef_exclusion_factor * self.target_site_radius
    eef_outside = ((eef_xy - site_pos) ** 2).sum(dim=-1) >= exclusion_radius ** 2
    success = inside_site & low_enough & high_enough_for_basket & eef_outside
    extras['Eureka/success_metric'] = success.float().mean()"""
        ),
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.05,
    },

    "PlaceInBasketDrop": {
        "consider_stage_in_success_metric": False,
        "description": """**Stage 2 (drop variant) — Drop the grasped object into the basket from outside.**

**Initial condition:**
The robot gripper already holds the target object with a stable grasp. The object starts
above the table, mid-trajectory. The basket (target receptacle) pose is known.

**Objective:**
Transport the grasped object, then **release it while the end-effector is still outside
the basket XY footprint**, so the object flies/falls into the basket. Do NOT hover
directly above the basket and drop — the gripper must open before reaching the basket.

## Task sequence and constraints
### 1) Transport
- Move the grasped object toward the basket while maintaining the grasp.
- Keep motion smooth and controlled.

### 2) Release outside basket
- Open the gripper **before** the EEF XY is within `drop_eef_exclusion_factor * target_site_radius`
  of the basket center (exclusion zone = 3× the basket clearance radius — much larger than the basket).
- `self.drop_outside_triggered`: bool [num_envs], latched True once drop-outside-exclusion-zone event occurs.
- Reward `drop_outside_triggered` becoming True, and reward moving EEF away from basket before releasing.

### 3) Object lands inside
- After release, the object should fall/slide into the basket.
- Final success: object inside basket volume AND `self.drop_outside_triggered` is True.

## Key attributes
- `self.robot_grasp_pos`: EEF TCP world pos [num_envs, 3]
- `self.grasped`: bool [num_envs], True if object stably grasped this step
- `self.drop_outside_triggered`: bool [num_envs], latched True when drop happens outside exclusion zone
- `self.target_site.data.root_pos_w`: basket center world pos [num_envs, 3]
- `self.target_site_radius`: basket XY clearance radius (scalar, small — object fits inside basket)
- `self.cfg.drop_eef_exclusion_factor`: 3.0 — EEF exclusion zone = 3× target_site_radius
- `self.basket_corners_world`: basket 8 corners in world frame [num_envs, 8, 3]
- `self.inside_site`: bool [num_envs], True if object XY within basket radius
- `self.low_enough`: bool [num_envs], True if object below basket rim

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
    success = inside_site & low_enough & high_enough_for_basket & self.drop_outside_triggered[env_ids]
    extras['Eureka/success_metric'] = success.float().mean()"""
        ),
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.05,
    },

    "PlaceInBasketUpright": {
        "consider_stage_in_success_metric": False,
        "description": """**Stage 2 of 2 — Place the grasped object inside the basket while keeping it upright.**

**Initial condition:**
The robot gripper already holds the target object with a stable grasp. The object starts
above the table, mid-trajectory. The basket (target receptacle) pose is known.

**Objective:**
Move the grasped object above the basket opening, then release it so it falls inside
while remaining upright. The object's rotation error relative to its default orientation
must be ≤ 10 degrees (0.1745 rad) when it lands in the basket.

## Task sequence and constraints
### 1) Transport
- Move the object toward the basket while maintaining the grasp.
- Keep the motion smooth and controlled.

### 2) Position above basket
- Align the object center-of-mass directly above the basket opening.
- Keep the object upright (rotation error < 10°) throughout transport.

### 3) Release
- Open the gripper when aligned above the basket.
- The object should fall and land inside (low enough, within basket radius, and upright).

Add regularisation on joint speed and action rate. Reward keeping the object upright
using `self.to_desired_rot` (w component → 1 means correct orientation).
""",
        "success_metric": (
            """obj_z = self.target_object.data.root_pos_w[env_ids, 2]
    basket_z = self.basket_corners_world[env_ids, :, 2]
    basket_top_z = basket_z.max(dim=1).values
    low_enough = obj_z < basket_top_z
    obj_xy = self.target_object.data.root_pos_w[env_ids, :2]
    site_pos = self.target_site.data.root_pos_w[env_ids, :2]
    dist2 = ((obj_xy - site_pos) ** 2).sum(dim=-1)
    inside_site = dist2 < self.target_site_radius ** 2
    current_rot = self.target_object.data.root_quat_w[env_ids]
    desired_rot = self.target_object.data.default_root_state[env_ids, 3:7]
    current_rot_inv = quat_conjugate(current_rot)
    q_error = quat_mul(desired_rot, current_rot_inv)
    q_error = q_error / torch.norm(q_error, dim=-1, keepdim=True).clamp_min(1e-9)
    q_error = torch.where(q_error[:, 0:1] < 0, -q_error, q_error)
    angle_error = 2.0 * torch.acos(torch.clamp(q_error[:, 0], -1.0, 1.0))
    upright = angle_error < 0.17453
    success = inside_site & low_enough & upright
    extras['Eureka/success_metric'] = success.float().mean()"""
        ),
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.05,
    },

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
        "description": """**Initial condition:**
The episode STARTS with the target object (alphabet_soup) already grasped in the gripper, mid-air (start_in_the_air). There is NO approach/grasp sub-task — the object is held from t=0 and the policy must keep it held.

**Objective:**
While holding the object, lightly collide it with the tomato_sauce can, then place the target object inside the basket. The collision must be gentle: the tomato_sauce may be displaced at most 0.2 m from its initial position. The basket must remain untouched throughout.

## Task sequence and constraints

### 1) Keep the grasp
- The object starts grasped in the air. Keep the gripper closed and the object held until the release step.

### 2) Light collision with tomato_sauce
- Move the held object so it contacts and slightly displaces the tomato_sauce (≥ 5 cm, ≤ 20 cm from its initial position).
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

    # # This one is for curriculum learning.
    # "TestPickItUp": {
    #     "description": """This is a curriculum learning task where the policy is already able to hover over the basket with object grasp. Now, you only formulate the last stage reward to slightly change the policy: let the object be correctly dropped after the condition in the last stage is met. You can gate the other stages and make the policy untouched or something
    #     """,
    #     "consider_stage_in_success_metric": False,
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


    "TestPlaceCreamCheeseInDrawer": {
        "description": """**Objective:** Pick up the cream cheese and place it inside the pre-opened bottom drawer of the Sektion cabinet.

## Scene
- Franka Panda at env-local (1.0, 0, 0)
- Sektion cabinet at env-local (-0.2, 0, 0.4); bottom drawer held open at ~0.39 m
- Cream cheese spawns at env-local ≈ (1.0, ±0.4, 0) — to the side of the robot, not between robot and drawer

## Key attributes
- `self.robot_grasp_pos`: EEF TCP world pos [num_envs, 3]
- `self.robot_grasp_rot`: EEF TCP world quat [num_envs, 4]
- `self.to_desired_rot`: quat error to desired grasp orientation [num_envs, 4]; reward w→1
- `self.target_object.data.root_pos_w`: cream cheese world pos [num_envs, 3]
- `self.drawer_interior_pos_w`: drawer interior centre world pos [num_envs, 3]
- `self.corners_target_obj`: cream cheese 8 AABB corners in world frame [num_envs, 8, 3] (already maintained by the env each step; see tip 10)
- `self.target_object_size`: cream cheese AABB extents along x/y/z [num_envs, 3]
- `self.grasped`: bool [num_envs], True if cheese stably grasped
- `self.high_enough`: bool [num_envs], True if cheese lifted above drawer opening
- `self.inside_site`: bool [num_envs], True if cheese inside drawer volume
- `self.grasped_and_lifted`: bool [num_envs], latched True once grasped+lifted
- `self._cabinet.data.joint_pos[:, self.drawer_joint_idx]`: bottom drawer opening in meters [num_envs]; starts at ~0.39 (open), decreases if the drawer is pushed closed

Design a dense reward that rewards approaching, grasping, lifting, transporting, and placing the cheese inside the drawer. Add regularisation on joint speed and action rate for smooth motion. Multi-stage decomposition is optional, not required.

## Constraint
The drawer is a free (passive) joint, so the robot can accidentally push it closed.
Penalize closing the drawer from its initial open position **before** the cream cheese is inside
(i.e. while `~self.inside_site`). Once the cheese is inside, this penalty should not apply.
Measure displacement as the drop in `self._cabinet.data.joint_pos[:, self.drawer_joint_idx]`
below its initial open value (~0.39 m); use a one-sided penalty (only closing is penalised).
""",
        "success_metric": (
            """success = self.inside_site[env_ids] & self.grasped_and_lifted[env_ids]
    extras['Eureka/success_metric'] = success.float().mean()"""
        ),
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.05,
    },

    "OpenDrawerAndPutCreamCheese": {
        "description": """**Objective:** Open the top drawer of the Sektion Cabinet, then place the cream_cheese inside it.

## Task sequence
1. Approach the drawer handle and grasp it (gripper must close around handle).
2. Pull the drawer open until it is sufficiently extended.
3. Release the handle and approach the cream_cheese on the table.
4. Grasp the cream_cheese and transport it to the open drawer.
5. Place the cream_cheese inside the drawer interior.

## Key attributes
- `self.robot_grasp_pos`: TCP world position [num_envs, 3]
- `self.robot_grasp_rot`: TCP world quaternion [num_envs, 4]
- `self.drawer_grasp_pos`: drawer handle world position [num_envs, 3]
- `self.drawer_grasp_rot`: drawer handle world quaternion [num_envs, 4]
- `self.drawer_interior_pos`: target placement position inside the open drawer (world) [num_envs, 3]
- `self._cabinet.data.joint_pos[:, self.drawer_top_joint_idx]`: drawer opening in meters [num_envs]; 0 = closed, ~0.38 = fully open
- `self._cabinet.data.joint_vel[:, self.drawer_top_joint_idx]`: drawer velocity [num_envs]
- `self._cream_cheese.data.root_pos_w`: cream_cheese world position [num_envs, 3]
- `self.grasped_cheese`: bool [num_envs], True if cream_cheese is stably grasped
- `self.grasped_drawer`: bool [num_envs], True if drawer handle is stably grasped
- `self.cfg.drawer_open_threshold`: 0.30 m — drawer counts as open when joint_pos > this
- `self.cfg.drawer_placement_tolerance`: 0.15 m — cheese counts as inside when closer than this
- `self.gripper_forward_axis`, `self.drawer_inward_axis`, `self.gripper_up_axis`, `self.drawer_up_axis`: axis tensors for orientation reward [num_envs, 3]
- Action last dim > 0 opens gripper, < 0 closes gripper

Add regularisation on joint speed and action rate for smooth motion.
""",
        "success_metric": (
            """drawer_joint_pos = self._cabinet.data.joint_pos[env_ids, self.drawer_top_joint_idx]
    drawer_open = drawer_joint_pos > self.cfg.drawer_open_threshold
    cheese_pos = self._cream_cheese.data.root_pos_w[env_ids]
    d_cheese = torch.norm(cheese_pos - self.drawer_interior_pos[env_ids], p=2, dim=-1)
    cheese_inside = d_cheese < self.cfg.drawer_placement_tolerance
    success = drawer_open & cheese_inside
    extras['Eureka/success_metric'] = success.float().mean()"""
        ),
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.05,
    },

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
