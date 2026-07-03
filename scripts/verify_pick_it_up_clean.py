"""Verify the clean PickItUp env: MRO, scene filter, and single-flag init behavior.

Run in conda eureka:
    ~/miniconda3/envs/eureka/bin/python -u scripts/verify_pick_it_up_clean.py --headless
"""
import argparse

from isaaclab_eureka.utils import get_freest_gpu


def main(args_cli):
    from isaaclab.app import AppLauncher

    device = f"cuda:{get_freest_gpu()}"
    app = AppLauncher(headless=True, enable_cameras=True, device=device).app

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    from isaaclab_tasks.direct.franka_cabinet.pick_it_up_clean import (
        EvalPickItUpClean, OodEvalPickItUpClean, PickItUpClean,
    )

    ok = True

    # ---- 1. MRO + method resolution ----
    for cls in (EvalPickItUpClean, OodEvalPickItUpClean):
        mro = [c.__name__ for c in cls.__mro__]
        print(f"[MRO] {cls.__name__}: {mro[:5]}")
        assert mro.index("ReplayPickItUpOneEnv") < mro.index("PickItUpClean") < mro.index("TestPickItUp"), \
            f"bad MRO for {cls.__name__}"
    print("[reset_idx] Eval ->", EvalPickItUpClean._reset_idx.__qualname__.split(".")[0],
          "| Ood ->", OodEvalPickItUpClean._reset_idx.__qualname__.split(".")[0])
    assert EvalPickItUpClean._reset_idx.__qualname__.startswith("PickItUpClean")
    assert OodEvalPickItUpClean._reset_idx.__qualname__.startswith("OodEvalPickItUpOneEnv")
    print("[setup_scene] Eval ->", EvalPickItUpClean._setup_scene.__qualname__.split(".")[0])
    assert EvalPickItUpClean._setup_scene.__qualname__.startswith("ReplayPickItUpOneEnv")
    print("  MRO/resolution OK\n")

    # ---- 2. instantiate EvalPickItUpClean, check scene filter ----
    task = "EvalPickItUpClean"
    env_cfg = parse_env_cfg(task, device=device, num_envs=4)
    env_cfg.sim.device = device
    env = gym.make(task, cfg=env_cfg).unwrapped
    names = sorted(env.rigid_objects.keys())
    print(f"[scene] rigid_objects = {names} (target={env.target_object_name}, site={env.target_site_name})")
    print(f"[scene] object_names = {env.object_names}")
    assert len(env.rigid_objects) == 2, f"expected 2 objects, got {len(env.rigid_objects)}"
    assert set(names) == {env.target_object_name, env.target_site_name}
    assert sorted(env.object_names) == names
    print("  scene filter OK (distractors removed)\n")

    # ---- 2a2. actuator gains promoted to cfg + new experiment folder ----
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    runner = load_cfg_from_registry("PickItUpClean", "rsl_rl_cfg_entry_point")
    print(f"[runner] PickItUpClean experiment_name = {runner.experiment_name}")
    assert runner.experiment_name == "pick_it_up_clean", "experiment folder not parallel"
    g = (env.cfg.arm_shoulder_stiffness, env.cfg.arm_forearm_stiffness, env.cfg.hand_stiffness,
         env.cfg.wrist_rot_noise_std)
    print(f"[gains] shoulder/forearm/hand stiffness + wrist_noise = {g}")
    assert g == (600.0, 500.0, 2000.0, 0.15), "gain/wrist cfg defaults changed unexpectedly"

    # ---- 2b. CCD enabled + curriculum ramp ----
    ccd = getattr(getattr(env.cfg.sim, "physx", None), "enable_ccd", False)
    print(f"[ccd] sim.physx.enable_ccd = {ccd}")
    assert ccd, "CCD not enabled"
    env.cfg.curriculum_enabled = True
    env.cfg.curriculum_ring_start_diam = 0.0
    env.cfg.curriculum_ring_end_diam = 0.16
    env.cfg.curriculum_steps = 12000
    env.common_step_counter = 0
    d0 = env._current_ring_outer_diam()
    env.common_step_counter = 6000
    d_half = env._current_ring_outer_diam()
    env.common_step_counter = 99999
    d_full = env._current_ring_outer_diam()
    print(f"[curriculum] outer_diam @step 0/6000/full = {d0:.3f}/{d_half:.3f}/{d_full:.3f}")
    assert abs(d0) < 1e-6 and abs(d_half - 0.08) < 1e-3 and abs(d_full - 0.16) < 1e-3, "curriculum ramp wrong"
    env.cfg.curriculum_enabled = False  # restore for the ring-bound test below

    all_ids = torch.arange(env.num_envs, device=device, dtype=torch.long)

    def snapshot():
        return (
            env._robot.data.joint_pos.clone(),
            env.target_object.data.root_pos_w.clone(),
            env.target_site.data.root_pos_w.clone(),
        )

    # ---- 3a. randomize_init=False -> two resets identical (deterministic) ----
    env.randomize_init = False
    env._reset_idx(all_ids); a = snapshot()
    env._reset_idx(all_ids); b = snapshot()
    det = all(torch.allclose(x, y, atol=1e-6) for x, y in zip(a, b))
    print(f"[flag] randomize_init=False -> resets identical: {det}")
    assert det, "deterministic reset expected but poses differ"

    # ---- 3b. randomize_init=True -> resets differ ----
    env.randomize_init = True
    env._reset_idx(all_ids); c = snapshot()
    env._reset_idx(all_ids); d = snapshot()
    rnd = any(not torch.allclose(x, y, atol=1e-6) for x, y in zip(c, d))
    print(f"[flag] randomize_init=True  -> resets differ:    {rnd}")
    assert rnd, "randomized reset expected but poses identical"

    # ---- 3c. uniform ring: object XY displacement from nominal within [0, outer_r] ----
    env.randomize_init = True
    env.cfg.init_ring_inner_diam = 0.0
    env.cfg.init_ring_outer_diam = 0.16
    outer_r = 0.5 * env.cfg.init_ring_outer_diam
    nom_xy = env._nominal_obj_xyz[:2].to(device)
    max_disp = 0.0
    for _ in range(50):
        env._reset_idx(all_ids)
        disp = (env.target_object.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2] - nom_xy).norm(dim=1)
        max_disp = max(max_disp, disp.max().item())
        assert disp.max().item() <= outer_r + 1e-3, f"object outside ring: {disp.max().item()} > {outer_r}"
    print(f"[ring] outer_r={outer_r:.3f}  observed max object disp over 50 resets={max_disp:.3f}  (<= outer_r)")
    assert max_disp > 0.5 * outer_r, "ring barely used -- sampling looks wrong"

    # ---- 3d. randomize_rotation toggles without error ----
    env.cfg.randomize_rotation = True
    env._reset_idx(all_ids)
    print("[flag] randomize_rotation toggled without error")

    print("\nALL CHECKS PASSED" if ok else "\nFAILED")
    env.close()
    app.close()


if __name__ == "__main__":
    main(argparse.ArgumentParser().parse_args())
