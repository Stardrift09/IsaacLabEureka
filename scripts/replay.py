
import torch


if __name__ == "__main__":
    """Create the environment for the task."""
    task = "LivingRoomScene1PickUpTheAlphabetSoupAndPutItInTheBasket"
    from isaaclab.app import AppLauncher
    from isaaclab_eureka.utils import MuteOutput, get_freest_gpu
    device = "cuda"
    if device == "cuda":
        device_id = get_freest_gpu()
        device = f"cuda:{device_id}"
    app_launcher = AppLauncher(headless=False, device=device)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab_tasks.utils import parse_env_cfg
    env_cfg: DirectRLEnvCfg = parse_env_cfg(task)
    env_cfg.sim.device = device
    env = gym.make(task, cfg=env_cfg)
    env = env.unwrapped
    env.reset()
    env.run_replay()
    print("finished replaying, exiting")
    env.close()

    print("env deleted")
    torch.cuda.empty_cache()

    print("empty cache")
    env.sim.stop()
    env.sim.clear()
    simulation_app.close()