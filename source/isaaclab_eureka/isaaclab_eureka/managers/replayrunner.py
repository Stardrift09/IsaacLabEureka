from torch.utils.tensorboard import SummaryWriter
import torch


class ReplayRunner:
    def __init__(self, env, log_dir):
        self.env = env
        # self.path = self.env.path
        self.writer = SummaryWriter(log_dir)
        self.episode_idx = 0
        self.accumulated_reward = 0

    def step(self,actions):

        obs, reward, terminated, truncated, info = self.env.step(actions)
        done = terminated | truncated
        # per-step logging
        self.writer.add_scalar("reward", reward.mean().item(), self.env.common_step_counter)

        self.accumulated_reward += reward.mean().item()
        self.writer.add_scalar("accumulated_reward", self.accumulated_reward, self.env.common_step_counter)


        return obs, reward, terminated, truncated, info
    

if __name__ == "__main__":
    """Create the environment for the task."""
    debug = True
    task = "ReplayLivingRoomScene1PickUpTheAlphabetSoupAndPutItInTheBasket-v0"
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
    replayrunner = ReplayRunner(env=env,log_dir="logs/replayrunner_test")
    if debug:
        print(env.actions)
    env.reset()
    while simulation_app.is_running():
        _, rewards, _, _, _ = replayrunner.step(env.actions)
        if env.common_step_counter >= env.max_len - 1: # only for envs with...
            print("finished replaying, exiting")
            env.close()
            simulation_app.close()
