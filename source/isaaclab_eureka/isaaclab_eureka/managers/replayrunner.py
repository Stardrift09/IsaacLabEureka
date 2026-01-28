from torch.utils.tensorboard import SummaryWriter
import torch


class ReplayRunner:
    def __init__(self, env, log_dir):
        self.env = env
        # self.path = self.env.path
        self.writer = SummaryWriter(log_dir)
        self.episode_idx = 0
        self.accumulated_reward = 0
        self.total_reward=0
        self.eureka_episode_sums = {}
        self.t=0
        self.device = "cuda"

        # self.env.common_step_counter = 0 # ensrue step counter to be zero

    def step(self,actions):
        # TODO: avoid using step, call the methods direc
        # tly
        # if self.env.common_step_counter == 0:
        #     self.env.reset_idx(env_ids=None)

        # self.env._apply_action()
        # self.env._get_rewards()

        obs, rewards, terminated, truncated, info = self.env.step(actions)
        rewards_eureka, rewards_dict = self.env._get_rewards_eureka()

        # self.eureka_episode_sums["eureka_total_rewards"] += rewards_eureka
        # eureka_episode_sums["oracle_total_rewards"] += rewards_oracle_replay
        for key in rewards_dict.keys():
            if key not in self.eureka_episode_sums:
                self.eureka_episode_sums[key] = torch.zeros(self.env.num_envs, device=self.device)
            self.eureka_episode_sums[key] += rewards_dict[key]
        for k in self.eureka_episode_sums.keys():
            self.writer.add_scalar("Replay/"+k, self.eureka_episode_sums[k].mean().item(), self.t) 

        self.t +=1
        return obs, rewards, terminated, truncated, info

if __name__ == "__main__":
    """Create the environment for the task."""
    debug = True
    task = "ReplayLivingRoomScene1PickUpTheAlphabetSoupAndPutItInTheBasket"
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
        _, _, _, _, _ = replayrunner.step(env.actions)
        if env.common_step_counter >= env.max_len - 1: # only for envs with...
            break
    print("finished replaying, exiting")
    env.close()

    del replayrunner
    print("env deleted")
    torch.cuda.empty_cache()

    print("empty cache")
    env.sim.stop()
    env.sim.clear()
    simulation_app.close()