
from time import time
import torch
from isaaclab_eureka.utils import eureka_root_dir
import os
from isaaclab_eureka.utils import load_tensorboard_logs



def get_replay_task_feedback(log_dir: str) -> tuple[str]:

    # We import here because doing this before launching Kit causes GCC_12.0 errors
    import numpy as np

    data = load_tensorboard_logs(log_dir)
    # Make a summary of each plot in the tensorboard logs
    total_feed_back_string = "We provide you the output of the reward function on some successful demonstrations as follows, and you can utilize it for better reward generation"
    for metric_name, metric_data in data.items():
        if "Replay/" in metric_name:
            metric_name = metric_name.split("Replay/", 1)[-1]
            data_string = f"{metric_data[0]:.2f}"
            feedback_string = (
                f"{metric_name}: {data_string}"
            )
            if "Replay/success_metric" in data and metric_name == "Replay/oracle_total_rewards":
                # If success metric is available, we do not provide the oracle feedback
                feedback_string = ""
            total_feed_back_string += feedback_string

    return total_feed_back_string



if __name__ == "__main__":
    """Create the environment for the task."""
    root = eureka_root_dir()
    log_dir = os.path.join(root, "logs", "replay_test")
    task = "LivingRoomScene1PickUpTheAlphabetSoupAndPutItInTheBasket"
    from isaaclab.app import AppLauncher
    from isaaclab_eureka.utils import MuteOutput, get_freest_gpu
    device = "cuda"
    if device == "cuda":
        device_id = get_freest_gpu()
        device = f"cuda:{device_id}"
    app_launcher = AppLauncher(headless=True, device=device)
    simulation_app = app_launcher.app

    # import omni.timeline
    # timeline = omni.timeline.get_timeline_interface()
    # timeline.play() 
    # block until Isaac Sim is ready
    # while not timeline.is_playing():
    #     omni.kit.app.get_app().update()
    #     print("message")
    #     import time
    #     time.sleep(10)

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab_tasks.utils import parse_env_cfg
    env_cfg: DirectRLEnvCfg = parse_env_cfg(task)
    env_cfg.sim.device = device
    env = gym.make(task, cfg=env_cfg)
    env = env.unwrapped
    env.reset()
    env.run_replay(log_dir)

    total_feed_back_string = get_replay_task_feedback(log_dir=log_dir)
    print(total_feed_back_string)
    print("finished replaying, exiting")
    env.close()

    print("env deleted")
    torch.cuda.empty_cache()

    print("empty cache")
    env.sim.stop()
    env.sim.clear()
    simulation_app.close()

