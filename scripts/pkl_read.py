# import pickle
# import os
# with open("libero/trajs/libero90/libero_90_kitchen_scene2_open_the_top_drawer_of_the_cabinet_traj_v2.pkl", "rb") as f: 
#     data = pickle.load(f)
#     print(type(data))
#     print(data.keys())
#     print(len(data["franka"]))
#     # print(data["metadata"])
#     # {'task_name': 'libero_90_kitchen_scene2_open_the_top_drawer_of_the_cabinet',
#     #  'robot_name': 'franka', 'num_episodes': 50, 'source': 'libero',
#     #  'original_file': 'KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet_demo.hdf5'}

#     print(data['franka'][0].keys())
#     # print(data['franka'][0]["init_state"])
#     # print(data['franka'][0]["actions"][0])
#     # print(data['franka'][0]["states"])

# os.system("sudo reboot")

import torch

# t1 = torch.tensor([[True,True],[False,False]]).float()
# print(t1)
# print(t1.mean())

# t2 = torch.tensor([[1,2,3],[1,2,0.05],[0,0,0.05]])
# root_pos_w = t2[:,2]
# low_enough = root_pos_w < 0.1
# obj_xy = t2[:,:2]
# dist2 = (obj_xy **2).sum(dim=-1)
# inside_site =  dist2 < 1
# terminated = inside_site & low_enough

# print(f"terminated{terminated}")
# print(f"terminated.float().mean(){terminated.float().mean()}")



from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-5.2",
    input=
    """
  I am using isaaclaberueka, trying to deploy it on slurm cluster, so I wonder how should I adjust apt-get update && apt-get install -y xvfb
export DISPLAY=:0
Xvfb :0 -screen 0 1920x1080x24 &
to multiple gpus? I have 8 a100 on one node. Morever, the task manager multiprocessing happens only in one gpu and lead to memory errors, how do I evenly split it on all gpus

    def __init__(
        self,
        task: str,
        rl_library: Literal["rsl_rl", "rl_games"] = "rsl_rl",
        num_processes: int = 1,
        device: str = "cuda",
        env_seed: int = 42,
        max_training_iterations: int = 100,
        success_metric_string: str = "",
    ):
        self._task = task
        self._rl_library = rl_library
        self._num_processes = num_processes
        self._device = device
        self._max_training_iterations = max_training_iterations
        self._success_metric_string = success_metric_string
        self._env_seed = env_seed
        # if self._success_metric_string:
        #     self._success_metric_string = "extras['Eureka/success_metric'] = " + self._success_metric_string

        self._processes = dict()
        # Used to communicate the reward functions to the processes
        self._rewards_queues = [multiprocessing.Queue() for _ in range(self._num_processes)]
        # Used to communicate the observations method to the main process
        self._observations_queue = multiprocessing.Queue()
        # Used to communicate the results of the training runs to the main process
        self._results_queue = multiprocessing.Queue()
        # Used to signal the processes to terminate
        self.termination_event = multiprocessing.Event()

        for idx in range(self._num_processes):
            p = multiprocessing.Process(target=self._worker, args=(idx, self._rewards_queues[idx]))
            self._processes[idx] = p
            p.start()
"""
)

print(response.output_text)