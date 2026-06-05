# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from collections import defaultdict

import GPUtil
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path

def load_tensorboard_logs(path: str):
    """Load tensorboard logs from a given path.

    Args:
        path: The path to the tensorboard logs.

    Returns:
        A dictionary with the tags and their respective values.
    """
    data = defaultdict(list)
    event_acc = EventAccumulator(path)
    event_acc.Reload()  # Load all data written so far

    for tag in event_acc.Tags()["scalars"]:
        events = event_acc.Scalars(tag)
        for event in events:
            data[tag].append(event.value)

    return data


def get_freest_gpu():
    """Get the GPU with the most free memory."""
    gpus = GPUtil.getGPUs()
    if not gpus:
        return None
    # Sort GPUs by memory usage
    gpus.sort(key=lambda gpu: gpu.memoryUsed)
    print([gpu.id for gpu in gpus])
    return gpus[0].id


class MuteOutput:
    """Context manager to mute stdout and stderr."""

    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")  # noqa: SIM115
        sys.stderr = open(os.devnull, "w")  # noqa: SIM115
        return self

    def __exit__(self, *args):
        sys.stdout = self._stdout
        sys.stderr = self._stderr


import pickle
def read_pkl(path):
    debug = True
    with open(path, "rb") as f: 
        data = pickle.load(f)
        if debug: 
            # print(type(data))
            # print(data.keys())
            # print(len(data["franka"])) # 50

            # print(data["metadata"])
            # {'task_name': 'libero_90_kitchen_scene2_open_the_top_drawer_of_the_cabinet',
            #  'robot_name': 'franka', 'num_episodes': 50, 'source': 'libero',
            #  'original_file': 'KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet_demo.hdf5'}

            # print(data['franka'][0].keys()) # ['init_state', 'actions', 'states']
            print(data['franka'][0]["init_state"])
            # print(data['franka'][0]["init_state"].keys())
            # print(data['franka'][0]["init_state"]['franka']["dof_pos"])
            # print(data['franka'][0]["init_state"]["basket"])
            # print(data['franka'][0]["actions"])
            # for i in range(len(data['franka'][0]["states"])):
            #     print(data['franka'][0]["states"][i]["franka"])
            # for i in range(50):
            #     print(len(data['franka'][i]["states"]))
    return data


def eureka_root_dir():
    return Path(os.path.realpath(__file__)).parents[3]


if __name__ == "__main__":
    root = eureka_root_dir()
    path = "/home/shaotongchen/workspace_eureka/IsaacLabEureka/libero/trajs/libero90/libero_90_kitchen_scene2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle_traj_v2.pkl"
    read_pkl(path=path)
    import pdb
    pdb.set_trace()
    # print(eureka_root_dir())
    # get_freest_gpu()