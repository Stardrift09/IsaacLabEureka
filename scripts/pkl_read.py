# import pickle
# import os
# from isaaclab_eureka.utils import eureka_root_dir
# root = eureka_root_dir()
# folder = os.path.join(root, "libero/trajs/libero90")
# pkl_files = [
#     os.path.join(folder, f)
#     for f in os.listdir(folder)
#     if f.endswith(".pkl")
# ]
# print(len(pkl_files))
# for pkl_file in pkl_files:
#   with open(pkl_file, "rb") as f: 
#       data = pickle.load(f)
#       # print(type(data))
#       # print(data.keys())
#       # print(len(data["franka"]))
#       # print(data["metadata"])
#       # {'task_name': 'libero_90_kitchen_scene2_open_the_top_drawer_of_the_cabinet',
#       #  'robot_name': 'franka', 'num_episodes': 50, 'source': 'libero',
#       #  'original_file': 'KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet_demo.hdf5'}

#       # print(data['franka'][0].keys())
#       print(data['franka'][0]["init_state"].keys())
#       # print(data['franka'][0]["actions"][0])
#       # print(data['franka'][0]["states"])




# import torch

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
    model="gpt-5.4",
    # reasoning={"effort": "high"},
    input=
    """
I am calculating reward:
for the RL env, episode length is 500 steps. The reward is accumulated and devided by episode length in s.
Now I have successful demos with length maximum 172(which is not the max cause it is undefined. How do I calculate the reward for this to make the two comparable?


"""
)

print(response.output_text)