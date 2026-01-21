import pickle

with open("libero/trajs/libero90/libero_90_kitchen_scene2_open_the_top_drawer_of_the_cabinet_traj_v2.pkl", "rb") as f: 
    data = pickle.load(f)
    print(type(data))
    print(data.keys())
    print(len(data["franka"]))
    # print(data["metadata"])
    # {'task_name': 'libero_90_kitchen_scene2_open_the_top_drawer_of_the_cabinet',
    #  'robot_name': 'franka', 'num_episodes': 50, 'source': 'libero',
    #  'original_file': 'KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet_demo.hdf5'}

    print(data['franka'][0].keys())
    print(data['franka'][0]["init_state"])
    # print(data['franka'][0]["actions"])
    # print(data['franka'][0]["states"])