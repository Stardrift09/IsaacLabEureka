from transformers import pipeline

Prompt = ["""
Describe the scene in general""",
"""
Determine whether the robot is currently grasping the object""",
"""
Is the object attached by an coordinate frame currently above the basket?"""
]


pipe = pipeline("image-text-to-text", model="llava-hf/llava-1.5-7b-hf")

for i in range(3):
    p = Prompt[i]
    messages = [
    {
    "role": "user",
    "content": [
        {"type": "image", "path": "/home/shaotongchen/Pictures/Screenshots/lab_18.png"},
        {"type": "text", "text": p},
    ],
    }
    ]

    out = pipe(text=messages, max_new_tokens=100)
    print(out)


# import torch

# q1 = torch.tensor([0.0022, 0.9276, 0.3735, 0.0054])
# q2 = torch.tensor([-0.0213, 0.9716, -0.2311, -0.0456])

# # make sure quaternions are normalized
# q1 = q1 / q1.norm()
# q2 = q2 / q2.norm()

# dot = torch.abs(torch.dot(q1, q2))
# angle_rad = 2 * torch.acos(dot)
# angle_deg = torch.rad2deg(angle_rad)
# print(angle_deg)



# test for master thesis
"""
/home/shaotongchen/Pictures/Screenshots/Screenshot from 2026-03-10 12-44-37.png:
[{'input_text': [{'role': 'user', 'content': [{'type': 'image', 'path': '/home/shaotongchen/Pictures/Screenshots/Screenshot from 2026-03-10 12-44-37.png'}, {'type': 'text', 'text': '\nDescribe the scene in general'}]}], 'generated_text': [{'role': 'user', 'content': [{'type': 'image', 'path': '/home/shaotongchen/Pictures/Screenshots/Screenshot from 2026-03-10 12-44-37.png'}, {'type': 'text', 'text': '\nDescribe the scene in general'}]}, {'role': 'assistant', 'content': ' The image features a white robotic arm with a camera on top, positioned on a black surface. The robotic arm is holding a small white box, possibly a tissue box. The scene appears to be a futuristic setting, with the robotic arm being the main focus.'}]}]
Both `max_new_tokens` (=100) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
[{'input_text': [{'role': 'user', 'content': [{'type': 'image', 'path': '/home/shaotongchen/Pictures/Screenshots/Screenshot from 2026-03-10 12-44-37.png'}, {'type': 'text', 'text': '\nDetermine whether the robot is currently grasping the object'}]}], 'generated_text': [{'role': 'user', 'content': [{'type': 'image', 'path': '/home/shaotongchen/Pictures/Screenshots/Screenshot from 2026-03-10 12-44-37.png'}, {'type': 'text', 'text': '\nDetermine whether the robot is currently grasping the object'}]}, {'role': 'assistant', 'content': ' In the image, the robot is not currently grasping the object. It is positioned next to a basket, and there is no indication that it is in the process of picking up or holding the object. The robot appears to be in a stationary position, and the object is placed separately from the robot.'}]}]
Both `max_new_tokens` (=100) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
[{'input_text': [{'role': 'user', 'content': [{'type': 'image', 'path': '/home/shaotongchen/Pictures/Screenshots/Screenshot from 2026-03-10 12-44-37.png'}, {'type': 'text', 'text': '\nIs the object currently above the basket?'}]}], 'generated_text': [{'role': 'user', 'content': [{'type': 'image', 'path': '/home/shaotongchen/Pictures/Screenshots/Screenshot from 2026-03-10 12-44-37.png'}, {'type': 'text', 'text': '\nIs the object currently above the basket?'}]}, {'role': 'assistant', 'content': ' Yes, the object is currently above the basket.'}]}]
Conclusion: hallucination in describing the scene, good answer in saying it is not grasped. Wrong answer in saying it is above the basket. The object is not above it.


"""