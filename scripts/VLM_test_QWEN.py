from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import numpy as np

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

IMAGE_DIR   = "/home/shaotongchen/workspace_eureka/IsaacLabEureka/logs/rl_runs/rsl_rl_eureka/pick_it_up/2026-04-11_18-41-03_Run-0_tueilsy-st-13/pictures"
NUM_SAMPLES = 6

all_files = sorted(
    [f for f in os.listdir(IMAGE_DIR) if f.endswith(".png") and f.startswith("rgb_")],
    key=lambda f: int(f.split("_")[1]),
)
if len(all_files) > NUM_SAMPLES:
    indices = np.linspace(0, len(all_files) - 1, NUM_SAMPLES, dtype=int)
    sampled = [all_files[i] for i in indices]
else:
    sampled = all_files

messages = [
    {
        "role": "user",
        "content": [
            *[{"type": "image", "image": os.path.join(IMAGE_DIR, f)} for f in sampled],
            {
                "type": "text",
                "text": (
                    "Analyze this sequence. Two coordinate frames are visible: "
                    "one at the gripper fingers, one on the target object.\n"
                    "Answer:\n"
                    "1. Is the object grasped? (yes/no/unsure)\n"
                    "2. Is it positioned above the basket? (yes/no/unsure)\n"
                    "3. Did the robot drop the object in the basket? (yes/no/unsure)\n"
                    "4. Did the robot stick its hand into the basket? (yes/no/unsure)\n"
                    "5. Brief reasoning."
                ),
            },
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text[0])
