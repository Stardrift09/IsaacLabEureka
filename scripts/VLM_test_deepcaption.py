from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "prithivMLmods/DeepCaption-VLA-7B", torch_dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained("prithivMLmods/DeepCaption-VLA-7B")
messages = [
    {
        "role": "user",
        "content": [
            *[
                {"type": "image", "image": f"/home/shaotongchen/workspace_eureka/IsaacLabEureka/logs/picture_test/rgb_{i}_0.png"}
                for i in range(1,65)
            ],
            {
                "type": "text",
                "text": """
                Analyze the sequence of images. There are two coordinate frames, 
                one is attached to the center of gripper fingers, another is on the target object.
                Answer:
                1. Is the object grasped? (yes/no/you are not sure)
                2. Is it positioned above the basket? (yes/no/you are not sure)
                3. Did the robot drop the object in the basket? (yes/no/you are not sure)
                4. Did the robot stick its hand into the basket? (yes/no/you are not sure)
                6. Brief reasoning.
                """
            },
        ],
    }
]
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)

