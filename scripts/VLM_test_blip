import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = AutoModelForVisualQuestionAnswering.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    dtype=torch.float16,
    device_map="auto"
)

image = Image.open("/home/shaotongchen/Pictures/Screenshots/lab_18.png").convert("RGB")

question = "Question: What is happening in this image? Answer:"

inputs = processor(images=image, text=question, return_tensors="pt")

# move safely
inputs = {k: v.to(model.device) for k, v in inputs.items()}

generated_ids = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

print("generated_ids shape:", generated_ids.shape)

# DEBUG: don't trim yet
output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("RAW OUTPUT:", output)