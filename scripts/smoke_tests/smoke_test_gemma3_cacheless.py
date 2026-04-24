import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import os

MODEL_ID = "google/gemma-3-4b-it"
# IMAGE_PATH = "image.jpg"
IMAGE_PATH = "image.png"
# CACHE_DIR = "/tmp/$USER/hf-cache/hub"
os.environ["HF_HOME"] = f"/tmp/{os.environ['USER']}/hf-cache"
CACHE_DIR = os.environ["HF_HOME"]

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
)

model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    dtype=torch.bfloat16,
    cache_dir=CACHE_DIR,
).eval()

image = Image.open(IMAGE_PATH).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "What is written in this image? Answer briefly."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)

inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
    )

prompt_len = inputs["input_ids"].shape[-1]
print(processor.decode(output[0][prompt_len:], skip_special_tokens=True))