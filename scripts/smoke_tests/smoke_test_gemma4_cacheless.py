import os
import sys
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from config import GEMMA4_MODEL_ID, HF_CACHE_DIR

IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "smoke_tests", "image.jpg")

print(f"Model     : {GEMMA4_MODEL_ID}")
print(f"Cache dir : {HF_CACHE_DIR}")
print(f"Image     : {IMAGE_PATH}")
print(f"CUDA      : {torch.cuda.is_available()} | devices: {torch.cuda.device_count()}\n")

processor = AutoProcessor.from_pretrained(GEMMA4_MODEL_ID, cache_dir=HF_CACHE_DIR)
model = AutoModelForImageTextToText.from_pretrained(
    GEMMA4_MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    cache_dir=HF_CACHE_DIR,
).eval()

image = Image.open(IMAGE_PATH).convert("RGB")
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": "What is written in this image? Answer briefly."},
]}]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True,
    tokenize=True, return_dict=True, return_tensors="pt",
)
inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=64, do_sample=False)

prompt_len = inputs["input_ids"].shape[-1]
print("Output:", processor.decode(output[0][prompt_len:], skip_special_tokens=True))
print("\nGemma 4 smoke test passed.")