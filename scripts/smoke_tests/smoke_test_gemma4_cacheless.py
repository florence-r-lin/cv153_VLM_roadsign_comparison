import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_ID = "google/gemma-4-E4B-it"
IMAGE_PATH = "image.jpg"
CACHE_DIR = "/tmp/$USER/hf-cache/hub"

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
)

model = AutoModelForMultimodalLM.from_pretrained(
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
            {"type": "text", "text": "What is written on this sign? Answer briefly."}
        ]
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
        do_sample=False
    )

prompt_len = inputs["input_ids"].shape[-1]
decoded = processor.decode(output[0][prompt_len:], skip_special_tokens=True)
print(decoded)