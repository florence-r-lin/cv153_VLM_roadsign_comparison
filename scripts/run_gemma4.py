"""
run_gemma4.py
Runs Gemma 4 (E4B-IT) on eval_metadata.csv and saves predictions.

Usage:
    export CUDA_VISIBLE_DEVICES=1
    python scripts/run_gemma4.py --prompt_type text_extraction
"""
import os
import sys
import argparse
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GEMMA4_MODEL_ID, HF_CACHE_DIR, PROCESSED_DIR, PREDS_DIR, MAX_NEW_TOKENS, TOP_N, PROMPTS

try:
    from transformers import AutoProcessor, AutoModelForImageTextToText
    _MODEL_CLS = AutoModelForImageTextToText
except ImportError:
    from transformers import AutoProcessor, AutoModelForCausalLM
    _MODEL_CLS = AutoModelForCausalLM


def load_model():
    print(f"Loading {GEMMA4_MODEL_ID} (class: {_MODEL_CLS.__name__}) ...")
    processor = AutoProcessor.from_pretrained(GEMMA4_MODEL_ID, cache_dir=HF_CACHE_DIR)
    model = _MODEL_CLS.from_pretrained(
        GEMMA4_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=HF_CACHE_DIR,
    ).eval()
    print(f"Loaded on {next(model.parameters()).device}")
    return processor, model


def run_inference(processor, model, image_path, prompt, top_n):
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": prompt},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            num_beams=top_n, num_return_sequences=top_n, do_sample=False,
        )
    prompt_len = inputs["input_ids"].shape[-1]
    return [processor.decode(o[prompt_len:], skip_special_tokens=True).strip() for o in outputs]


def main(prompt_type, top_n):
    meta_path = os.path.join(PROCESSED_DIR, "eval_metadata.csv")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError("eval_metadata.csv not found. Run prepare_data.py first.")
    df = pd.read_csv(meta_path)
    prompt = PROMPTS[prompt_type]
    print(f"Prompt: {prompt}\nImages: {len(df)}, Top-N: {top_n}\n")
    processor, model = load_model()
    results = []
    for i, row in df.iterrows():
        if i % 20 == 0:
            print(f"  [{i+1}/{len(df)}]")
        try:
            preds = run_inference(processor, model, row["image_path"], prompt, top_n)
        except Exception as e:
            print(f"  ERROR: {e}")
            preds = ["ERROR"] * top_n
        record = {"image_path": row["image_path"], "class_id": row["class_id"],
                  "label_name": row["label_name"], "prompt_type": prompt_type}
        for j, p in enumerate(preds):
            record[f"pred_{j+1}"] = p
        results.append(record)
    os.makedirs(PREDS_DIR, exist_ok=True)
    out_path = os.path.join(PREDS_DIR, f"gemma4_{prompt_type}.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_type", choices=list(PROMPTS.keys()), default="text_extraction")
    parser.add_argument("--top_n", type=int, default=TOP_N)
    args = parser.parse_args()
    main(args.prompt_type, args.top_n)