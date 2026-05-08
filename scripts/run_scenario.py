"""
run_scenario.py
Runs all 3 scenario prompts for a given model on eval_metadata.csv.

Usage:
    export CUDA_VISIBLE_DEVICES=0
    python scripts/run_scenario.py --model gemma3
    python scripts/run_scenario.py --model gemma4
"""
import os, sys, argparse
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    GEMMA3_MODEL_ID, GEMMA4_MODEL_ID, HF_CACHE_DIR,
    PROCESSED_DIR, PREDS_DIR, MAX_NEW_TOKENS, PROMPTS
)

SCENARIO_KEYS = ["scenario_new_driver", "scenario_highway", "scenario_explain"]

try:
    from transformers import AutoProcessor, AutoModelForImageTextToText
    _MODEL_CLS = AutoModelForImageTextToText
except ImportError:
    from transformers import AutoProcessor, AutoModelForCausalLM
    _MODEL_CLS = AutoModelForCausalLM


def load_model(model_name):
    model_id = GEMMA3_MODEL_ID if model_name == "gemma3" else GEMMA4_MODEL_ID
    print(f"Loading {model_id} ...")
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=HF_CACHE_DIR)
    model = _MODEL_CLS.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16, cache_dir=HF_CACHE_DIR,
    ).eval()
    print(f"Loaded on {next(model.parameters()).device}")
    return processor, model


def run_inference(processor, model, image_path, prompt):
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": prompt},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    prompt_len = inputs["input_ids"].shape[-1]
    return processor.decode(output[0][prompt_len:], skip_special_tokens=True).strip()


def main(model_name):
    meta_path = os.path.join(PROCESSED_DIR, "eval_metadata.csv")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError("eval_metadata.csv not found. Run prepare_data.py first.")
    df = pd.read_csv(meta_path)
    print(f"Model: {model_name} | Images: {len(df)} | Scenarios: {len(SCENARIO_KEYS)}\n")

    processor, model = load_model(model_name)

    for scenario_key in SCENARIO_KEYS:
        prompt = PROMPTS[scenario_key]
        print(f"\n--- Running: {scenario_key} ---")
        print(f"Prompt: {prompt}\n")

        results = []
        for i, row in df.iterrows():
            if i % 20 == 0:
                print(f"  [{i+1}/{len(df)}]")
            try:
                pred = run_inference(processor, model, row["image_path"], prompt)
            except Exception as e:
                print(f"  ERROR: {e}")
                pred = "ERROR"
            results.append({
                "image_id":    i,
                "image_path":  row["image_path"],
                "class_id":    row["class_id"],
                "label_name":  row["label_name"],
                "prompt_type": scenario_key,
                "prediction":  pred,
            })

        os.makedirs(PREDS_DIR, exist_ok=True)
        out_path = os.path.join(PREDS_DIR, f"{model_name}_{scenario_key}.csv")
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"Saved → {out_path}")

    print(f"\nAll scenarios done for {model_name}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gemma3", "gemma4"], required=True)
    args = parser.parse_args()
    main(args.model)