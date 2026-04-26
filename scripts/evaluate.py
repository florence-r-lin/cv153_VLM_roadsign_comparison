"""
evaluate.py
Computes Top-1, Top-N accuracy and BLEU for both models.

Usage:
    python scripts/evaluate.py --prompt_type text_extraction
"""
import os
import sys
import re
import json
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PREDS_DIR, METRICS_DIR, TOP_N

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    _sf = SmoothingFunction().method1
    def _bleu(pred, ref):
        r = _norm(ref).split()
        h = _norm(pred).split()
        if not h or not r:
            return 0.0
        return sentence_bleu([r], h, smoothing_function=_sf)
except ImportError:
    def _bleu(pred, ref):
        r = set(_norm(ref).split())
        h = _norm(pred).split()
        if not h or not r:
            return 0.0
        return sum(1 for w in h if w in r) / len(h)


def _norm(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _match(pred, label):
    return _norm(label) in _norm(pred)


def evaluate_model(df, model_name, top_n):
    pred_cols = [f"pred_{j+1}" for j in range(top_n) if f"pred_{j+1}" in df.columns]
    n = len(df)
    top1 = sum(_match(row["pred_1"], row["label_name"]) for _, row in df.iterrows())
    topn = sum(any(_match(row[c], row["label_name"]) for c in pred_cols) for _, row in df.iterrows())
    bleus = [_bleu(row["pred_1"], row["label_name"]) for _, row in df.iterrows()]
    return {
        "model": model_name,
        "n_images": n,
        "top1_accuracy": round(top1 / n, 4),
        f"top{len(pred_cols)}_accuracy": round(topn / n, 4),
        "avg_bleu": round(sum(bleus) / n, 4),
    }


def main(prompt_type):
    results = []
    for model_name in ["gemma3", "gemma4"]:
        path = os.path.join(PREDS_DIR, f"{model_name}_{prompt_type}.csv")
        if not os.path.isfile(path):
            print(f"Skipping {model_name} — {path} not found.")
            continue
        metrics = evaluate_model(pd.read_csv(path), model_name, TOP_N)
        results.append(metrics)
        print(f"\n{model_name.upper()}: {metrics}")

    if not results:
        print("No prediction files found. Run run_gemma3.py and run_gemma4.py first.")
        return

    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(os.path.join(METRICS_DIR, f"{prompt_type}_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(results).to_csv(os.path.join(METRICS_DIR, f"{prompt_type}_results.csv"), index=False)
    print(f"\nSaved → {METRICS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_type", choices=["text_extraction", "meaning"], default="text_extraction")
    args = parser.parse_args()
    main(args.prompt_type)