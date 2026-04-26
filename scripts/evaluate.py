"""
evaluate.py
Computes Top-1, Top-N accuracy and BLEU for both models.
Splits evaluation into text signs (have readable text) vs symbol signs (no text).

Usage:
    python scripts/evaluate.py --prompt_type text_extraction
    python scripts/evaluate.py --prompt_type meaning
"""
import os
import sys
import re
import json
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PREDS_DIR, METRICS_DIR, TOP_N

# ── Sign categorization ───────────────────────────────────────────────────────
TEXT_SIGNS = {
    "Speed limit (15km/h)", "Speed limit (30km/h)", "Speed limit (40km/h)",
    "Speed limit (50km/h)", "Speed limit (5km/h)", "Speed limit (60km/h)",
    "Speed limit (70km/h)", "speed limit (80km/h)", "Give Way", "No entry",
    "No stopping", "Horn", "No horn", "No Car", "No Uturn",
    "Danger Ahead", "Under Construction",
}

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


def _match_numeric(pred, label):
    nums = re.findall(r"\d+", label)
    if not nums:
        return _match(pred, label)
    return any(n in pred for n in nums)


def compute_metrics(df, model_name, top_n, subset_name):
    if len(df) == 0:
        return None
    pred_cols = [f"pred_{j+1}" for j in range(top_n) if f"pred_{j+1}" in df.columns]
    n = len(df)

    def match_row(row):
        if "km/h" in str(row["label_name"]).lower():
            return _match_numeric(row["pred_1"], row["label_name"])
        return _match(row["pred_1"], row["label_name"])

    def match_topn(row):
        if "km/h" in str(row["label_name"]).lower():
            return any(_match_numeric(row[c], row["label_name"]) for c in pred_cols)
        return any(_match(row[c], row["label_name"]) for c in pred_cols)

    top1  = sum(match_row(row) for _, row in df.iterrows())
    topn  = sum(match_topn(row) for _, row in df.iterrows())
    bleus = [_bleu(row["pred_1"], row["label_name"]) for _, row in df.iterrows()]

    return {
        "model":    model_name,
        "subset":   subset_name,
        "n_images": n,
        "top1_accuracy": round(top1 / n, 4),
        f"top{len(pred_cols)}_accuracy": round(topn / n, 4),
        "avg_bleu": round(sum(bleus) / n, 4),
    }


def print_table(results):
    if not results:
        return
    headers = list(results[0].keys())
    print("\n" + "─" * 72)
    print("  " + "  ".join(f"{h:<18}" for h in headers))
    print("─" * 72)
    for r in results:
        print("  " + "  ".join(f"{str(v):<18}" for v in r.values()))
    print("─" * 72)


def main(prompt_type):
    all_results = []
    for model_name in ["gemma3", "gemma4"]:
        path = os.path.join(PREDS_DIR, f"{model_name}_{prompt_type}.csv")
        if not os.path.isfile(path):
            print(f"Skipping {model_name} — {path} not found.")
            continue
        df = pd.read_csv(path)
        df_text   = df[df["label_name"].isin(TEXT_SIGNS)]
        df_symbol = df[~df["label_name"].isin(TEXT_SIGNS)]
        for subset_df, subset_name in [
            (df,        "all"),
            (df_text,   "text_signs"),
            (df_symbol, "symbol_signs"),
        ]:
            m = compute_metrics(subset_df, model_name, TOP_N, subset_name)
            if m:
                all_results.append(m)

    if not all_results:
        print("No prediction files found. Run run_gemma3.py and run_gemma4.py first.")
        return

    print(f"\nResults — prompt_type: {prompt_type}")
    print_table(all_results)

    os.makedirs(METRICS_DIR, exist_ok=True)
    out_json = os.path.join(METRICS_DIR, f"{prompt_type}_results.json")
    out_csv  = os.path.join(METRICS_DIR, f"{prompt_type}_results.csv")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    pd.DataFrame(all_results).to_csv(out_csv, index=False)
    print(f"\nSaved → {out_json}")
    print(f"Saved → {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_type", choices=["text_extraction", "meaning"], default="text_extraction")
    args = parser.parse_args()
    main(args.prompt_type)