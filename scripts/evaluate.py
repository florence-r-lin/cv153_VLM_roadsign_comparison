"""
evaluate.py exact match + BLEU.
Outputs per image results table and aggregated summary

Usage:
    python scripts/evaluate.py --prompt_type text_extraction
    python scripts/evaluate.py --prompt_type meaning
"""
import os, sys, re, argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PREDS_DIR, METRICS_DIR, TEXT_SIGNS

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    _sf = SmoothingFunction().method1
    def _bleu(pred, ref):
        r = _norm(ref).split(); h = _norm(pred).split()
        return sentence_bleu([r], h, smoothing_function=_sf) if h and r else 0.0
except ImportError:
    def _bleu(pred, ref):
        r = set(_norm(ref).split()); h = _norm(pred).split()
        return sum(1 for w in h if w in r) / len(h) if h and r else 0.0


def _norm(text):
    """Lowercase and strip punctuation before exact match."""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _exact_match(pred, label):
    if "km/h" in str(label).lower():
        nums = re.findall(r"\d+", str(label))
        return any(n in str(pred) for n in nums)
    return _norm(label) in _norm(str(pred))


def build_results_table(df, model_name):
    # Handle old column name from previous runs
    if "pred_1" in df.columns and "prediction" not in df.columns:
        df = df.rename(columns={"pred_1": "prediction"})
    df = df.copy()
    df["model"]       = model_name
    df["sign_type"]   = df["label_name"].apply(
        lambda x: "text_sign" if x in TEXT_SIGNS else "symbol_sign"
    )
    df["exact_match"] = df.apply(
        lambda r: _exact_match(r["prediction"], r["label_name"]), axis=1
    )
    df["bleu_score"]  = df.apply(
        lambda r: round(_bleu(r["prediction"], r["label_name"]), 4), axis=1
    )
    return df


def summarize(df):
    rows = []
    for subset, sub_df in [
        ("all",          df),
        ("text_signs",   df[df["sign_type"] == "text_sign"]),
        ("symbol_signs", df[df["sign_type"] == "symbol_sign"]),
    ]:
        if len(sub_df) == 0:
            continue
        rows.append({
            "model":           sub_df["model"].iloc[0],
            "subset":          subset,
            "n_images":        len(sub_df),
            "exact_match_pct": round(sub_df["exact_match"].mean() * 100, 2),
            "avg_bleu":        round(sub_df["bleu_score"].mean(), 4),
        })
    return pd.DataFrame(rows)


def main(prompt_type):
    all_tables, all_summaries = [], []

    for model_name in ["gemma3", "gemma4"]:
        path = os.path.join(PREDS_DIR, f"{model_name}_{prompt_type}.csv")
        if not os.path.isfile(path):
            print(f"Skipping {model_name} — {path} not found.")
            continue
        table   = build_results_table(pd.read_csv(path), model_name)
        summary = summarize(table)
        all_tables.append(table)
        all_summaries.append(summary)

    if not all_tables:
        print("No prediction files found. Run run_gemma3.py and run_gemma4.py first.")
        return

    combined_table   = pd.concat(all_tables,     ignore_index=True)
    combined_summary = pd.concat(all_summaries,  ignore_index=True)

    # Print summary
    print(f"\nResults — {prompt_type}")
    print("─" * 60)
    print(f"  {'model':<10} {'subset':<15} {'n':<6} {'exact_match%':<14} {'avg_bleu'}")
    print("─" * 60)
    for _, r in combined_summary.iterrows():
        print(f"  {r['model']:<10} {r['subset']:<15} {r['n_images']:<6} {r['exact_match_pct']:<14} {r['avg_bleu']}")
    print("─" * 60)

    os.makedirs(METRICS_DIR, exist_ok=True)
    combined_table.to_csv(
        os.path.join(METRICS_DIR, f"{prompt_type}_results_table.csv"), index=False
    )
    combined_summary.to_csv(
        os.path.join(METRICS_DIR, f"{prompt_type}_summary.csv"), index=False
    )
    print(f"\nSaved → {METRICS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_type", choices=["text_extraction", "meaning"], default="text_extraction")
    args = parser.parse_args()
    main(args.prompt_type)