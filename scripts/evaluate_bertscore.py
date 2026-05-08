"""
evaluate_bertscore.py
Computes BERTScore F1 for scenario prompt predictions vs reference answers.

Usage:
    pip install --no-cache-dir bert-score
    python scripts/evaluate_bertscore.py
"""
import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PREDS_DIR, METRICS_DIR

SCENARIO_KEYS = ["scenario_new_driver", "scenario_highway", "scenario_explain"]


def main():
    try:
        from bert_score import score as bert_score
    except ImportError:
        print("bert-score not installed. Run: pip install --no-cache-dir bert-score")
        return

    ref_path = os.path.join(METRICS_DIR, "scenario_references.csv")
    if not os.path.isfile(ref_path):
        print("References not found. Run generate_references.py first.")
        return

    refs_df = pd.read_csv(ref_path)
    all_results = []

    for model_name in ["gemma3", "gemma4"]:
        for scenario_key in SCENARIO_KEYS:
            path = os.path.join(PREDS_DIR, f"{model_name}_{scenario_key}.csv")
            if not os.path.isfile(path):
                print(f"Skipping {model_name} {scenario_key} — file not found.")
                continue

            preds_df = pd.read_csv(path)
            merged = preds_df.merge(
                refs_df[refs_df["scenario_key"] == scenario_key],
                on="label_name", how="left"
            )
            merged["reference"] = merged["reference"].fillna(
                "Observe and follow the instruction indicated by this road sign."
            )

            predictions = merged["prediction"].astype(str).tolist()
            references  = merged["reference"].astype(str).tolist()

            print(f"\nScoring {model_name} / {scenario_key} ({len(predictions)} images)...")
            P, R, F1 = bert_score(
                predictions, references,
                lang="en", verbose=False,
                device="cuda" if __import__("torch").cuda.is_available() else "cpu"
            )

            merged["bertscore_f1"] = F1.tolist()

            # Save per-image results
            out_table = os.path.join(METRICS_DIR, f"{model_name}_{scenario_key}_bertscore.csv")
            merged[["image_id","label_name","prediction","reference","bertscore_f1"]].to_csv(
                out_table, index=False
            )

            avg_f1 = F1.mean().item()
            print(f"  Avg BERTScore F1: {avg_f1:.4f}")

            all_results.append({
                "model":        model_name,
                "scenario":     scenario_key,
                "n_images":     len(merged),
                "avg_bertscore_f1": round(avg_f1, 4),
            })

    summary = pd.DataFrame(all_results)
    summary_path = os.path.join(METRICS_DIR, "bertscore_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"\n{'─'*60}")
    print(f"{'model':<10} {'scenario':<25} {'n':<6} {'avg_bertscore_f1'}")
    print(f"{'─'*60}")
    for _, r in summary.iterrows():
        print(f"{r['model']:<10} {r['scenario']:<25} {r['n_images']:<6} {r['avg_bertscore_f1']}")
    print(f"{'─'*60}")
    print(f"\nSaved → {summary_path}")


if __name__ == "__main__":
    main()