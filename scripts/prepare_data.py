"""
prepare_data.py
Reads Dataset 1 (Traffic Sign Classification) and creates
data/processed/eval_metadata.csv for evaluation.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --n_images 500
"""
import os
import sys
import argparse
import random
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DS1_DATA_DIR, DS1_LABELS, PROCESSED_DIR, NUM_EVAL_IMAGES, RANDOM_SEED


def main(n_images, seed):
    if not os.path.isdir(DS1_DATA_DIR):
        raise FileNotFoundError(f"Dataset not found at {DS1_DATA_DIR}.\nCheck config.py paths.")
    if not os.path.isfile(DS1_LABELS):
        raise FileNotFoundError(f"labels.csv not found at {DS1_LABELS}.")

    labels_df = pd.read_csv(DS1_LABELS)
    label_map = dict(zip(labels_df["ClassId"], labels_df["Name"]))
    print(f"Found {len(label_map)} classes.")

    records = []
    for class_id in sorted(label_map.keys()):
        class_dir = os.path.join(DS1_DATA_DIR, str(class_id))
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                records.append({
                    "image_path": os.path.join(class_dir, fname),
                    "class_id":   class_id,
                    "label_name": label_map[class_id],
                })

    print(f"Found {len(records)} total images.")

    if n_images and n_images < len(records):
        random.seed(seed)
        per_class = max(1, n_images // len(label_map))
        sampled = []
        by_class = {}
        for r in records:
            by_class.setdefault(r["class_id"], []).append(r)
        for recs in by_class.values():
            sampled.extend(random.sample(recs, min(per_class, len(recs))))
        random.shuffle(sampled)
        records = sampled[:n_images]
        print(f"Sampled {len(records)} images (stratified, seed={seed}).")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, "eval_metadata.csv")
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_images", type=int, default=NUM_EVAL_IMAGES)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    main(args.n_images, args.seed)