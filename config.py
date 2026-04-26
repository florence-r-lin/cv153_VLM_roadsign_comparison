import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Dataset 1: Traffic Sign Classification ────────────────────────────────────
DS1_ROOT     = os.path.join(PROJECT_ROOT, "data", "traffic_signs")
DS1_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "traffic_signs", "traffic_Data", "DATA")
DS1_LABELS   = os.path.join(PROJECT_ROOT, "data", "traffic_signs", "labels.csv")

# ── Dataset 2: US Road Signs (COCO format) ────────────────────────────────────
DS2_ROOT       = os.path.join(PROJECT_ROOT, "data", "us_road_signs", "coco")
DS2_TRAIN_DIR  = os.path.join(PROJECT_ROOT, "data", "us_road_signs", "coco", "train")
DS2_VALID_DIR  = os.path.join(PROJECT_ROOT, "data", "us_road_signs", "coco", "valid")
DS2_TEST_DIR   = os.path.join(PROJECT_ROOT, "data", "us_road_signs", "coco", "test")
DS2_TRAIN_JSON = os.path.join(PROJECT_ROOT, "data", "us_road_signs", "coco", "train", "_annotations.coco.json")
DS2_VALID_JSON = os.path.join(PROJECT_ROOT, "data", "us_road_signs", "coco", "valid", "_annotations.coco.json")
DS2_TEST_JSON  = os.path.join(PROJECT_ROOT, "data", "us_road_signs", "coco", "test", "_annotations.coco.json")

# ── Processed / output paths ──────────────────────────────────────────────────
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
PREDS_DIR     = os.path.join(PROJECT_ROOT, "outputs", "predictions")
METRICS_DIR   = os.path.join(PROJECT_ROOT, "outputs", "metrics")

# ── Model IDs ─────────────────────────────────────────────────────────────────
GEMMA3_MODEL_ID = "google/gemma-3-4b-it"
GEMMA4_MODEL_ID = "google/gemma-4-E4B-it"

# ── HF cache: /tmp avoids blowing your home-dir quota ────────────────────────
HF_CACHE_DIR = f"/tmp/{os.environ.get('USER', 'user')}/hf-cache/hub"

# ── Eval settings ─────────────────────────────────────────────────────────────
MAX_NEW_TOKENS  = 64
NUM_EVAL_IMAGES = 200
RANDOM_SEED     = 42
TOP_N           = 3

# ── Prompts ───────────────────────────────────────────────────────────────────
PROMPTS = {
    "text_extraction": (
        "What text is written on this road sign? "
        "Reply with only the text you see, nothing else."
    ),
    "meaning": (
        "What does this road sign mean? "
        "Reply with one short sentence."
    ),
}