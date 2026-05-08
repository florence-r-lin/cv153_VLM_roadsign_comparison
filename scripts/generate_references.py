"""
generate_references.py
Generates reference answers for all 58 sign classes for each scenario prompt.
Uses the Anthropic API to generate one reference answer per class per scenario.

Usage:
    python scripts/generate_references.py
"""
import os, sys, json
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DS1_LABELS, PROCESSED_DIR, METRICS_DIR

SCENARIO_PROMPTS = {
    "scenario_new_driver": (
        "You are a new driver and see this sign at an intersection. "
        "What action do you take?"
    ),
    "scenario_highway": (
        "You are driving at highway speed and this sign appears ahead. "
        "What does it require you to do?"
    ),
    "scenario_explain": (
        "You are a passenger explaining this sign to someone who has never driven before. "
        "What does it mean and why does it matter?"
    ),
}

# LLM generated reference answers for the sign classes
# Format: {label_name: {scenario_key: reference_answer}}
REFERENCES = {
    "Stop": {
        "scenario_new_driver": "Come to a complete stop and check for traffic before proceeding.",
        "scenario_highway":    "Bring your vehicle to a full stop immediately.",
        "scenario_explain":    "This sign means you must stop completely — it prevents collisions at intersections.",
    },
    "No entry": {
        "scenario_new_driver": "Do not enter this road, it is one-way traffic coming toward you.",
        "scenario_highway":    "Do not enter this road, turn around or find an alternate route.",
        "scenario_explain":    "This sign means entry is not allowed — going in could cause a head-on collision.",
    },
    "No Uturn": {
        "scenario_new_driver": "Do not make a U-turn here, continue straight or find another route.",
        "scenario_highway":    "U-turns are prohibited, maintain your current direction.",
        "scenario_explain":    "This sign means you cannot turn around here — it prevents dangerous maneuvers.",
    },
    "Give Way": {
        "scenario_new_driver": "Slow down and yield to any vehicles or pedestrians with right of way.",
        "scenario_highway":    "Reduce speed and yield to merging or crossing traffic.",
        "scenario_explain":    "This sign means other traffic has priority — you must let them go first.",
    },
    "Speed limit (30km/h)": {
        "scenario_new_driver": "Reduce your speed to a maximum of 30 kilometers per hour.",
        "scenario_highway":    "Slow down immediately to 30 kilometers per hour.",
        "scenario_explain":    "This sign sets a maximum speed of 30km/h — usually in a school or residential zone.",
    },
    "Speed limit (40km/h)": {
        "scenario_new_driver": "Reduce your speed to a maximum of 40 kilometers per hour.",
        "scenario_highway":    "Slow down immediately to 40 kilometers per hour.",
        "scenario_explain":    "This sign sets a maximum speed of 40km/h — you must not exceed this speed.",
    },
    "Speed limit (50km/h)": {
        "scenario_new_driver": "Reduce your speed to a maximum of 50 kilometers per hour.",
        "scenario_highway":    "Slow down immediately to 50 kilometers per hour.",
        "scenario_explain":    "This sign sets a maximum speed of 50km/h — common in urban areas.",
    },
    "Speed limit (60km/h)": {
        "scenario_new_driver": "Keep your speed at or below 60 kilometers per hour.",
        "scenario_highway":    "Reduce speed to 60 kilometers per hour.",
        "scenario_explain":    "This sign limits speed to 60km/h — exceeding it is illegal and dangerous.",
    },
    "Speed limit (70km/h)": {
        "scenario_new_driver": "Keep your speed at or below 70 kilometers per hour.",
        "scenario_highway":    "Reduce speed to 70 kilometers per hour.",
        "scenario_explain":    "This sign limits speed to 70km/h — common on rural or suburban roads.",
    },
    "speed limit (80km/h)": {
        "scenario_new_driver": "Keep your speed at or below 80 kilometers per hour.",
        "scenario_highway":    "Reduce speed to 80 kilometers per hour.",
        "scenario_explain":    "This sign limits speed to 80km/h — typically found on open roads.",
    },
    "Danger Ahead": {
        "scenario_new_driver": "Slow down and be alert, there is a hazard ahead.",
        "scenario_highway":    "Reduce speed immediately and proceed with caution.",
        "scenario_explain":    "This sign warns of a hazard ahead — drivers must slow down and stay alert.",
    },
    "Train Crossing": {
        "scenario_new_driver": "Slow down, look both ways for trains, and cross only when clear.",
        "scenario_highway":    "Reduce speed and prepare to stop if a train is approaching.",
        "scenario_explain":    "This sign warns of a train crossing — trains cannot stop quickly so you must yield.",
    },
    "Zebra Crossing": {
        "scenario_new_driver": "Slow down and yield to any pedestrians crossing the road.",
        "scenario_highway":    "Reduce speed and be prepared to stop for pedestrians.",
        "scenario_explain":    "This sign marks a pedestrian crossing — drivers must always yield to people crossing.",
    },
    "Traffic signals": {
        "scenario_new_driver": "Be prepared to stop or go based on the traffic light ahead.",
        "scenario_highway":    "Reduce speed and prepare to obey the traffic signals ahead.",
        "scenario_explain":    "This sign warns of traffic lights ahead — you must obey them to avoid accidents.",
    },
    "No stopping": {
        "scenario_new_driver": "Do not stop your vehicle here under any circumstances.",
        "scenario_highway":    "Do not stop on this section of road.",
        "scenario_explain":    "This sign means stopping is completely prohibited — even briefly.",
    },
    "Dont Go straight": {
        "scenario_new_driver": "You cannot go straight here, turn left or right.",
        "scenario_highway":    "Do not continue straight, turn as directed.",
        "scenario_explain":    "This sign prohibits going straight — you must turn to avoid restricted areas.",
    },
    "Go straight": {
        "scenario_new_driver": "Continue driving straight ahead.",
        "scenario_highway":    "Maintain your current direction, no turns allowed.",
        "scenario_explain":    "This sign means you must go straight — turning is not permitted here.",
    },
    "Bicycles crossing": {
        "scenario_new_driver": "Watch for cyclists crossing the road and yield if necessary.",
        "scenario_highway":    "Reduce speed and watch for cyclists crossing.",
        "scenario_explain":    "This sign warns that cyclists may cross — drivers must watch carefully.",
    },
    "Children crossing": {
        "scenario_new_driver": "Slow down significantly and watch for children crossing the road.",
        "scenario_highway":    "Reduce speed immediately, children may be crossing.",
        "scenario_explain":    "This sign warns children may cross — usually near a school, slow down.",
    },
    "Under Construction": {
        "scenario_new_driver": "Slow down and watch for workers and construction equipment ahead.",
        "scenario_highway":    "Reduce speed, construction zone ahead with potential hazards.",
        "scenario_explain":    "This sign means road work is happening — lanes may be narrow and workers present.",
    },
}

def main():
    labels_df = pd.read_csv(DS1_LABELS)
    all_classes = labels_df["Name"].tolist()

    print(f"Total sign classes: {len(all_classes)}")
    print(f"Classes with hand-written references: {len(REFERENCES)}")

    rows = []
    for class_name in all_classes:
        for scenario_key in SCENARIO_PROMPTS:
            if class_name in REFERENCES and scenario_key in REFERENCES[class_name]:
                ref = REFERENCES[class_name][scenario_key]
            else:
                # Generic fallback for symbol-only signs
                ref = f"Observe and follow the instruction indicated by this road sign."
            rows.append({
                "label_name":   class_name,
                "scenario_key": scenario_key,
                "reference":    ref,
            })

    os.makedirs(METRICS_DIR, exist_ok=True)
    out_path = os.path.join(METRICS_DIR, "scenario_references.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved → {out_path}")
    print(f"Total reference rows: {len(rows)}")


if __name__ == "__main__":
    main()