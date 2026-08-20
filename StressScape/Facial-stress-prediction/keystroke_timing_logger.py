"""Standalone live typing-session capture for the paired facial+keystroke
fusion evaluation (Phase B of the manuscript revision).

This produces REAL data from a real typing session -- it cannot be run on
your behalf, someone has to actually type. Run it once per consenting
author/session:

    python keystroke_timing_logger.py --name sahil

You will be asked to type a short fixed phrase a few times (mirrors the CMU
dataset's fixed-password paradigm, so the extracted features are directly
comparable to what the model was trained on). Each run appends one row to
results/Keystroke/fusion_sessions.csv with the extracted feature vector, the
trained Random Forest's stress prediction, and a timestamp -- ready to pair
with that same author's facial-model prediction for the fusion table.

Consent: same framing as the facial photo capture already documented in the
manuscript's Ethics section -- inform the participant before running this,
and only run it for consenting authors.
"""

import argparse
import csv
import time
from pathlib import Path

import joblib
import numpy as np

from keystroke_features import LiveKeystrokeTimer, FEATURE_NAMES

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "results" / "Keystroke" / "random_forest_model.pkl"
OUT_CSV = BASE_DIR / "results" / "Keystroke" / "fusion_sessions.csv"
PHRASE = "the quick brown fox jumps over the lazy dog"
REPS = 5


def run_session(name):
    if not MODEL_PATH.exists():
        print(f"ERROR: {MODEL_PATH} not found. Run train_keystroke_rf.py first.")
        return

    model = joblib.load(MODEL_PATH)
    timer = LiveKeystrokeTimer()

    print("=" * 70)
    print("KEYSTROKE TIMING SESSION")
    print("=" * 70)
    print(f"\nParticipant: {name}")
    print(f"You will type the phrase below {REPS} times, pressing ENTER after each:\n")
    print(f'  "{PHRASE}"\n')
    print("Type naturally -- this captures real per-key timing, not just the text.\n")

    timer.start()
    time.sleep(0.2)
    for i in range(REPS):
        input(f"[{i + 1}/{REPS}] Type the phrase, then press Enter: ")
    timer.stop()

    feats = timer.extract_features()
    if feats is None:
        print("Not enough keystrokes captured -- session discarded, try again.")
        return

    proba = model.predict_proba(feats)[0]
    stress_prob = float(proba[1])
    label = "Stressed" if stress_prob >= 0.5 else "Not Stressed"

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUT_CSV.exists()
    with open(OUT_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "participant", *FEATURE_NAMES, "stress_probability", "predicted_label", "n_keystrokes"])
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"), name,
            *feats[0].tolist(), stress_prob, label, timer.event_count(),
        ])

    print(f"\nCaptured {timer.event_count()} keystrokes.")
    print(f"Keystroke RF prediction: {label} ({stress_prob:.1%} stress probability)")
    print(f"Row appended to: {OUT_CSV}")
    print("\nNext: run the facial capture/prediction for the same participant, then pair")
    print("the two predictions for the fusion evaluation table.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Participant identifier (must match their facial-capture folder name)")
    args = parser.parse_args()
    run_session(args.name)
