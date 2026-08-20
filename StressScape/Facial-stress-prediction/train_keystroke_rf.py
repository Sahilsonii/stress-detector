"""Keystroke-dynamics stress-proxy Random Forest, trained on the real public
CMU Keystroke Dynamics Benchmark (Killourhy & Maxion, 2009).

The public dataset has no backspace/word-count/error-rate columns and no
stress ground truth -- it is 51 subjects x 400 repetitions of typing the
fixed password ".tie5Roanl", with per-keystroke hold (H), down-down (DD) and
up-down/flight (UD) latencies. This script derives a small, real feature set
from those timings and a documented proxy stress label, then trains the same
RandomForestClassifier configuration described in the manuscript plus a
DecisionTreeClassifier baseline for comparison.

Outputs (mirrors the facial pipeline's results/{model}/ layout):
  results/Keystroke/metrics.json
  results/Keystroke/classification_report.txt
  results/Keystroke/confusion_matrix.png
  results/Keystroke/feature_importance.png
  results/Keystroke/threshold_optimization.png
  results/Keystroke/feature_statistics.json   (Table 2 equivalent)
  results/Keystroke/split_summary.json        (Table 3 equivalent)
  results/Keystroke/threshold_table.json      (Table 4 equivalent)
  results/Keystroke/random_forest_model.pkl
  results/Keystroke/decision_tree_model.pkl
"""

import json
import csv
from pathlib import Path

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "DSL-StrongPasswordData.csv"
OUT_DIR = BASE_DIR / "results" / "Keystroke"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
FEATURE_NAMES = ["hold_mean", "hold_std", "flight_mean", "flight_std", "total_duration"]


def load_raw():
    with open(DATA_PATH, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    h_idx = [i for i, c in enumerate(header) if c.startswith("H.")]
    dd_idx = [i for i, c in enumerate(header) if c.startswith("DD.")]
    ud_idx = [i for i, c in enumerate(header) if c.startswith("UD.")]
    subjects, feats = [], []
    for row in rows:
        subjects.append(row[0])
        h = np.array([float(row[i]) for i in h_idx])
        dd = np.array([float(row[i]) for i in dd_idx])
        ud = np.array([float(row[i]) for i in ud_idx])
        feats.append([
            h.mean(), h.std(),
            ud.mean(), ud.std(),
            dd.sum(),
        ])
    return np.array(subjects), np.array(feats, dtype=float)


def group_split(subjects, seed=RANDOM_STATE, train_frac=0.6, val_frac=0.2):
    uniq = sorted(set(subjects))
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    train_subj = set(uniq[:n_train])
    val_subj = set(uniq[n_train:n_train + n_val])
    test_subj = set(uniq[n_train + n_val:])
    assert not (train_subj & val_subj) and not (train_subj & test_subj) and not (val_subj & test_subj)
    return train_subj, val_subj, test_subj


def make_labels(subjects, X, train_mask):
    """Within-subject relative typing-irregularity proxy label.

    Deliberately defined as each rep's deviation from THAT SUBJECT's own
    personal baseline (not a global z-score of the same columns fed to the
    classifier) -- using a global threshold on the raw feature values here
    would make the label an almost-exact linear function of three of the
    model's five input features (verified: that construction gave ~99%
    test accuracy, a data-leakage red flag). The classifier only ever sees
    the absolute per-rep features, never the subject's baseline or this
    relative score, so it cannot trivially recover the label -- it has to
    learn population-level cues that correlate with within-subject
    slowdown/hesitation, which is a genuinely harder and more defensible
    task. Thresholded on the TRAIN split only.
    """
    hold_std, flight_std, total_dur = X[:, 1], X[:, 3], X[:, 4]
    baseline = {}
    for s in set(subjects):
        m = subjects == s
        baseline[s] = (np.median(hold_std[m]), np.median(flight_std[m]), np.median(total_dur[m]))
    b_hs = np.array([baseline[s][0] for s in subjects])
    b_fs = np.array([baseline[s][1] for s in subjects])
    b_td = np.array([baseline[s][2] for s in subjects])

    score = (0.4 * (hold_std - b_hs) / (b_hs + 1e-9)
             + 0.4 * (flight_std - b_fs) / (b_fs + 1e-9)
             + 0.2 * (total_dur - b_td) / (b_td + 1e-9))
    threshold = np.quantile(score[train_mask], 0.70)
    labels = (score >= threshold).astype(int)
    return labels, float(threshold)


def threshold_sweep(y_true, proba):
    rows = []
    for t in np.arange(0.10, 0.90 + 1e-9, 0.05):
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({"threshold": round(float(t), 2), "precision": prec, "recall": rec,
                      "f1_score": f1, "specificity": spec})
    return rows


def main():
    subjects, X = load_raw()
    train_subj, val_subj, test_subj = group_split(subjects)
    train_mask = np.isin(subjects, list(train_subj))
    val_mask = np.isin(subjects, list(val_subj))
    test_mask = np.isin(subjects, list(test_subj))

    y, label_threshold = make_labels(subjects, X, train_mask)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5,
                                 min_samples_leaf=2, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)

    dt = DecisionTreeClassifier(max_depth=10, min_samples_split=5, min_samples_leaf=2,
                                 random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)

    # --- headline RF-vs-DT comparison: BOTH at their natural default (0.5) threshold ---
    # (threshold optimisation below is a separate, later analysis -- keeping it
    # out of this comparison avoids comparing a tuned RF against an untuned DT)
    test_proba = rf.predict_proba(X_test)[:, 1]
    rf_pred_default = (test_proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, rf_pred_default, labels=[0, 1]).ravel()

    rf_accuracy = accuracy_score(y_test, rf_pred_default)
    rf_precision = precision_score(y_test, rf_pred_default, zero_division=0)
    rf_recall = recall_score(y_test, rf_pred_default, zero_division=0)
    rf_f1 = f1_score(y_test, rf_pred_default, zero_division=0)

    dt_pred = dt.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_pred)
    dt_precision = precision_score(y_test, dt_pred, zero_division=0)
    dt_recall = recall_score(y_test, dt_pred, zero_division=0)
    dt_f1 = f1_score(y_test, dt_pred, zero_division=0)

    # --- threshold optimisation: selected on VALIDATION, reported at chosen theta on TEST ---
    val_proba = rf.predict_proba(X_val)[:, 1]
    val_sweep = threshold_sweep(y_val, val_proba)
    best = max(val_sweep, key=lambda r: r["f1_score"])
    chosen_threshold = best["threshold"]
    test_pred = (test_proba >= chosen_threshold).astype(int)
    otn, ofp, ofn, otp = confusion_matrix(y_test, test_pred, labels=[0, 1]).ravel()
    opt_accuracy = accuracy_score(y_test, test_pred)
    opt_precision = precision_score(y_test, test_pred, zero_division=0)
    opt_recall = recall_score(y_test, test_pred, zero_division=0)
    opt_f1 = f1_score(y_test, test_pred, zero_division=0)

    # --- 5-fold group-aware CV accuracy (mean +/- std), mirrors "cross-validation across
    # five folds" language already in the manuscript ---
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=5)
    cv_accs = []
    for tr_idx, te_idx in gkf.split(X, y, groups=subjects):
        m = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5,
                                    min_samples_leaf=2, random_state=RANDOM_STATE)
        m.fit(X[tr_idx], y[tr_idx])
        cv_accs.append(accuracy_score(y[te_idx], m.predict(X[te_idx])))
    cv_accs = np.array(cv_accs)

    importances = rf.feature_importances_
    importance_pct = {name: float(v) for name, v in zip(FEATURE_NAMES, importances / importances.sum() * 100)}

    metrics = {
        "model": "RandomForestClassifier",
        "hyperparameters": {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5,
                             "min_samples_leaf": 2, "random_state": RANDOM_STATE},
        "label_rule": "within-subject relative typing-irregularity score: 0.4*(hold_std-personal_baseline)/baseline + 0.4*(flight_std-personal_baseline)/baseline + 0.2*(total_duration-personal_baseline)/baseline, thresholded at the 70th percentile of the TRAIN split's score distribution. Classifier features are the absolute per-rep values only -- baseline and relative score are never fed to the model.",
        "label_threshold_raw_score": label_threshold,
        "test_accuracy_default_threshold_0.5": rf_accuracy,
        "test_precision_default_threshold_0.5": rf_precision,
        "test_recall_default_threshold_0.5": rf_recall,
        "test_f1_default_threshold_0.5": rf_f1,
        "test_confusion_matrix_default_threshold_0.5": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "chosen_decision_threshold": chosen_threshold,
        "chosen_threshold_selected_on": "validation split (F1-maximizing)",
        "test_accuracy_optimized_threshold": opt_accuracy,
        "test_precision_optimized_threshold": opt_precision,
        "test_recall_optimized_threshold": opt_recall,
        "test_f1_optimized_threshold": opt_f1,
        "test_confusion_matrix_optimized_threshold": {"tn": int(otn), "fp": int(ofp), "fn": int(ofn), "tp": int(otp)},
        "cv_accuracy_mean": float(cv_accs.mean()),
        "cv_accuracy_std": float(cv_accs.std()),
        "decision_tree_baseline": {"accuracy": dt_accuracy, "precision": dt_precision,
                                    "recall": dt_recall, "f1_score": dt_f1},
        "rf_vs_dt_accuracy_diff_pp": float((rf_accuracy - dt_accuracy) * 100),
        "feature_importance_pct": importance_pct,
        "n_subjects_total": int(len(set(subjects))),
        "n_subjects_train": len(train_subj), "n_subjects_val": len(val_subj), "n_subjects_test": len(test_subj),
        "n_samples_train": int(train_mask.sum()), "n_samples_val": int(val_mask.sum()), "n_samples_test": int(test_mask.sum()),
        "class_balance_positive_rate_pct": float(y.mean() * 100),
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    report = classification_report(y_test, rf_pred_default, target_names=["Not Stressed", "Stressed"])
    cm_text = f"\n\nConfusion Matrix (default threshold 0.5):\n[[{tn} {fp}]\n [{fn} {tp}]]\n"
    cm_text += f"\nConfusion Matrix (optimized threshold {chosen_threshold}):\n[[{otn} {ofp}]\n [{ofn} {otp}]]\n"
    (OUT_DIR / "classification_report.txt").write_text(
        "Classification Report for Keystroke Random Forest (real CMU DSL-StrongPasswordData)\n"
        + "=" * 60 + "\n\n" + report + cm_text
    )

    # feature statistics table (Table 2 equivalent)
    stats = {}
    for i, name in enumerate(FEATURE_NAMES):
        col = X[:, i]
        stats[name] = {"mean": float(col.mean()), "std": float(col.std()), "min": float(col.min()),
                        "max": float(col.max()), "median": float(np.median(col))}
    (OUT_DIR / "feature_statistics.json").write_text(json.dumps(stats, indent=2))

    (OUT_DIR / "split_summary.json").write_text(json.dumps({
        "train": {"subjects": len(train_subj), "samples": int(train_mask.sum())},
        "val": {"subjects": len(val_subj), "samples": int(val_mask.sum())},
        "test": {"subjects": len(test_subj), "samples": int(test_mask.sum())},
    }, indent=2))

    (OUT_DIR / "threshold_table.json").write_text(json.dumps(val_sweep, indent=2))

    joblib.dump(rf, OUT_DIR / "random_forest_model.pkl")
    joblib.dump(dt, OUT_DIR / "decision_tree_model.pkl")

    # --- figures ---
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = np.array([[tn, fp], [fn, tp]])
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Not Stressed", "Stressed"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Not Stressed", "Stressed"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Random Forest Confusion Matrix (Keystroke, real test set)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    order = np.argsort(importances)[::-1]
    ax.bar([FEATURE_NAMES[i] for i in order], importances[order] / importances.sum() * 100)
    ax.set_ylabel("Feature importance (%)")
    ax.set_title("Random Forest Feature Importance (Keystroke)")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ts = [r["threshold"] for r in val_sweep]
    ax.plot(ts, [r["precision"] for r in val_sweep], label="Precision", marker="o")
    ax.plot(ts, [r["recall"] for r in val_sweep], label="Recall", marker="o")
    ax.plot(ts, [r["f1_score"] for r in val_sweep], label="F1-score", marker="o")
    ax.plot(ts, [r["specificity"] for r in val_sweep], label="Specificity", marker="o")
    ax.axvline(chosen_threshold, color="gray", linestyle="--", label=f"Chosen theta={chosen_threshold}")
    ax.set_xlabel("Decision threshold (theta)"); ax.set_ylabel("Score")
    ax.set_title("Threshold Optimization (validation split)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "threshold_optimization.png", dpi=150)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
