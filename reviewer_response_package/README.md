# Reviewer response package

Evidence package for the Stress-Scape Major Revision response.

- `response_to_reviewers.txt` — point-by-point reply to the editor's 8-item letter.
- `facial_results/{MobileNetV2,EfficientNetB0,ResNet50V2}/` — real `metrics.json`, `classification_report.txt`, confusion matrix + training history figures for each trained facial model.
- `facial_results/comparison/` — regenerated cross-model comparison figures (now built from real confusion matrices, not approximated ones).
- `facial_results/FINAL_MODEL_COMPARISON_REPORT.pdf` — full regenerated comparison report.
- `keystroke_results/` — real `metrics.json`, `classification_report.txt`, confusion matrix, feature-importance chart, threshold-optimization curve, feature statistics, and split summary for the newly-trained keystroke Random Forest (trained on the real public CMU dataset — see `../manuscript_corrections.md` section 7 for why the feature set changed).

`../manuscript_corrections.md` (repo root) has the full text-level manuscript edits these numbers support.

**Still pending:** the fusion evaluation section in both the manuscript and this response depends on a real paired facial+keystroke session (see `../StressScape/Facial-stress-prediction/keystroke_timing_logger.py`). Once that data exists, `response_to_reviewers.txt` section 2 and `manuscript_corrections.md` section 8 need a final update with the real numbers.
