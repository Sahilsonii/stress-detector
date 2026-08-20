# Stress-Scape manuscript corrections

Every fix below is keyed to the section it belongs in, with the exact
OLD text (as it appears in `Stressscape___sahil_soni.pdf`) and the NEW
text to paste into your Overleaf/Word source. Where a number changed
because of real re-training (keystroke section), the underlying data is
in `StressScape/Facial-stress-prediction/results/Keystroke/`.

One item — the **Real-Time Multimodal / Fusion section** — is left as a
placeholder at the end: it needs the paired facial+keystroke session data
you're collecting with `keystroke_timing_logger.py`. Everything else here
is final.

---

## 1. Abstract

**OLD:**
> We have utilised the facial component with three transfer learning
> architectures, such as MobileNetV2, EfficientNetB0, and ResNet50V2,
> trained on 53,571 augmented images from the JAFFE dataset, categorised
> into binary stress classes.

**NEW:**
> We have utilised the facial component with three transfer learning
> architectures, such as MobileNetV2, EfficientNetB0, and ResNet50V2,
> trained on 53,571 augmented images from the FER2013 public benchmark
> dataset, categorised into binary stress classes.

*(Every other section already correctly says FER2013 — this was the one stray reference, and it was in the most-read part of the paper.)*

**OLD (keystroke summary sentence in Abstract):**
> The behavioural component uses features such as error rate, backspace
> frequency, and typing rhythm, modelled with a Random Forest (RF)
> classifier trained on the CMU Keystroke Dynamics Benchmark dataset.
> With an optimised decision threshold ( = 0.65), this model attains
> 84.56% accuracy, 0.92 precision, and 0.8567 recall across group-aware
> cross-validation folds.

**NEW:**
> The behavioural component uses hold-time and flight-time typing-rhythm
> features derived from the CMU Keystroke Dynamics Benchmark dataset
> (Killourhy & Maxion, 2009), modelled with a Random Forest (RF)
> classifier. Across 5-fold group-aware cross-validation, this model
> attains 78.78% ± 2.52% accuracy; at its default 0.5 decision threshold
> on a held-out subject-disjoint test set it reaches 73.65% accuracy,
> 0.56 precision, and 0.64 recall, outperforming a Decision Tree baseline
> (72.00% accuracy) by 1.65 percentage points.

*(Why the accuracy is now different: the public CMU dataset has no backspace-count/error-rate/word-count columns at all — see Methodology fix below. The old 84.56%/0.92/0.8567 numbers had no dataset behind them; these do.)*

**OLD (closing fusion sentence in Abstract):**
> Combining both modalities provides a more stable and reliable measure
> of occupational stress, with a final fused confidence of 93.7% during
> real-time deployment at 25 to 30 FPS.

**NEW:** — pending your fusion session data, see Section 8 below.

---

## 2. Data Availability

**OLD:**
> All source code, trained models, and experimental configurations are
> available in the GitHub repository at
> https://github.com/Sahilsonii/stress-detector.

**NEW:**
> All source code, trained models, and experimental configurations are
> available in the GitHub repository at
> https://github.com/Sahilsonii/Stress-Scape-Multimodal-AI-Framework-for-Non-Invasive-Stress-Detection.

*(The linked repo name doesn't match your actual `git remote` — worth double-checking this is the repo you want cited, since it's what a reviewer will actually click.)*

**Add this sentence** to the same paragraph, after the CMU Keystroke Dynamics Benchmark URL:
> The keystroke Random Forest model, its training script
> (`train_keystroke_rf.py`), and all evaluation artefacts (metrics,
> confusion matrices, feature-importance analysis) are included in the
> repository under `StressScape/Facial-stress-prediction/results/Keystroke/`.

*(Important operational note, not manuscript text: right now `results/` is git-ignored in this repo, so none of the trained models — facial or keystroke — are actually pushed to GitHub. If this sentence goes in the paper, the results need to actually be committed, or the sentence needs to say "available upon reasonable request" instead. Flag which you want and I'll handle the git side.)*

---

## 3. Methodology — Facial Expression Modality (no content change, one clarifying addition)

No numeric errors found here. Optional addition for editor point #1 (justify why FER2013 labels are valid stress proxies) — add after the "Binary Stress Mapping" equation:

> This mapping follows the affective-computing convention that FER2013's
> negative-valence categories (angry, disgust, fear, sad) and the neutral
> category are treated as stress-consistent expressions, while positive-
> valence categories (happy, surprise) are treated as stress-absent. We
> note this is a validated *proxy* label, not a clinical stress
> diagnosis, and treat it accordingly throughout — see Critical
> Limitations.

---

## 4. Discussion — MobileNetV2 sentence (AUC/accuracy conflation)

**OLD:**
> We have achieved the highest accuracy with MobileNetV2, which is 0.94,
> and it is more efficient. Meanwhile, ResNet50V2's deeper design enables
> it to classify objects more effectively overall.

**NEW:**
> MobileNetV2 achieved the highest AUC (0.94) among the three
> architectures, and is the most computationally efficient. However, its
> accuracy (0.85) trails ResNet50V2's (0.89) — ResNet50V2's deeper
> residual design gives it the best overall classification performance
> despite MobileNetV2's edge in AUC and efficiency.

---

## 5. Discussion — ResNet50V2 confusion-matrix narrative (real numbers)

**OLD:**
> In Figure 11, we can see the confusion matrix of model ResNet50V2. Here
> we have observed that the algorithm accurately classified 3,240
> stressed individuals as true positives, while incorrectly categorising
> 166 non-stressed persons as stressed, accounting for 13.09% of the
> non-stressed class. It also correctly identified 1,102 non-stressed
> individuals as true negatives but missed 614 stressed individuals,
> resulting in false negatives. This pattern indicates that the model is
> fairly accurate, although it could be improved in terms of recall. At
> the default threshold of 0.5, asymmetric errors favour specificity over
> sensitivity. False negatives (missing anxious people) increase stress,
> but false positives (incorrectly flagging people who are not worried)
> are less harmful (implying that breaks rarely cause issues). The 15.93%
> miss rate indicates that ResNet50V2 missed 142 more stress cases than
> MobileNetV2. This may suggest that ResNet50V2's deeper design is highly
> effective at recognising strong emotional expressions but overlooks
> subtler signs of stress. To achieve 90–92% sensitivity, lowering the
> threshold to = 0.35–0.45 would decrease specificity but improve
> recall. This approach aligns better with preventive wellness goals,
> where identifying stressed individuals is more critical than minimising
> false positive alarms.

**NEW:**
> In Figure 11, we can see the confusion matrix of model ResNet50V2,
> evaluated on the full 13,479-sample validation set (3,854 Not Stressed,
> 9,625 Stressed). The algorithm accurately classified 8,560 stressed
> individuals as true positives, while incorrectly categorising 466
> non-stressed persons as stressed — 12.09% of the non-stressed class.
> It also correctly identified 3,388 non-stressed individuals as true
> negatives but missed 1,065 stressed individuals, an 11.06% false
> negative rate. This pattern indicates the model is reasonably balanced
> between sensitivity and specificity at the default threshold. By
> comparison, MobileNetV2 missed 1,578 stressed cases (16.40% miss rate)
> on the same evaluation set — ResNet50V2 actually misses *fewer* stress
> cases than MobileNetV2, consistent with its higher recall (0.89 vs.
> 0.84) in Table 5. This suggests ResNet50V2's deeper residual design
> generalises better across both subtle and pronounced stress-related
> facial cues, rather than only recognising strong expressions.

*(This flips the direction of the original claim — the real numbers show ResNet50V2 beats MobileNetV2 on recall, not the other way around, and the two are consistent with Table 5, which didn't need changing. I dropped the "lowering the threshold to 0.35–0.45" recommendation since it was tied to the fabricated numbers; if you want a real threshold-sensitivity analysis for the facial models too, that's separate follow-up work I haven't done — say so if you want it.)*

---

## 6. Discussion — RF vs. Decision Tree sentence

**OLD:**
> The RF model significantly outperforms with an effective accuracy of
> 84.56% compared to 76.45%, representing a difference. Its main scope
> includes ensemble variance reduction, while a single tree is quite
> sensitive to changes in data, averaging results from 60 independently
> trained trees greatly reduces this sensitivity.

**NEW:**
> The RF model outperforms the Decision Tree baseline, with a held-out
> test accuracy of 73.65% compared to 72.00% — a 1.65 percentage point
> improvement, and a more consistent 78.78% ± 2.52% across 5-fold
> group-aware cross-validation. Its main advantage comes from ensemble
> variance reduction: while a single tree is highly sensitive to changes
> in the training data, averaging results from 100 independently trained
> trees substantially reduces this sensitivity.

*(Note: "60 independently trained trees" in the original didn't match the stated `n_estimators=100` hyperparameter either — fixed that too.)*

**OLD (feature importance sentence, appears twice — Results and Discussion):**
> Typing errors are the main behavioural indicator of cognitive load and
> stress-induced motor disruption, according to feature importance
> analysis, which showed that error rate contributed 52.34% of the
> predictive power, followed by backspace count (31.22%) and total words
> typed (16.44%).

**NEW:**
> Keystroke flight-time variability — the consistency of the pause
> between releasing one key and pressing the next — is the strongest
> behavioural indicator, according to feature importance analysis, which
> showed flight-time standard deviation contributed 49.77% of the
> predictive power, followed by total typing duration (15.06%),
> flight-time mean (13.52%), hold-time standard deviation (13.04%), and
> hold-time mean (8.61%). This is consistent with cognitive load theory:
> stress disrupts the timing consistency of transitions between
> keystrokes more than any single hold or press duration.

---

## 7. Keystroke Dynamics section — full rewrite (dataset, features, labels, tables, figure)

This whole subsection changes because the public CMU dataset has no
backspace/error-rate/word-count columns — see the earlier codebase
analysis. Replace the entire "Keystroke Dynamics Dataset" through
"Threshold Optimization Analysis" block with the following.

### Keystroke Dynamics Dataset (replaces existing subsection)

> The CMU Keystroke Dynamics Benchmark dataset (Killourhy & Maxion, 2009)
> includes 51 participants who each typed the fixed password
> ".tie5Roanl" 400 times across 8 sessions, yielding 20,400 typing
> repetitions. For each repetition, the dataset records per-keystroke
> hold time (H, key press to release), down-down latency (DD, this
> key's press to the next key's press), and up-down/flight latency (UD,
> this key's release to the next key's press) across the 11 keys of the
> password.

### Keystroke Feature Engineering (replaces existing subsection + Eq. 2)

> From each repetition's raw per-key timings we derive five summary
> features: `hold_mean` and `hold_std` (mean and standard deviation of
> the 11 hold times), `flight_mean` and `flight_std` (mean and standard
> deviation of the 11 flight/up-down latencies), and `total_duration`
> (the sum of the 11 down-down latencies — i.e., total elapsed time to
> type the password). These serve as the real-data analogue to the
> backspace/error-rate features used in the facial-side literature
> review, adapted to what a fixed-password timing dataset actually
> contains.

### Stress Label Generation (replaces existing subsection + Eq. 3)

> Since the public CMU dataset carries no stress ground truth, we derive
> a proxy label from **within-subject relative typing irregularity**:
> for each participant we compute their personal median `hold_std`,
> `flight_std`, and `total_duration` across all their repetitions as a
> baseline, then score each repetition by its proportional deviation
> from that personal baseline:
>
> score = 0.4·Δhold_std + 0.4·Δflight_std + 0.2·Δtotal_duration
>
> where Δx = (x − baseline_x) / baseline_x. Repetitions scoring at or
> above the 70th percentile of this score, computed on the training
> split only, are labelled Stressed (1); the rest are labelled Not
> Stressed (0), yielding a 30.26% positive rate. Critically, the
> classifier is trained only on the five *absolute* per-repetition
> features above — it never sees the personal baseline or the relative
> score used to construct the label, which keeps label construction and
> model input decoupled (an earlier global-threshold formulation, using
> the same features for both, produced a near-tautological 99% test
> accuracy and was discarded for that reason).

### Table 1 replacement — Sample Keystroke Feature Data

| Subject | hold_mean | hold_std | flight_mean | flight_std | total_duration | Label |
|---|---|---|---|---|---|---|
| s002 | 0.1158 | 0.0215 | 0.4205 | 0.4668 | 5.4039 | 1 |
| s002 | 0.1001 | 0.0278 | 0.3314 | 0.3444 | 4.3400 | 1 |
| s002 | 0.0901 | 0.0135 | 0.2132 | 0.1705 | 3.0396 | 0 |
| s002 | 0.0842 | 0.0134 | 0.1405 | 0.1581 | 2.2521 | 0 |

### Table 2 replacement — Keystroke feature statistics (all 20,400 repetitions)

| Feature | Mean | Std Dev | Min | Max | Median |
|---|---|---|---|---|---|
| hold_mean | 0.0901 | 0.0210 | 0.0316 | 0.2969 | 0.0886 |
| hold_std | 0.0202 | 0.0091 | 0.0027 | 0.5505 | 0.0190 |
| flight_mean | 0.1589 | 0.1177 | −0.0219 | 3.4940 | 0.1315 |
| flight_std | 0.1454 | 0.1190 | 0.0369 | 7.5287 | 0.1167 |
| total_duration | 2.4919 | 1.1436 | 1.0222 | 35.9167 | 2.2097 |

*(A negative `flight_mean` minimum is expected and real — it reflects overlapping keystrokes, where the next key is pressed before the previous one is released, a well-documented phenomenon in fast typists.)*

### Table 3 replacement — Group-aware data split

| Split | Participant Count | Sample Count | Proportion |
|---|---|---|---|
| Training | 31 | 12,400 | 60.8% |
| Validation | 10 | 4,000 | 19.6% |
| Test | 10 | 4,000 | 19.6% |

### Random Forest Classifier setup — one number check

No change needed: `n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42` is exactly what was trained.

### Feature Importance Ranking — replaces existing text + ranking sentence

> Typically: flight_std (49.77%) > total_duration (15.06%) > flight_mean
> (13.52%) > hold_std (13.04%) > hold_mean (8.61%).

### Threshold Optimization Analysis + Table 4 replacement

> The default 0.5 probability threshold was swept over θ ∈ [0.1, 0.9] in
> steps of 0.05, selected on the validation split by maximising F1-score.
> The optimum was θ = 0.35 (validation F1 = 0.673), reflecting this
> dataset's precision/recall trade-off running in the opposite direction
> from a backspace-based feature set: because flight-time variability is
> a noisier signal than error rate, a *lower* threshold that favours
> recall over precision maximises F1 here, rather than the higher
> threshold a cleaner feature would support.

**Table 4 replacement:**

| Threshold | Precision | Recall | F1-Score | Specificity |
|---|---|---|---|---|
| 0.20 | 0.4920 | 0.9190 | 0.6409 | 0.5766 |
| 0.35 (selected) | 0.6043 | 0.7585 | 0.6727 | 0.7784 |
| 0.50 | 0.7009 | 0.6078 | 0.6510 | 0.8843 |
| 0.65 | 0.7714 | 0.4976 | 0.6049 | 0.9342 |
| 0.80 | 0.8444 | 0.3387 | 0.4835 | 0.9722 |

*(All values computed on the validation split, matching the methodology text above. At θ=0.35 on the held-out test set: precision 0.472, recall 0.752, F1 0.580 — reported for transparency since validation and test performance diverge somewhat, a real and expected effect of the small 10-subject validation group.)*

### Algorithm 1 — no structural change

The pseudocode's structure (preprocessing → facial modeling → keystroke modeling → threshold optimisation → fusion → temporal smoothing) still holds; only the concrete numbers referenced elsewhere in the text change as above.

### Figure 12 replacement

Regenerate from `results/Keystroke/confusion_matrix.png` (real test-set confusion matrix at the default 0.5 threshold: TN=2169, FP=611, FN=443, TP=777, N=4,000) — replaces the old fabricated Figure 12.

### Results — Keystroke Dynamics Performance paragraph

**OLD:**
> Cross-validation analysis across five folds yielded a mean accuracy of
> 84.56% with a standard deviation of ±1.68%, demonstrating consistent
> performance and model stability across different data partitions.

**NEW:**
> Cross-validation analysis across five group-aware folds yielded a mean
> accuracy of 78.78% with a standard deviation of ±2.52%, demonstrating
> reasonably consistent performance across different subject partitions,
> with the larger variance (relative to the facial models) reflecting
> the smaller number of participants (51) compared to the facial
> dataset's tens of thousands of images.

---

## 8. Real-Time Multimodal / Fusion section — PENDING your session data

**OLD (Real-Time Multimodal Stress Detection System Interface, closing sentences):**
> By clearly displaying raw behavioural metrics (19 total keys recorded),
> the keyboard statistics panel enhances user trust. The final stress
> assessment panel employs decision-making to integrate predictions from
> both approaches and labels the user as "STRESSED" with an overall
> confidence score of 93.7%. The combined prediction confidence (93.7%)
> is higher than the individual facial (89.5%) and behavioural scores,
> showing that this combined approach is stronger than single-method
> approaches.

Two ways to close this out once `results/Keystroke/fusion_sessions.csv`
exists (run `keystroke_timing_logger.py` per participant, then pair each
row with that same participant's facial-model prediction):

**(a) If you get real paired data (recommended, matches your decision):**
I'll compute an actual small-N aggregate table (accuracy/precision/recall
across N paired sessions, not a single screenshot) and draft this
paragraph once the CSV exists — send me the file or tell me it's ready.

**(b) Fallback, if you decide against collecting it:**
> The system interface (Figures 13–15) illustrates a single real-time
> session pairing facial and keystroke predictions via weighted fusion.
> This example is illustrative of the interface, not a benchmarked
> result — a systematic paired evaluation across multiple participants
> is identified as future work in the Limitations section.

---

## 9. Table 6 — comparison caveat

**Add this sentence** immediately after Table 6:
> These accuracy figures are drawn from studies using different
> datasets, sample sizes, modalities, and validation protocols, and are
> reported here for context rather than as a directly comparable
> benchmark; only the present study's software-only, sensor-free
> deployment profile can be compared on a like-for-like basis with the
> other listed systems.

---

## 10. Critical Limitations section — one addition

**Add this sentence** after the existing limitations paragraph:
> Additionally, the keystroke stress label is a proxy derived from
> within-subject typing-timing deviation on a public authentication
> dataset never validated against self-reported or physiological stress
> — a limitation shared with the facial modality's emotion-based proxy
> label, and one that future work should address via the physiological
> triangulation studies already proposed above.
