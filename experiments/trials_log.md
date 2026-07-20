# Trials Log

One row per training/eval run. Newest at top. Edit freely; add a Notes line under any row.

| # | date | model (MH / athlete / combined) | data version & size | rating source | threshold | class weights | epochs | precision | recall | F1 | kappa | notes |
|---|------|--------------------------------|---------------------|---------------|-----------|---------------|--------|-----------|--------|----|-------|-------|
| GATE-clean | 2026-07-20 | **AND gate on fresh held-out** (mh-2516 + sport-2516) | 292 held-out (44% rel), never trained on | mh0.5/sp0.4 | — | — | 0.51 | 0.95 | **0.66** | — | **Honest clean number** (the 0.71 was split-contaminated). Beats baseline 0.61. High recall, low precision (117 FP). mh & sport each F1 0.80 on own dims. Precision is the target. |
| 9 | 2026-07-20 | **mh + baseline negatives** | 2516 mh + **1020 baseline (mh=0)** = 3536, balanced 46% pos | argmax | balanced | 3 | pending | pending | pending | — | Root-cause fix: athlete_mh sample is mh_core-matched → no clean negatives. Add `baseline` (sports, no MH keywords) as true mh-negatives to cut mh false-positives → lift gate precision. |
| AND | 2026-07-20 | **mh ∧ sport gate** (deliverable) | 2516, test 252 (relevant 44%) | grid best mh0.55/sp0.50 | — | — | 0.57 | 0.93 | **0.71** | — | **Working Layer-1 gate, beats baseline** (F1 0.71 vs 0.62, acc 0.66 vs 0.44), **recall 0.93**. Precision 0.57 (permissive) — sport near-degenerate (corpus is all sport). ⚠ Re-splitting per trial contaminates cross-model eval → future gate evals use a **fresh 292-comment held-out set** (never trained on). |
| 8 | 2026-07-20 | **sport** on 2516 | sport 1732/784 (test 69% pos) | thr 0.5 | balanced | 3 | 0.70 | 0.92 | 0.80* | — | **Weak — near base rate.** Probs capped 0.32–0.59 (never confident); built-in thr-0.6 metric 0/0/0. eval_loss stuck ~0.68. Corpus is overwhelmingly sport-context, so little signal to separate the rare non-sport minority. *F1 0.80 ≈ its 0.82 base-rate baseline. |
| 7 | 2026-07-20 | **mh** on 2516 | 2516 (mh 1619/897, test 162/90) | thr 0.6 / argmax | balanced | 3 | 0.74 | 0.73 | 0.73 (0.81@.4) | — | **Data broke the base-rate trap.** Real discrimination: specificity 48/90 (53%) at thr 0.6 vs 5/40 (12%) at 1199; probs spread to 0.755. Moderate not strong (balanced-acc ~0.63); F1 metric weak on 64%-pos class. eval_loss 0.68→0.64 (best 0.636). |
| 6 | 2026-07-20 | sport on 1199 | sport 860/339 | — | balanced | 3 | killed | killed | killed | — | Started sport-on-1199, then **killed** — superseded by 2516 expansion before completing. |
| 5 | 2026-07-20 | **mh** (mental-health dimension only) | per-dimension labels 1199, mh 766/433 (test 80/40) | argmax | balanced | 3 | 0.70 | 1.00 | 0.82 | — | **F1 0.82 but base-rate-inflated.** Predicts positive for 115/120; TN only 5/40. Always-positive baseline F1=0.80 → model +0.02 only. Learned *slight* discrimination, not strong. Confirms data-volume ceiling (both AND & mh near base rate on 959 ex). Lever = more data (expansion to 2500 running). |
| 4 | 2026-07-20 | mental_health = sports×MH | NEW labels 1199, Claude(sonnet) mh∧sport, 549/650 = 46/54 | multi-agent workflow, consistent rubric | argmax (0.35 best) | balanced | 3 | 0.48 (0.51@.35) | 0.54 (0.87@.35) | **0.50 (0.64@.35)** | — | **Learned but weak.** Probs spread 0.19–0.57 (no longer constant → pipeline+labels sound), eval_loss 0.704→0.678→0.670 across epochs. But precision ~0.5 at all thresholds, acc ~0.55 ≈ majority baseline. Below 0.80 bar. Compound AND target too hard on 959 ex → decompose (Trial 5+). Built-in metric 0/0/0 = threshold-0.6 artifact. |
| 3 | 2026-07-20 | mental_health (combined) | Codex weak labels 1000 (620/380) | Codex AI weak labels | argmax | balanced | 3 | — | — | — | — | KILLED mid-run. roberta-base but loss stuck at ~0.69 (chance) through step 120 → not learning. Suspected LR 5e-5 instability and/or weak labels. |
| 2 | 2026-07-06 | mental_health (combined) | same as Trial 1, test split n=100 (65/35) | — (eval only) | off + swept | balanced | 1 | 0.65* | 1.00* | 0.79* | — | **model is a constant predictor** (P≈0.508 for every input). "Scores" are fake = majority baseline. Training failed: 1 epoch + roberta-large on CPU. Fix: roberta-base + 3-4 epochs. |
| 1 | 2026-07-06 | mental_health (combined) | mental_health 1000 (620/380) | Codex AI weak labels ("central" rubric) | 0.6 | balanced | 1 | 0 (crashed) | 0 | undefined | — | predicted 0 relevant; threshold 0.6 too high, weak labels, 1 epoch CPU |

## Trial detail notes

### Trial 1 — baseline (broken)
- Command: `.\.venv\Scripts\python.exe code\train_relevance.py -r train_relevance -t comments -g mental_health`
- Outcome: model saved to `models/filter_relevance_mental_health`; eval predicted all-irrelevant (recall 0, F1 crash).
- Cause: `threshold=0.6` filtered out every positive; labels are AI weak labels; 1 epoch on CPU.
- Next: turn threshold off (argmax), re-evaluate the saved model before retraining.
