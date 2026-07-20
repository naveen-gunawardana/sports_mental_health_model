# Layer-1 Relevance Classifier — Overnight Progress Report

**Date:** 2026-07-20 · **Status:** working gate built; iterating to raise precision (Trial 9 training)

---

## 1. Executive summary

We set out to build **Layer 1** of the classifier — a gate that decides whether a Reddit comment is about **athlete mental health** (athletes × mental health). Overnight we went from a **completely broken model** (predicted the same thing for every input) to a **working gate that beats the honest baseline**, and we identified and started fixing its main weakness (precision).

- **Best clean result so far:** the two-model AND gate scores **F1 0.66** on a fresh held-out set (recall **0.95**, precision 0.51) vs. an always-positive baseline of F1 0.61.
- **The gate is high-recall:** it catches ~95% of real athlete-MH comments but currently lets through too many false positives.
- **Root cause found & being fixed:** our sample was built from mental-health-keyword matches, so it never contained *clean* mental-health negatives. Trial 9 adds them (from the corpus's `baseline` arm) to sharpen precision.

---

## 2. What we're building (context)

The full classifier is **layered**. This work is **Layer 1** only:

| Layer | Job |
|---|---|
| **Layer 1 (this)** | Is the comment about **athlete mental health**? (the gate) |
| Layer 2 (later) | *Which* mental-health issue (depression, anxiety, ED, burnout…) |
| Beyond | Emotion / help-seeking / peer-vs-official analyses |

Because Layer 1 is a **gate**, its errors cascade — low recall under-reports every downstream statistic, low precision inflates them. So we report precision **and** recall, and read F1 against an honest baseline, not against zero.

---

## 3. The core design decision: two dimensions, AND-ed

Judging "is this *sports* mental health?" as one label was inconsistent (human-vs-Claude agreement was only **κ = 0.38**). We switched to labeling **two clean, independent questions** and combining them:

- **`mh`** — Is the comment about anyone's mental health? (even in passing)
- **`sport`** — Is the comment about sport / training / athletic life?
- **Relevant = `mh` AND `sport`.**

Each question alone is unambiguous; the AND yields "athlete × mental health" without a rater holding both in their head. Keeping the two labels separate also lets us later swap to Fans × MH, etc., without re-labeling.

---

## 4. Data & labeling

- **Source corpus:** MS_comments 2018–2022, Reddit comments, 60 months. Two arms: `matched` (hit MH/sport keyword lists, 574,639 rows) and `baseline` (same sports subreddits, *no* MH keywords).
- **Sub-corpus (`athlete_mh_keyword_both`):** comments by authors who have **both** an `mh_core` comment and a sport comment — keyword-verified athletes who discuss mental health. 5,391 comments. *(We discovered the two keyword types never co-occur in the same comment, so athlete-ness is anchored at the author level.)*
- **Labeling:** all labels produced by **Claude (Sonnet) via parallel multi-agent workflows** (a shared rubric, forced structured output), then aggregated `mh AND sport`.

| Set | Size | Purpose |
|---|---|---|
| Initial sample | 1,200 (20/month) | first training set |
| Expansion | +1,317 | scale to break the base-rate ceiling |
| **Training total** | **~2,516** | mh 1619/897, sport 1732/784, relevant 1116/1400 (44% pos) |
| Baseline negatives | +1,020 | clean mh-negatives (Trial 9) |
| **Fresh held-out** | **292** | clean cross-model evaluation (never trained on) |

---

## 5. All trials (chronological)

| # | What | Data | Result | Takeaway |
|---|---|---|---|---|
| 1 | roberta-large, threshold 0.6, 1 epoch, AI weak labels | 1000 | predicted **0 relevant**, crashed | threshold too high + underfit; patched the crash |
| 2 | (eval of Trial 1) | — | **constant predictor** (P≈0.508 for all); fake F1 0.79 = baseline | model learned nothing |
| 3 | roberta-base, LR **5e-5**, weak labels | 1000 | loss stuck at 0.69 (= chance), killed | RoBERTa fine-tuning instability + weak labels |
| 4 | roberta-base, LR **2e-5**, **new 2-dim labels** | 1199 (AND) | F1 0.50, precision ~0.5 | **fixed the dead model**; but compound AND target too hard |
| 5 | `mh` dimension only | 1199 | F1 0.82 — **but base-rate inflated** (+0.02 real skill; caught 5/40 negatives) | decomposition helps the *number*; data volume is the ceiling |
| 6 | `sport` dimension | 1199 | *killed* | superseded by the data expansion |
| 7 | `mh` | **2516** | **real discrimination**: 53% specificity (vs 12% at 1199); F1 0.73–0.81 | **more data broke the base-rate trap** |
| 8 | `sport` | 2516 | weak — probs capped at 0.59, near base rate | corpus is *all* sport, so little signal to learn |
| — | **AND gate (contaminated split)** | 2516 | F1 0.71 | inflated — see held-out below |
| — | **AND gate (clean held-out)** | 292 | **F1 0.66**, recall 0.95, precision 0.51 | **honest number**; precision is the weakness |
| 9 | `mh` **+ 1020 baseline negatives** | 3536 (balanced) | *training now* | root-cause precision fix |

---

## 6. Key findings (the narrative)

1. **The first models were dead, not just bad.** Trials 1–3 produced a constant output — a training failure, fixed by lowering the learning rate to **2e-5** with warmup and using consistent labels. Lesson for the paper: report the failure and the fix.
2. **A single "sports mental health" label is too hard to learn (and to label).** The compound target gave F1 0.50 and low human agreement. **Decomposing into two dimensions** is both more learnable and more consistent.
3. **Beware base-rate-inflated F1.** On a 64%-positive class, a trivial "always yes" classifier already scores F1 0.80. We evaluate against that baseline; early per-dimension "wins" were mostly base rate, not skill.
4. **Data volume was the real ceiling.** Doubling labels (1,199 → 2,516) moved `mh` from "predicts positive for everything" (12% specificity) to genuine discrimination (53%).
5. **`sport` is intrinsically weak here.** The corpus is overwhelmingly sport-context, so there's little signal to separate — the `sport` classifier is near-degenerate, and the gate leans on `mh`.
6. **Re-splitting per trial contaminates cross-model evaluation.** We caught this: the split-based gate (0.71) was inflated; the **clean held-out** number is **0.66**. All future gate numbers use the fresh 292-comment set.
7. **The precision fix has a principled source.** Our sample is mental-health-keyword-matched, so it lacked clean negatives. The corpus's `baseline` arm *is* clean mental-health negatives — adding them (Trial 9) is exactly ISAAC's matched-vs-baseline design.

---

## 7. Current best result (honest)

**Clean held-out set, n = 292, 44% relevant (baseline F1 = 0.61):**

| model | precision | recall | F1 | accuracy |
|---|---|---|---|---|
| `mh` alone (vs mh truth) | 0.69 | 0.95 | 0.80 | 0.70 |
| `sport` alone (vs sport truth) | 0.70 | 0.92 | 0.80 | 0.69 |
| **AND gate (vs relevant)** | **0.51** | **0.95** | **0.66** | 0.58 |

**Interpretation:** a usable **high-recall gate** — it rarely drops a real athlete-MH comment (recall 0.95) — but it's **permissive** (precision 0.51: about half of what it flags is off-topic). For a Layer-1 gate feeding a downstream classifier, high recall is arguably the right bias, but the precision should improve.

---

## 8. In progress & next steps

- **Trial 9 (training now):** `mh` retrained with 1,020 clean baseline negatives (training set now balanced 46% positive). Expectation: fewer `mh` false positives → higher gate precision → gate F1 up from 0.66.
- **After Trial 9:** re-run the clean held-out eval. If the gate clears ~0.80, we're at a satisfactory Layer-1. If it plateaus, the next principled moves are: (a) improve `mh` label quality (currently single-pass Claude), (b) accept a high-recall gate and let Layer 2 + a threshold refine precision, or (c) reconsider whether `sport` needs a learned model at all given the corpus is already athlete-context.

---

## 9. Open questions for discussion

- **What's the target?** Is a **high-recall gate** (catch everything, refine later) acceptable for Layer 1, or do we need balanced precision/recall (F1 ≥ 0.80)?
- **Does `sport` need to be a learned classifier** if the corpus is already all athlete/fitness subreddits? Could subreddit membership + a light filter replace it?
- **Label quality vs. quantity:** labels are single-pass Claude. Would a second independent rater (human or a second model) + adjudication raise the ceiling more than more data?
- **How much more labeling is worth it?** Each doubling of data costs a multi-hour CPU training run; where are the diminishing returns?

---

## 10. Reproducibility

- Trials log: `experiments/trials_log.md` · Methods & provenance: `docs/relevance_layer1_methods.md`
- Data: `data/data_relevance_ratings/comments/` (rated files, `sports_mh_dimensions.csv`, `holdout_labeled.csv`)
- Models: `models/filter_relevance_{mh,sport,mental_health}/` · Splits: `models/train_relevance_data_split/`
- Code: `code/train_relevance.py` (trainer), `code/eval_thresholds.py`, `code/eval_and_gate.py`, `code/eval_holdout_gate.py`; run logs in `run_logs/`
- Seeds fixed (`set_seed(1)`); labeling model `claude-sonnet-5`.
