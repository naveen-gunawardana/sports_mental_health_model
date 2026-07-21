# Layer-1 Relevance Classifier — Progress Report

**Date:** 2026-07-20 · **Status:** ✅ **DONE.** Narrow gate F1 0.92; **current deliverable is the
broadened + augmented gate at F1 0.89** (P 0.87 / R 0.92; mh now includes sports/performance
psychology and correctly fires on it). Both clear the 0.8 target on all metrics. *(An earlier
"broadened" run was invalidated by a split-cache bug — see process doc §6c; now fixed.)*

> Full diagnostic narrative (ceiling → breakthrough → scope broadening): `docs/relevance_layer1_process.md`.
> Broadened `mh` rubric: `docs/mh_rubric_broad.md`.

---

## 1. Executive summary

We built **Layer 1** of the classifier — a gate that decides whether a Reddit comment is about
**athlete mental health** (athletes × mental health). We went from a **completely broken model**
(same output for every input) → a working-but-capped gate (F1 0.68) → a **final gate at F1 0.92**
that clears Babak's 0.8 target on every metric.

- **Final result (held-out, n=292, 44% relevant):** two-model AND gate scores **F1 0.92, precision
  0.91, recall 0.92, accuracy 0.93** vs. an always-positive baseline of F1 0.61.
- **What broke the ceiling:** the gate was stuck at F1 0.68 because both underlying models were
  chronically **under-confident** (max probability only ~0.57). We ruled out four expensive causes
  (more data, cleaner labels, a learned combiner, 3× model capacity) — each moved the number by ~0.
  The fix was **domain match**: swapping roberta-base (pretrained on books/Wikipedia) for
  `cardiffnlp/twitter-roberta-base` (pretrained on informal social-media text ≈ Reddit). Max
  confidence jumped 0.57 → **1.00** and gate F1 0.68 → **0.92**.
- **The operating point is robust:** the gate holds at 0.91–0.92 across the entire threshold grid,
  not a fragile knife-edge.

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

### Metrics table (precision / recall / F1)

| Trial | Model (data) | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 1 | mh, roberta-large, weak labels (1000) | 0.6 | 0.00 | 0.00 | 0.00 |
| 2 | (dead-model eval) | argmax | 0.65 | 1.00 | 0.79 ⚠️ |
| 3 | mh, LR 5e-5 (1000) | — | *killed — never learned* | | |
| 4 | **AND** (mh∧sport, 1199) | argmax | 0.48 | 0.54 | 0.50 |
| 4 | **AND** (1199) | 0.35 | 0.51 | 0.87 | 0.64 |
| 5 | mh (1199) | argmax | 0.70 | 1.00 | 0.82 ⚠️ |
| 6 | sport (1199) | — | *killed — superseded* | | |
| 7 | mh (2516) | 0.6 | 0.74 | 0.73 | 0.73 |
| 7 | mh (2516) | 0.4 | 0.68 | 0.99 | 0.81 |
| 8 | sport (2516) | argmax | 0.70 | 0.92 | 0.80 ⚠️ |
| — | AND gate — split (contaminated) | mh.55/sp.50 | 0.57 | 0.93 | 0.71 |
| — | mh alone — clean held-out | argmax | 0.69 | 0.95 | 0.80 |
| — | sport alone — clean held-out | argmax | 0.70 | 0.92 | 0.80 |
| — | **AND gate — clean held-out ★** | mh.50/sp.40 | **0.51** | **0.95** | **0.66** |
| 9 | mh + baseline negatives (3536) | — | *training now* | | |

**★ = honest headline** (the clean held-out AND gate). **⚠️ = base-rate inflated** — that F1 is near what a trivial "always yes" classifier scores on a mostly-positive class, so it is not real skill. Note the pattern across all rows: **recall is consistently high, precision is the weak column** — the models catch most relevant comments but over-flag. Trial 9 targets that precision column.

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

**Clean held-out set, n = 292, 44% relevant (baseline F1 = 0.61).** Both models retrained per Babak's config (F1 early-stopping per epoch, patience 2, class-weighted, GPU):

| model | precision | recall | F1 |
|---|---|---|---|
| `mh` model | 0.70 | 0.82 | 0.76 |
| `sport` model | 0.76 | 0.85 | 0.80 |
| **AND gate** (balanced, mh.5/sp.5) | **0.55** | 0.82 | **0.66** |
| AND gate (precision-favored, mh.65) | 0.62 | 0.42 | 0.50 |

Both models now **discriminate** (the old `sport` was permissive; now P 0.76). Gate precision rose 0.51 → 0.55 and it rejects **79 of 164 negatives** (was 47). But the **threshold frontier tops out at F1 0.66** — pushing precision to ~0.62 crashes recall to 0.42. **No operating point reaches precision *and* recall ≥ 0.8** (Babak's goal).

### Error analysis (gate: 85 FP, 23 FN)
- **FP causes split evenly** — `mh` model wrong 51, `sport` model wrong 44 (both 10). Neither model is the sole culprit.
- **Errors cluster in the low-confidence band** (P 0.40–0.65); max confidence is only P(mh)=0.73, P(sport)=0.58 — the models are *uncertain exactly on the hard cases*.
- **Many FPs are genuine label ambiguity** — model disagrees with the label on borderline cases ("got a puppy… so overwhelmed"; "7 pints of IPA is binge drinking"; "anxiety about needing the bathroom").
- **FNs are real relevant, narrowly missed** (probabilities just under 0.5).

**Conclusion:** the precision ceiling is **label ambiguity + low model confidence, not data volume** (2,500 is enough, as Babak said). Next lever is **label quality** — two independent labeling passes + adjudication of the borderline disagreements — not more data.

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
