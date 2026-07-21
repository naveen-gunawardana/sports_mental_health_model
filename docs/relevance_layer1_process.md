# Layer-1 Relevance Classifier — Process & Diagnostic Log

*A running, paper-facing narrative of how the Layer-1 gate was built, where it got stuck, how we
diagnosed the ceiling, and how we broke it. Written to record the **reasoning and the failures**,
not just the final number (BRM-style methods reporting). Newest developments appended.*

Companion files: `experiments/trials_log.md` (one row per run), `docs/relevance_layer1_methods.md`
(provenance), `docs/OVERNIGHT_REPORT.md` (exec summary).

---

## 0. What Layer 1 is

A binary **gate**: is a Reddit comment about **athlete mental health** (athletes × mental health)?
It is the first layer of a layered classifier; its errors cascade into every downstream statistic,
so we report **precision *and* recall** and read F1 against an honest baseline, not against zero.

**Design decision (unchanged, and validated):** judging "is this *sports* mental health?" as one
label was inconsistent (human-vs-Claude κ = 0.38). We split it into two clean, independent
questions and AND them:

- `mh` — is the comment about anyone's mental health?
- `sport` — is the comment about sport / training / athletic life?
- **relevant = mh AND sport.**

Each question alone is unambiguous; the AND yields "athlete × mental health" without a rater
holding both in their head.

---

## 1. The wall: gate stuck at F1 ≈ 0.68

After the model was working (see trials 1–11 for the dead-model fixes, data expansion, and LR
tuning), the two-model AND gate plateaued at **F1 ≈ 0.66–0.68** on the fresh 292-comment held-out
set (44% relevant, always-yes baseline F1 = 0.61). High recall (~0.95), low precision (~0.55).
Babak's target is **F1 0.8 across all metrics.** So something was capping it, and we set out to
find *what*, by elimination.

---

## 2. Ruling out the usual suspects (four independent negative results)

We treated "why is the gate stuck?" as a diagnosis problem and killed one hypothesis at a time.

| # | Hypothesis: "the ceiling is…" | Test | Result | Verdict |
|---|---|---|---|---|
| 1 | **too little data** | 1199 → 2516 labels; also tried ~1000 | more data helped up to a point; ~1000 *undertrains*, but 2516 → more didn't move the ceiling | ❌ not it |
| 2 | **noisy labels** | 2 independent labeling passes + opus adjudication; measured agreement | κ = **0.81 (mh) / 0.86 (sport)** — labels already reliable. Retraining on "cleaner" adjudicated labels made the gate **worse** (0.55) | ❌ not it |
| 3 | **the gate architecture (hard AND)** | replaced hard AND with a **learned logistic meta-classifier** over `[p_mh, p_sport, p_mh·p_sport]`, scored honestly by 5-fold out-of-fold | tied the hard AND **exactly** (F1 0.66 vs 0.66) | ❌ not it |
| 4 | **too little model capacity** | swapped `mh` roberta-base → **roberta-large** (3× params); fixed the batch-2 cold-start collapse with gradient accumulation (eff. batch 16) + warmup | val F1 fine (0.81), but held-out **P(mh)max 0.57 → 0.59** and gate F1 0.66 — no better | ❌ not it |

### The smoking gun

The stacking experiment (test 3) surfaced the real bottleneck in one number:

> **P(mh)max = 0.57** — across all 292 held-out comments, the mh model's *single highest*
> confidence was 0.57. It was **never confident about anything.**

When a model's probabilities top out that low, **no threshold and no combiner can produce
precision** — there is nothing sharp to threshold. That reframed the whole problem: the ceiling was
not data, labels, or gate logic. It was **the mh model's inability to separate the signal.**

And test 4 showed that throwing *raw capacity* at it (roberta-large, 3× params) barely moved
P(mh)max (0.57 → 0.59). So the fix wasn't "a bigger model."

---

## 3. The breakthrough: domain match, not size

Why would a 355M-parameter model still be under-confident on a signal? Because it was pretrained on
the **wrong kind of text**. roberta-base/large are pretrained on **books + Wikipedia** — formal,
edited prose. Our data is **informal Reddit comments**: slang, abbreviations, emoji, run-on
venting. The model had never really "seen" language like this, so it couldn't read the subtle
mental-health cues buried in long, rambling comments.

**Fix:** swap in a model pretrained on the *right* domain —
[`cardiffnlp/twitter-roberta-base`](https://huggingface.co/cardiffnlp/twitter-roberta-base),
a roberta-base pretrained on ~58M tweets. Same size, same speed (~50 s/epoch), but its priors match
informal social-media text. (We'd wanted a *mental-health*-pretrained model,
`mental/mental-roberta-base`, but that repo is gated/401.)

### Result (held-out gate, n = 292, 44% relevant)

| metric | roberta-base | roberta-large | **twitter-roberta** |
|---|---|---|---|
| val F1 (mh dim) | 0.81 | 0.81 | **0.96** |
| **P(mh)max** | 0.57 | 0.59 | **1.00** |
| gate precision | 0.55 | 0.54 | **0.74** |
| gate recall | 0.87 | 0.87 | 0.86 |
| **gate F1** | 0.68 | 0.66 | **0.79** |
| gate accuracy | ~0.62 | ~0.61 | **0.80** |

**Gate F1 0.68 → 0.79** (≈ Babak's 0.8 target), and the under-confidence ceiling
(**P(mh)max 0.57 → 1.00**) is gone entirely. **Domain-adaptive pretraining beat 3× raw capacity by
+0.13 F1.** That is the headline finding.

---

## 4. Lessons worth keeping (for the paper)

1. **Diagnose before you scale.** The instinct "stuck → get more data / a bigger model" was wrong
   twice (tests 1 and 4). The cheap diagnostic — *look at the max probability the model ever emits*
   — pointed straight at the real problem.
2. **P(max) is a cheap, powerful ceiling diagnostic.** If a classifier's most-confident prediction
   is barely above the threshold, the problem is representation, not the decision rule.
3. **Domain match > model size** for informal-text classification. A base-sized in-domain model beat
   a 3×-larger out-of-domain one, at a fraction of the compute (~50 s/epoch vs ~57 min/epoch on a
   6 GB card).
4. **Report the failures.** The four negative results (data, labels, gate, capacity) are what make
   the domain-match result credible — they show it wasn't luck.

---

## 5. Applying the fix to `sport` too → final gate

`sport` had the same disease (roberta-base, P(sport)max stuck ~0.68 — the permissive half of the
gate). Same cure: retrain it with `cardiffnlp/twitter-roberta-base`. Sport val F1 **0.80 → 0.96**,
**P(sport)max 0.68 → 1.00.**

### Final Layer-1 gate (held-out, n = 292, both models domain-matched)

| | precision | recall | F1 | accuracy |
|---|---|---|---|---|
| baseline (always-yes) | 0.44 | 1.00 | 0.61 | 0.44 |
| roberta-base gate (old) | 0.55 | 0.87 | 0.68 | 0.62 |
| **twitter-roberta gate (final)** | **0.91** | **0.92** | **0.92** | **0.93** |

**Gate F1 0.68 → 0.92 — clears Babak's 0.8 target on every metric.** Both models now emit
P(max) = 1.00 (fully confident), and the gate is **flat at 0.91–0.92 across the entire threshold
grid** — the operating point is robust, not a fragile knife-edge. Operating point: mh ≥ 0.45,
sport ≥ 0.40. **Layer 1 is done.**

## 6b. Scope change — broadening `mh` to include performance psychology

After the 0.92 gate was built, a demo surfaced a scope question: comments like *"I don't play well
when my parents are at the game"* or *"I have confidence issues shooting"* were classified **not**
mental health. The old rubric treated sports/performance psychology as skill talk, not mental
health. **Research decision:** for an athlete population, performance struggles (confidence, nerves,
choking, pressure/expectations) are how many athletes first experience mental-health difficulty and
can signal a larger issue — so they should count. `mh` was widened (see `docs/mh_rubric_broad.md`).

**Process:** relabeled all 2,808 comments (train + held-out) with the broadened rubric via a
47-agent labeling workflow, then combined by **union** (mh = old-positive OR new-positive) — a
strict *broadening* that adds the new performance-psych positives but never drops an existing mh
label (QA showed the from-scratch relabel added noise in both directions; union protects against
losing real cases like "insomnia"/"anxious"). Train mh-positive 1,619 → 1,807. Retrained the
twitter-roberta mh model; held-out is now 47% relevant.

**Result:** gate **F1 0.89** (precision 0.91, recall 0.87) — still clears 0.8 on all metrics, down
slightly from the narrow 0.92 because the broadened task is genuinely harder.

**Honest limitation (accepted):** the model catches performance anxiety when there is a real
emotional/distress signal (*"pressure from my parents… so anxious I freeze"* → fires), but **bare**
confidence statements with no distress word (*"I have confidence issues shooting"*) still read as
neutral. Two causes: such terse phrases are out-of-distribution vs. the longer real comments the
model trained on (recall on actual corpus data is 0.87), and "confidence" is genuinely ambiguous
(it appears in positive talk too). Making bare-confidence fire would need targeted augmentation at
the risk of over-flagging; we chose to accept the current behavior.

## 6. Bottom line for the paper

The entire ~0.68 → 0.92 jump came from **one change: matching the pretraining domain** (formal
books/Wikipedia → informal social media). It was invisible until we stopped asking "how do we tune
the gate?" and started asking "why is the model never confident?" — the P(max) diagnostic. Four
expensive levers (more data, cleaner labels, a learned combiner, 3× model capacity) each moved the
number by ~0. A same-sized, same-speed, in-domain model moved it by +0.24 F1.

### Reproducibility
- Model swap lives in `code/train_relevance.py` (the `group in ("mh","sport")` branch selects
  `cardiffnlp/twitter-roberta-base`; loaded via `Auto*` classes so any HF checkpoint works).
- Held-out eval + P(·)max: `code/eval_holdout_grid.py` (writes `run_logs/gate_status.txt`).
- Stacking experiment: `code/eval_holdout_stack.py`. Seeds fixed (`set_seed(1)`).
- Deliverable models backed up: `models/filter_relevance_mh_twitter_backup` (winning mh),
  `models/filter_relevance_{mh,sport}_base_backup` (previous roberta-base).
