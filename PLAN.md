# AM Relevance Classifier — Plan

**Goal (by next meeting):** Train a relevance classifier that decides **athlete mental-health relevant vs. not** — i.e., is a comment about **athletes AND mental health**.

## Where this fits — this is Layer 1
The full classifier is layered; we are building **Layer 1**.
- **Layer 1 — Relevance (now):** athlete × mental health, yes/no. The **gate** that filters the noisy keyword-matched corpus down to genuinely relevant comments.
- **Layer 2 — Issue type (later):** which mental-health issue (depression, anxiety, eating disorder, burnout, …) among what Layer 1 passes.
- **Beyond:** emotion / help-seeking / peer-vs-official classifiers — all run on Layer 1's output.

**Consequence:** Layer 1 is a gate, so its errors cascade — low recall under-reports every downstream count, low precision inflates them. That's why we report **precision *and* recall** and don't blindly over-filter.

---

## Core design: two relevance dimensions, aggregated
Instead of one fuzzy "athlete mental health" judgment, label each comment on **two separate questions** (two separate Claude prompts):

- **A — Mental health:** *Is this about anyone's mental health?* (any mention, even in passing)
- **B — Athlete:** *Is this about an athlete / their athletic life?*

- **Final relevance = A AND B** (athlete × mental health).
- Keep both labels separately, so later you can swap to **Fans × MH**, **Athletes × physical injury**, etc. **without re-labeling.**
- **Focus now: Athletes × Mental Health.**

---

## Phase 0 — Unblock the current model (quick)
- The trained model predicts 0 relevant *only* because `threshold = 0.6` is too high — **turn thresholding OFF (argmax)** and re-evaluate to see the real precision / recall / F1.
- Patch `f1_calculator` so it doesn't crash when the model predicts 0 positives.

## Phase 1 — Labels: scale to ~1500
- **Use Claude to label ~1500 comments** via the two prompts (A = mental health, B = athlete), then aggregate.
- **Hand-label a subset yourself** — gold standard + agreement check.
- **Inter-rater agreement = Cohen's Kappa** (Claude vs. you); already scripted in `code/metrics_interrater.py`. Target **≥ 0.6**.
- **Stratify the 1500 evenly by month** so each time bin has enough rows for tight 95% CIs in the temporal analysis.

## Phase 2 — Add sports-word filtering
- Add **sport/athlete keyword filtering** to the candidate pipeline so we're scoring athlete-context comments.
- We already have `sport_athlete` / `sport_basketball` tags + subreddit anchor in the data — reuse them.

## Phase 3 — Train
- **Train on the current ~65/35 split** (relevant / irrelevant). **No need to force 90/10** in the training data.
- **Class weighting:** upweight the **minority class (irrelevant, ~1/3)** so getting an irrelevant one wrong costs more. `compute_class_weight("balanced")` in the script already does this; can push harder via `custom_training` / `penalize_confusion`.
- **One classifier per dimension** (an MH model and an athlete model) recommended — matches the two-prompt labels and keeps the pipeline swappable. (One combined model is the alternative.)

## Phase 4 — Evaluate (threshold OFF first)
- **Report precision / recall / F1 for BOTH models with NO threshold first.**
- Tradeoff to remember: **raise threshold → precision ↑, recall ↓** (fewer false positives, but you miss real ones).
- **Only if precision is too low, add a threshold** and re-report.

## Phase 5 — Temporal / statistical analysis
- Apply the models to the corpus → relevance-labeled data over time.
- **Regression models** on the resulting time series.
- **Confidence intervals:** confirm the sample size gives good **95% CIs** per time bin — this is *why* the 1500 must be month-stratified.

## Phase 6 — Records & version control
- **Git:** the repo is now git-tracked — **commit after each meaningful change** (data regen, new labels, each trial). Consider tagging each trial.
- **Trials log:** record every run — data version, params (threshold, class weights, epochs), and metrics — in `experiments/trials_log.md`. Keep it editable; add notes per run.
- **Take notes on everything.**

---

## Open decisions to confirm with Babak
- One combined classifier vs. two per-dimension classifiers.
- Exact split of the 1500 between Claude-labeled and hand-labeled.
- Which sport keyword set / subreddits define "athlete" for Dimension B.
