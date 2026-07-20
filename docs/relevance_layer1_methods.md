# Layer-1 Relevance Classifier — Methods & Provenance Log

*Athlete mental-health Reddit corpus (Mentality Sports / ISAAC-style pipeline). This document is the paper-facing record of how the Layer-1 relevance classifier was built: data, labeling methodology, model, every trial, decisions, and limitations. Keep it updated as results land.*

Last updated: 2026-07-20.

---

## 1. Objective
The full classifier is **layered**. **Layer 1** (this document) is a binary gate: *is a comment about **athlete mental health** (athletes × mental health)?* It filters the keyword-matched corpus down to genuinely relevant comments before all downstream analysis (Layer 2 = which MH issue; plus emotion / help-seeking / peer-vs-official). Because it is a gate, its precision and recall cap the quality of every downstream result, so both are reported.

## 2. Source corpus
- **MS_comments_2018_2022**: monthly Reddit **comment** CSVs, Jan 2018 – Dec 2022 (**60 months**), in two arms:
  - `matched` — comments that hit the MH/sport keyword lists (**574,639** rows).
  - `baseline` — comments from the same sports subreddits **without** MH keywords (contrast set for discourse comparison).
- Columns: `id, parent id, text, author, time, subreddit, score, matched patterns, source_row`. The `matched patterns` column records which keyword set fired (`mh_core`, `mh_crisis`, `sport_athlete`, `sport_basketball`, `baseline`).
- **Keyword taxonomy** (`Data_Info.docx`): disorder-anchored MH terms follow the IOC consensus statement (Reardon et al., 2019) and Gouttebarge et al. (2019) prevalence meta-analysis; symptom vocabulary mapped from DSM-5-TR (APA, 2022) and the nine SCL-90-R (Derogatis, 1994) dimensions, translated to lay/Reddit phrasing. Built for **recall**; precision is intended to be restored by the learned classifier's off-topic class.

## 3. Sub-corpus and annotation sample
- **Empirical finding:** `mh_core` and `sport_athlete` keywords co-occur in the **same comment 0 times** across all 574,639 matched rows (short comments rarely contain both a MH phrase and an athlete-identity phrase). So "both keywords in one comment" yields an empty set; athlete-ness must be anchored differently.
- **`athlete_mh_keyword_both`** sub-corpus: comments by authors who have **≥1 `mh_core` comment AND ≥1 `sport_athlete`/`sport_basketball` comment** somewhere (keyword-verified athletes who also discuss MH); their `mh_core` comments = **5,391** across 60 months (per-month min 40, median 89).
- **Annotation sample:** stratified random, **20 comments/month × 60 months = 1,200** (seed = 1). Comments carry blinded integer `random_id`s; a separate key file (`relevance_sample_athlete_key.csv`) maps `random_id` → subreddit/author/time so raters judge text alone, unbiased by source.

## 4. Labeling methodology *(core methods contribution)*
Judging "is this *sports* mental health?" as a single label proved inconsistent (see §6, κ = 0.38). We instead label **two independent binary dimensions** and aggregate:

- **Dimension `mh` — mental health?** `1` if the comment refers to or implies anything about a person's mental health (depression, anxiety, panic, eating disorder / body dysmorphia, psychological burnout, trauma/PTSD, OCD, ADHD, addiction, insomnia, suicidality, therapy/counseling/psychiatry, or the phrase "mental health") — even in passing, as advice, or about a third person. `0` for non-clinical keyword use ("not eating too much pizza"), pure nutrition/diet, physical injury/rehab only, pure game talk, figurative/joke use ("ptsd from double rims"), or bot/automod boilerplate.
- **Dimension `sport` — sport / athletic life?** `1` if the **comment content** is about sport, exercise, training, competition, fitness, coaches/teammates, sport injury, or being an athlete. `0` if it is about non-sport life (a cashier interaction, a job, relationships) **even when the author is an athlete** — the comment, not the author, is judged.
- **Relevant = `mh` AND `sport`.** Keeping the two labels separately makes the pipeline reconfigurable (e.g. Fans × MH, Athletes × injury) without re-labeling.

**Procedure.** All 1,200 comments were labeled by **Claude (model: `claude-sonnet-5`)** via a **24-agent parallel workflow** (50 comments/agent, one shared rubric, forced structured output). **1,199/1,200** labeled (agent for `random_id` 352935 dropped it; excluded from training). Rubric and script archived (see §9).

**Label distribution (n = 1,199):** `mh` = 766, `sport` = 860, **relevant (mh∧sport) = 549 (46%)**, irrelevant = 650 (54%). This near-balanced split replaced an earlier weak-labeled set (see §6).

**Human validation.** Against **98** hand labels (annotator: Naveen), agreement with the `mh` dimension = 71%, with the aggregated `relevant` label = 70%; **14** comments the human had marked relevant were correctly reclassified to 0 as *mental-health-but-not-sport*. NOTE: the hand labels were produced under a shifting single-label standard (partly "any MH", partly "sports MH"), so they are a **noisy reference**, not gold — this motivated the two-dimension redesign.

## 5. Classifier
- **Model:** RoBERTa-base + sequence-classification head, `max_length` 512. (roberta-large was tried and abandoned — see §6.)
- **Split:** 80/10/10 train/validation/test (seed = 1), persisted to `models/train_relevance_data_split/`.
- **Loss:** class-weighted cross-entropy, weights = `sklearn compute_class_weight("balanced")`; an optional asymmetric confusion penalty exists but is unused.
- **Optimization:** learning rate **2e-5**, **warmup 60 steps**, weight decay 0.01, batch size 8, **3 epochs**, `load_best_model_at_end` on validation loss.
- **Decision rule:** **argmax (thresholding OFF)** for the first evaluation; a probability threshold is added only if precision is insufficient, and (per ISAAC convention) is set per-group at deploy time in `filter_relevance.py`.
- **Deployment:** the trained model is applied to the full corpus via `code/filter_relevance.py -g mental_health`, keeping only comments predicted relevant — this is the Layer-1 gate at scale.

## 6. Trials (reproducibility & failure record)
See `experiments/trials_log.md` for the running table. Summary:

| Trial | Model / config | Labels | Outcome | Cause / action |
|---|---|---|---|---|
| 1 | roberta-large, thr 0.6, 1 epoch | Codex AI weak labels, 1000 (620/380), "central" rubric | Predicted **0 relevant**; F1 crashed | Threshold too high; underfit; `f1_calculator` divided by zero → **patched** |
| 2 | eval of Trial-1 model, thresholds swept | — | Model is a **constant predictor** (P≈0.508 for every input); apparent 0.79 F1 = majority baseline | Not a threshold issue — model never learned |
| 3 | roberta-base, 3 ep, **LR 5e-5** | Codex weak labels | Loss stuck at **~0.69 (= chance)** through step 120; **killed** | Suspected RoBERTa fine-tuning instability at 5e-5 and/or weak/noisy labels |
| 4 | roberta-base, 3 ep, **LR 2e-5 + warmup 60** | two-dimension Claude labels, 1199 (549/650) | **Learned but weak**: probs spread 0.19–0.57, eval_loss 0.704→0.678→0.670; test F1 0.50 (argmax) / 0.64 (thr 0.35), **precision ~0.5 at all thresholds**, acc ≈ majority baseline | LR + balanced labels fixed the *dead-model* problem, but the compound `mh∧sport` target is too hard on 959 examples → **decompose** into per-dimension models |
| 5 | roberta-base, `mh` dimension only, 3 ep | 1199 (mh 766/433) | F1 0.82 (argmax) — **but base-rate-inflated**: predicts positive 115/120, TN only 5/40; always-positive baseline F1=0.80, so +0.02 real skill | Decomposition raises the *number* but the model is near base rate → **data volume is the ceiling** |
| 6 | `sport` dimension, 1199 | 1199 (sport 860/339) | *killed* | superseded by the data expansion before completing |
| 7+ | per-dimension (`mh`, `sport`) on **expanded 2516** | 1199 + 1317 workflow-labeled expansion (mh 1619/897, sport 1732/784, relevant 1116/1400) | *in progress* | Test whether doubling the labels breaks the base-rate ceiling; evaluate the AND-gate honestly vs the always-positive baseline |

**Methodological note for the paper:** per-dimension F1 must be read against the **base-rate (always-positive) baseline**, not zero — the `mh`/`sport` dimensions have 64–72% positive rates, so a trivial classifier already scores F1 0.78–0.82. The lower-base-rate AND target (44% positive) is the more honest headline metric. All trials so far sit near their base-rate baselines on ~959 examples, motivating the expansion to 2516.

## 7. Key decisions and rationale
1. **Author-level "both tags"** for the sub-corpus, because MH and sport keywords never co-occur in one comment (§3).
2. **Two-dimension labeling** instead of one "sports mental health" label — the compound judgment gave low inter-rater agreement (κ = 0.38) and a boundary the model could not learn; the two dimensions are each unambiguous.
3. **Train on the natural ~46/54 (earlier ~62/38) class split, not a forced 90/10.** The model needs negatives to learn the boundary; 90/10 is the *output* precision target of the gate, achieved by the decision threshold, not by deleting training negatives. Keyword-only filtering was tested and caps at ~66% precision (safe exclusions of bots/physical-therapy/nutrition moved 62%→66%), because the remaining false positives contain genuine MH keywords used incidentally — separating those requires a semantic classifier.
4. **roberta-base over roberta-large** — the larger model with limited CPU training collapsed to a constant output (Trials 1–2).

## 8. Limitations / open items
- **Labels are currently single-rater (Claude).** Independent human labels + **Cohen's Kappa ≥ 0.6** on a blind subset are still needed to validate them (use `code/metrics_interrater.py -g mental_health -n 2`).
- Human hand labels (n = 98) were internally inconsistent → weak validation reference.
- One comment (`352935`) dropped by its labeling agent.
- Sampling used a custom month-stratified draw. ISAAC's `filter_sample.py` is **year**-stratified; the `isaac-data-loader` offers true month-stratified sampling — use it for temporal-CI-grade samples.
- No `keywords/mental_health_*.txt` or `athlete_*.txt` files yet, so `filter_keywords` / `filter_keywords_adv` (the recall and advanced-precision stages) are not yet wired for this task.
- Trained on **CPU** (no GPU) → slow; hyperparameters chosen partly for CPU feasibility.

## 9. Provenance / file map
- Sub-corpus: `data/athlete_mh_keyword_both/RC_YYYY-MM.csv` (60 files).
- Sample + key: `data/data_relevance_ratings/comments/relevance_sample_athlete_{0,1}_rated.csv`, `..._key.csv`.
- Two-dimension labels: `data/data_relevance_ratings/comments/sports_mh_dimensions.csv` (`random_id, mh, sport, relevant`).
- Training files: `data/data_relevance_ratings/comments/relevance_sample_mental_health_{0,1}_rated.csv` (relevant = mh∧sport).
- Labeling workflow script: `.claude/.../workflows/scripts/label-sports-mh-wf_86f9852b-df5.js` (rubric embedded).
- Trainer: `code/train_relevance.py` (group `mental_health` → roberta-base; LR 2e-5, warmup 60). Split: `models/train_relevance_data_split/`. Model: `models/filter_relevance_mental_health/`.
- Eval + logs: `run_logs/train_mh_trial*.log`; threshold-sweep eval described in Trial 2.
- Human hand labels: `Downloads/MS_comments_2018_2022/hand-matched-data - hand-label.csv`.
- Reproducibility: `set_seed(1)` throughout; labeling model `claude-sonnet-5`.

## 10. Software
Python 3.11 (`.venv`); `transformers`, `torch` (CPU), `scikit-learn`. Pipeline = ISAAC / Illinois_Social_Attitudes (`code/`), CLI-driven (`python code/cli.py -r <stage> -g mental_health -t comments`).
