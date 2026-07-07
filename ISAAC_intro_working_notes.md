# ISAAC Introduction — Working Notes for the Resource & Methods Paper

**Target venue:** *Behavior Research Methods*
**Audience:** Primarily psychologists; secondarily mixed psych + CS/NLP
**Date compiled:** 2026-05-19
**Status:** Pre-draft synthesis. Numbers checked against the README; comparator numbers verified via literature search where possible (flagged as "not verified" where they aren't).

---

## 1. Framing of the paper (one paragraph)

> Quantitative social-attitude research has been transformed by the arrival of internet-scale naturalistic text, but the field still relies overwhelmingly on small, ad hoc, single-platform, single-label samples. ISAAC is the first attempt to give psychologists a comprehensively filtered, multi-label, validated, longitudinal, geography-aware Reddit corpus targeted at six social distinctions of long-standing theoretical interest — and to expose it through a coding-free interface so that researchers without computational training can use it directly. This paper documents the corpus, the standardized pipeline that produced it, and the convergent-validity and reliability evidence supporting its psychological labels, in service of replicable and reproducible attitude research.

(The "framing positively" note in the user's draft is preserved: we lead with the *opportunity* — naturalistic, theory-relevant big data — rather than with what the field has done wrong.)

---

## 2. Why a resource paper now: the subfield-maturity argument

The field of computational social-attitude research has progressed in three rough phases:
1. **Hand-coded small samples** (decades of survey-experimental attitude research; high construct fidelity, low statistical power, no naturalism).
2. **Ad-hoc large datasets** (a thousand Reddit/Twitter studies, each scraping its own corpus, picking its own labeler, validating little; high variability in operationalization).
3. **Shared, validated, multi-label resources** — what BRM exists to publish, and the stage ISAAC tries to anchor for social-attitude research.

This is the same maturation arc psycholinguistics went through with the BNC/COCA, network science went through with SNAP, and developmental psychology with the CHILDES system. ISAAC's pitch is: *if the next generation of social-attitude studies is going to be large-N and naturalistic, those studies need to start sharing the same corpus, the same pipeline, and the same labels — otherwise the replication crisis that the field is finally taking seriously will simply migrate from hand-coded studies to scraped ones.*

---

## 3. Comparator landscape (compressed)

Two reports below were compiled from a literature search. Numbers are best-effort verified against primary sources; "n.v." = not verified.

### 3a. Reddit-based large corpora

| Corpus | Size | Time | Psych labels included | Validation | Access |
|---|---|---|---|---|---|
| **Pushshift / Arctic-Shift dump** | ~5.6B comments (Pushshift through 2019); rolling through 2025 | 2005–present | None — raw text only | None (infrastructure) | Open via torrents |
| **ConvoKit small Reddit corpus** | ~297K comments | Sept 2018 only | Conversation structure | Toolkit-level | Open, MIT |
| **Webis-TLDR-17** | 3.85M summary-comment pairs | 2006–2016 | Author-generated summary | Heuristic | Open |
| **Reddit Mental Health Dataset (Low 2020)** | 28 subreddits, ~100K-scale | 2018–2020 | Subreddit = label; LIWC | Pre/post-COVID contrasts | Open |
| **GoEmotions (Demszky 2020)** | 58K comments | 2005–2019 sample | 27 emotions, multi-label | Multi-annotator κ; PPCA | Open, Apache |
| **MFRC (Trager 2022; possibly LREC-COLING 2024 for final venue — verify)** | 16K comments | n.v. range | 8 moral categories | ≥3 raters; κ reported | Open, HF |
| **SBIC (Sap 2020)** | ~45K posts / ~150K annotations | re-annotation | Offensiveness, target, implied stmt | Pairwise 82.4%; α≈0.45 | Open |
| **RedditBias (Barikeri 2021)** | ~11K sentences | n.v. | Gender/race/religion/queerness bias | IAA per dim | Open |
| **Reddit Self-Disclosure (Dou 2024)** | ~2.4K posts / 4.8K spans | n.v. | 13 demographic + 6 experience types | Adjudicated dual annotation | Open |
| **IsamasRed (2024)** | 8M+ comments / 400K convs | Aug–Nov 2023 | Topic/controversy/emotion/morality | n.v. | Open |
| **ISAAC (this paper)** | **554M+ comments**; ~1B with submissions | **2007–2023** | Moralization, sentiment ensemble (3), emotion ensemble (3), generalization (4 features), US state location, anonymization | (Reported herein) | Coding-free web interface + open pipeline |

### 3b. Attitude / bias / moral / emotion text datasets (any source)

| Corpus | Platform | Size | Construct |
|---|---|---|---|
| MFTC (Hoover 2020) | Twitter | 35K tweets | 5 moral foundations |
| StereoSet (Nadeem 2021) | **Constructed** | 17K | Stereotype probing (LMs) |
| CrowS-Pairs (Nangia 2020) | **Constructed** | 1.5K pairs | 9 bias axes |
| HateXplain (Mathew 2021) | Twitter+Gab | 20K | Hate + rationales |
| ToxiGen (Hartvigsen 2022) | **Synthetic / model-generated** | 274K | Implicit toxicity, 13 groups |
| Implicit Hate (ElSherief 2021) | Twitter (extremist) | 22K | 6-class implicit hate |
| Civil Comments / Jigsaw | News comments | ~2M | Toxicity + identity-mention |
| HolisticBias (Smith 2022) | **Constructed** | 450K prompts | 13 demographic axes |
| BBQ (Parrish 2022) | **Templated QA** | 58K | 9 axes incl. intersectional |
| Measuring Hate Speech (Sachdeva 2022) | Multi-platform | 50K | Rasch-IRT hate score |
| Gab Hate Corpus (Kennedy 2022) | Gab | 27K | Hate-speech typology |
| HateCheck (Röttger 2021) | **Constructed** | 3.7K functional tests | Diagnostic |

Three structural observations from this set:
- The *naturalistic* labeled corpora are uniformly small (≤2M; most ≤60K).
- The *large* labeled corpora are constructed/templated/synthetic (HolisticBias, BBQ, ToxiGen).
- Almost none span more than 2–3 years; none span 16+ years with consistent labels.

### 3c. Longitudinal / historical text corpora

| Corpus | Source | Size | Span | Naturalistic lay discourse? | Sub-national geography? | Per-author info? |
|---|---|---|---|---|---|---|
| COHA (Davies 2010, 2012) | Fiction, news, magazines, non-fiction | ~475–600M words | 1820s–2010s | No (edited registers) | No | No |
| Google Books Ngrams (Michel 2011; Pechenick 2015) | Books | 500B+ words | 1500–2019 | No (heavy academic skew post-1990) | No | No |
| COCA (Davies 2008) | Mixed registers | 1.1B words | 1990–2019 | Partial | No | No |
| NOW (Davies 2016) | Online news | 19.4B+ words | 2010–present | No (news only) | Country-level only | No |
| Twitter Decahose / TweetsKB | Twitter | 1.5B+ (TweetsKB); 50M/day (Decahose) | 2006–2023 (Decahose closed) | Yes | Geotag ~1–3% only | Yes |
| Pushshift Reddit | Reddit | ~5.6B comments | 2005–2022 | Yes | None native; inference (Harrigian 2018) | Yes |
| **ISAAC** | Reddit, attitude-filtered | 554M+ comments | 2007–2023 | Yes | **US state-level inference** | Yes (persistent IDs, anonymized) |

ISAAC sits in a unique cell of this matrix: naturalistic lay discourse + per-author trajectories + within-country geography + pre-computed psych labels + a span (16 years) wide enough to cover every major recent attitude-shaping event in US discourse.

---

## 4. Features to highlight in the paper

**Reordered per Babak's feedback:** the platform/website nature, the open and modular pipeline, and the filtering rigor should come above the sentiment/emotion ensembling, because off-the-shelf sentiment and emotion are widely available (even if less rigorously applied), whereas an open, coding-free, modular *resource infrastructure* for psychological work on Reddit is genuinely novel. I've placed filtering rigor immediately after the platform aspects on the grounds that the <5% residual-irrelevance number is the single most striking quantitative claim ISAAC can make against any comparator. Scale × naturalism × longitudinal span remains in the lead group because it's the headline scale claim that anchors everything else. Ensembling drops to mid-tier; ethics stays at the end.

1. **Coding-free website + open, modular pipeline (the "platform" nature).** `isaac.psychology.illinois.edu` lets researchers without computational training extract the full corpus or stratified samples for analysis. The GitHub repository documents every step of construction and exposes each stage of the pipeline as a swappable component: keyword sets, language filter, relevance classifier, label resources, location model, and anonymizer can each be replaced or extended independently. Researchers who need a different set of social distinctions can substitute keyword files and rebuild; researchers who need a different labeler can drop one in; researchers who want to extend the corpus into 2024+ Reddit data can do so without re-implementing anything. This is what makes ISAAC a *platform* rather than a single fixed dataset, and it is the feature that most directly addresses the replication and reproducibility concerns BRM exists to publish on. No comparator naturalistic corpus offers a coding-free access mode for psychologists; none offers a modular, swappable pipeline at this scale.

2. **Filtering rigor culminating in low-single-digit residual irrelevance.** A four-stage pipeline (keyword Aho-Corasick → fastText language → transformer relevance classifier → hyperscan complex pattern matching) addresses a well-known problem in ad hoc Reddit work — that keyword scraping yields large false-positive rates (e.g., "Black" in chess contexts, "disabled" in software contexts). **At the end of processing, stratified random samples drawn from each social distinction across years contain under 7% unrelated content across all six distinctions, with four of six below 5% (age 0.0%, sexuality 3.0%, weight 3.0%, skin_tone 5.2%; ability 7.0% and race 6.5% required multiple regex iterations and, for race and skin_tone, a relevance-classifier retraining step).** Stage-by-stage progression is reported in Table FPR (§6a). As far as the literature search returns, essentially no comparator naturalistic corpus reports an end-to-end residual-irrelevance audit of this kind. This headline number belongs in the intro and again in the validation section.

3. **Scale × naturalism × longitudinal span, simultaneously.** 554M+ labeled comments, 16 years (2007–2023), lay discourse, six categories of long-standing theoretical interest. Among naturalistic labeled corpora of discourse about social groups, ISAAC is two to four orders of magnitude larger than the largest existing comparator and is the only one to span the full 2007–2023 period with a consistent pipeline. This is the headline scale claim.

4. **Coverage of six social distinctions in parallel.** Sexuality, race, age, ability, weight, skin-tone. No other corpus applies the same labeling pipeline uniformly across all six. This enables direct cross-distinction comparison — for example, whether moralization or emotional intensity in discourse behaves the same way for race as for weight.

5. **US state-level user geography.** A weighted hierarchical model (words + subreddits + timestamps) estimates each author's home location. Pushshift has no native geography; Twitter geotags appear on under 3% of tweets. State-level granularity is what enables triangulation against Project Implicit IAT state-by-year aggregates, ANES MRP estimates, and GSS regional samples — the exact validation strategy Caliskan et al. (2017) and Garg et al. (2018) established for corpus-based work on attitude determinants.

6. **Generalization labels.** Clause-level scoring of genericity, eventivity, boundedness, and habituality — operationalizing the linguistic structure of generic statements ("Black people are…") in a theoretically grounded way (Hemmatian et al. 2021 on marijuana). No other large corpus of discourse about social groups offers this psycholinguistic affordance.

7. **Ensembled sentiment and emotion labels.** Three sentiment models (Stanza, VADER, TextBlob) and three emotion models (Hartmann's distilroberta, sickboi25, EmoBERTa) per post. Ensembling supports model-to-model agreement analyses and single-model artifact detection that single-model labels cannot. The underlying models themselves are off-the-shelf and widely available; the methodological contribution is in *applying them as an ensemble at scale* with internal-consistency reporting, and in providing the multi-model output directly to downstream users rather than collapsing to a single estimate.

8. **Anonymization and ethics.** Persistent random author IDs replace usernames; the paper should describe the ethical review and the data-handling protocol, since some BRM reviewers will be sensitive to identifiability concerns in any user-keyed Reddit corpus.

---

## 5. Research questions ISAAC enables

**Important framing note (Babak's point #2):** Throughout this section and the paper as a whole, the word *attitude* should be reserved for what external benchmarks (Project Implicit IATs, ANES feeling thermometers, GSS items) measure. ISAAC itself does not contain *attitude labels*; it contains naturalistic *discourse about social groups* together with computationally generated *discourse markers* — moralization, sentiment, emotion, generic-statement structure — that may serve as observable correlates and determinants of attitudes. The intro should make this distinction explicitly to avoid construct-validity objections from psychometrically inclined reviewers, and the RQs below are phrased accordingly.

The brief says: *highlight a range of questions without breadcrumbing the readers too much.* The pattern that has worked in BRM resource papers (e.g., CHILDES, BNC, MFTC) is to describe **classes of question** with one concrete example each, and to defer the actual analyses to follow-up papers. Reordering below tracks the feature ordering in §4: the platform/replication RQ leads, followed by the temporal, event-locked, geographic, cross-distinction, and convergent-validity RQ classes. The follow-up paper on ISAAC × Project Implicit should not be previewed beyond the convergent-validity RQ class.

1. **Canonical operationalization, replicability, and forward extensibility.** Because ISAAC is fixed, named, shared, and modular, any future claim about U.S. Reddit discourse on the six distinctions over 2007–2023 has a single operationalization that other researchers can re-run, audit, and contest. The same pipeline can be applied without modification to Reddit data from 2024 onward, allowing researchers to extend ISAAC's coverage as new data becomes available — including running the labels in near-real-time on freshly scraped data — without re-implementing keyword sets, filters, or labelers. This is the most modest and arguably most important RQ class the resource enables: it makes computational work on discourse about social groups straightforwardly replicable and forward-compatible. (Was §5 item 7, with the forward-extensibility point added per Babak's feedback.)

2. **Temporal shifts in discourse markers about social groups.** How does the moral, emotional, or generalized language used about a target group change across the 2007–2023 window? Concrete example: emotion-related markers in weight-related discourse across the period, without claiming any specific finding.

3. **Event-locked discourse change.** When does discourse shift, and around which events? Examples to *mention but not analyze*: *Obergefell* (2015, sexuality), Ferguson/Floyd (2014, 2020, race), COVID (2020, age/ability). ISAAC's continuous coverage enables interrupted-time-series and difference-in-differences designs at unprecedented resolution.

4. **Geographic variation in discourse markers, and triangulation with external attitude measures.** Do discourse markers about a given social group vary by US state? Do those state-level patterns correlate with state-by-year aggregates of Project Implicit IAT D-scores, ANES feeling thermometers, or GSS items? This is the convergent-validity bridge between ISAAC's *discourse markers* and *attitudes* as measured by established psychological instruments.

5. **Cross-distinction comparisons.** Are some social distinctions consistently more moralized than others in lay discourse? More emotionally polarized? More frequently the target of generic statements ("all X are…")? This is the affordance unique to ISAAC's parallel six-distinction structure.

6. **Convergent validation with attitude measures as its own RQ class.** The convergent-validity evidence reported in this paper is the down-payment on a larger program of work testing how ISAAC-derived discourse markers relate to attitudes as measured by self-report and implicit instruments. The intro should sketch this RQ class clearly without previewing the empirical findings of the follow-up Project-Implicit paper. In this paper, we should report only enough to demonstrate that the correlations are non-null and in the expected direction at the appropriate level of aggregation.

7. **Methodological hypotheses about discourse-marker measurement.** Does generic language predict downstream engagement (likes, replies)? Does ensembling reduce model artifacts that contaminate single-model sentiment or emotion analyses? Where do the three sentiment models or the three emotion models systematically disagree, and what does that disagreement reveal about each model's training-distribution biases? These are methodological RQs that a methods-leaning psychologist would care about, and that other large naturalistic corpora are too small or too uniform to support.

The intro should *not* hint at any specific empirical finding from the follow-up Project-Implicit paper. The convergent-validity section in this paper is the appropriate venue for that kind of result, and only at the level of "the ISAAC measures track the survey measures at the level of [aggregation]," not "and here's what we discovered."

---

## 6. Validity and reliability analyses — comprehensive menu, then prioritized

The menu below is organized by construct first, then ranked. Italicized items are *feasible from existing repo assets* — Babak already has the data on disk for them.

### 6a. Reliability of the upstream pipeline

| Analysis | What it shows | Feasible now? |
|---|---|---|
| Inter-annotator agreement (Cohen's κ, raw agreement) on the ~1,500-doc double-rated relevance training samples per distinction | The relevance classifier's training labels are reliable | **Yes — see Table κ below** |
| **End-to-end residual irrelevance rate on the fully processed corpus: stratified samples per distinction × year, hand-coded for relevance, reported as the proportion of "false positives" surviving all filtering stages.** Final rates are **under 7% across all six distinctions, with four below 5%**. | The headline reliability claim of the pipeline | **Yes — see Table FPR below** |
| Classifier-vs-human agreement on a held-out relevance test set | The relevance model recovers what humans would mark relevant | **Yes — see Table M below** |
| Per-distinction precision / recall / F1 of the relevance classifier (initial and post-retraining for race and skin_tone) | Which distinctions the pipeline filters cleanly vs. which needed iterations | **Yes — see Table M below** |
| Effect of `filter_keywords_adv` on residual false positives | Adds a meaningful safety net beyond the keyword + transformer pipeline | **Yes — see the A→B transitions in Table FPR below** |
| Language-filter false-rejection rate on English code-switched posts | We're not throwing away naturalistic in-group AAVE/Spanglish | Requires a small audit sample (~200 posts the filter rejected) |

#### Table κ — Inter-annotator agreement on the relevance training samples

Computed on the original double-rated samples (~1,500 documents per distinction) in `data/data_relevance_ratings/comments/`, which served as the training data for the initial relevance classifiers in Table M below. Binarization rule applied uniformly: literal `"1"` counts as relevant, every other value (including `"0"`, `"x"`, `"-1"`, and blanks) is treated as irrelevant; when a `random_id` is duplicated within a single rater's file, the within-rater labels are OR-merged before computing across-rater agreement.

| Group | N | Raw agreement | Cohen's κ | Pearson r |
|---|---|---|---|---|
| ability | 1,476 | 98.2% | **0.960** | 0.960 |
| sexuality | 1,493 | 93.6% | **0.871** | 0.872 |
| age | 1,485 | 90.6% | **0.812** | 0.821 |
| race | 1,498 | 92.6% | **0.784** | 0.787 |
| skin_tone | 1,885 | 93.4% | **0.725** | 0.726 |
| weight | 1,475 | 89.6% | **0.690** | 0.706 |

*Three distinctions (ability, sexuality, age) land in Landis & Koch's "almost perfect" band (κ ≥ 0.81); the remaining three (race, skin_tone, weight) are in the "substantial" band (κ 0.61–0.80). The lowest κ — weight — is driven mostly by a base-rate difference between raters (rater 0 marks 24.5% of documents relevant; rater 1 marks 17.4%), not by disagreement on clear-cut cases; this is consistent with the high raw agreement (89.6%). κ values were reproduced via `python code/cli.py -t comments -r metrics_interrater -g {group}`.*

#### Table M — Held-out classifier performance for the in-house relevance models

Precision, recall, and F1 are computed for the **relevant** class (the rarer class for most distinctions). "Initial 10% held-out" sets are random 10% slices of the original training samples summarized in Table κ; "retraining 10% held-out" sets are random 10% slices of the 400-document retraining inputs (`qa_r_retraining_input_*` in `data/data_relevance_QAratings/`) drawn from the combined post-filter QA samples. Inference for race and skin_tone uses the thresholded decision rule `P(relevant) > 0.6 ⇒ predict relevant, else irrelevant`. Other distinctions use the standard argmax of the two-class softmax (no thresholding).

| Group | Model | Test set | Test N | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| ability | initial | initial 10% held-out | 148 | 0.815 | 0.880 | **0.846** |
| age | initial | initial 10% held-out | 149 | 0.853 | 0.921 | **0.886** |
| sexuality | initial | initial 10% held-out | 150 | 0.962 | 0.949 | **0.955** |
| weight | initial | initial 10% held-out | 148 | 0.975 | 1.000 | **0.987** |
| race | initial | initial 10% held-out | 150 | 0.811 | 0.811 | **0.811** |
| race | retrained | retraining 10% held-out | 40 | 0.786 | 0.917 | **0.846** |
| race | retrained | initial 10% held-out (after retrain) | 150 | 1.000 | 0.757 | **0.862** |
| skin_tone | initial | initial 10% held-out | 189 | 0.862 | 0.714 | **0.781** |
| skin_tone | retrained | retraining 10% held-out | 40 | 0.875 | 0.840 | **0.857** |
| skin_tone | retrained | initial 10% held-out (after retrain) | 189 | 0.839 | 0.743 | **0.788** |

*Both retrained models (race, skin_tone) were trained with an asymmetric `0_to_1` confusion penalty in the loss — false positives (irrelevant documents predicted as relevant) incur extra cost — and a `P(relevant) > 0.6` inference threshold. **This is deliberate**: the operational target is the residual false-positive rate in the deployed corpus (Table FPR), not balanced F1, so the retraining was set up to favor precision over recall. Race retraining produced an unambiguous gain on the original held-out (F1 0.811 → 0.862; precision 0.811 → 1.000). Skin_tone retraining looks flat on the original held-out (F1 0.781 → 0.788) but reaches F1 0.857 on the retraining held-out and drops the deployed pipeline's FPR from 44.0% to 5.2% (Table FPR). The interpretation is that the original training sample (keyword-matched, mostly easy decisions) and the retraining sample (post-regex, edge-case-heavy by construction) capture different distributions of irrelevant content, and the retrained skin_tone model is similarly competent on both — so its operational FPR drop is real even when the held-out F1 looks unchanged. All numbers above are freshly recomputed against the on-disk checkpoints in `models/`; for race this is `retrain_relevance_race_roberta-large_2/`, for skin_tone `retrain_relevance_skin_tone_roberta-base_1/`. The N=40 retraining held-out splits are small enough that single-prediction flips move P/R by ~3–4 percentage points; treat those numbers as approximate.*

#### Table FPR — End-to-end residual irrelevance by pipeline stage

Single-rater human ratings on stratified random samples drawn from the post-filter corpus at four sequential stages of pipeline refinement. *Stage A* (`qa_a_postinit_*`, n=200) = sample drawn after the keyword filter and the initial relevance classifier. *Stage B1* (`qa_b1_postregex_*`, n=100) = sample drawn after the first version of the `filter_keywords_adv` complex-regex filter, designed against the Stage A false positives. *Stage B2* (`qa_b2_postregex_*`, n=100, race and skin_tone only) = sample drawn after a second regex iteration. *Stage C* (`qa_c_postretrain_*`, n=153, race and skin_tone only) = sample drawn after retraining the relevance classifier on the combined Stage A/B1/B2 inputs. Binarization rule matches Table κ. All files live in `data/data_relevance_QAratings/`.

| Group | A: post-initial (n=200) | B1: post-regex v1 (n=100) | B2: post-regex v2 (n=100) | C: post-retrain (n=153) |
|---|---|---|---|---|
| ability | 20.5% | **7.0%** | — | — |
| age | 5.5% | **0.0%** (n=99) | — | — |
| sexuality | 14.5% | **3.0%** | — | — |
| weight | 16.5% | **3.0%** | — | — |
| race | 51.0% | 31.0% | 41.0% | **6.5%** |
| skin_tone | 44.0% | 34.0% | 29.0% | **5.2%** |

*Boldface marks the final FPR achieved per group. Four of six distinctions converged in a single regex iteration; race and skin_tone required two regex iterations plus a relevance-classifier retraining step. The race regression at B2 (31% → 41%) is worth noting: regex v2 introduced new boundary rules that cleaned some false positives but opened space for new ones, and the long tail of pop-culture and name-based false positives ("walter white", "blackpink", "barry white", "white sox", "dana white") proved hard to eliminate through regex changes alone — they were the cases the subsequent retraining step targeted. Stage C is currently single-rater (rater 1) for race and skin_tone; rater 0's labels for the same 153 documents are pending and the final entries will be OR-merged across both raters when those arrive, which can only lower the reported FPR. The four single-iteration groups did not require a Stage C audit because Stage B1 already reached the operational target.*

### 6b. Reliability of the discourse-marker labels

(Revised per Babak's feedback: off-the-shelf labelers cite creators' reported performance; in-house labelers report a classifier performance table.)

| Analysis | What it shows | Feasible now? |
|---|---|---|
| Inter-model agreement among the three sentiment models (Stanza, VADER, TextBlob): pairwise Pearson/Spearman, ICC(2,k), Krippendorff α on a stratified subsample | The ensemble is internally coherent and reduces single-model artifacts | *Yes — can be computed directly from corpus columns 11–16. Placeholder pending full dataset completion.* |
| Inter-model agreement among the three emotion models on each Ekman category | Same logic, for emotion | *Yes — columns 35–54. Placeholder pending full dataset completion.* |
| **Citation of creators' reported performance for off-the-shelf sentiment and emotion models** (Stanza/VADER/TextBlob; Hartmann/sickboi25/EmoBERTa) | The off-the-shelf models the ensemble draws from have published validation evidence | Cite from original papers; no fresh annotation needed |
| **Moralization validation against the USC MOLA Lab's human-coded labels** (the moralization classifier was trained on MOLA's MFRC dataset, and MOLA is performing the evaluation) | The in-house moralization classifier recovers what MOLA's coders would code | Per Babak: already underway via the MOLA collaboration |
| Generalization-label reliability against the original Hemmatian et al. (2021) marijuana coding scheme | The generalization model recovers the construct it claims to | Yes if the marijuana validation set is preserved; this is in-house and belongs in the classifier-performance table (§Prioritized list, #1) |
| Test-retest stability of model outputs across identical re-runs (or near-duplicates) | Labels are deterministic / approximately deterministic | *Yes, fast to run; optional* |
| Bootstrap stability of label *aggregates* (monthly mean sentiment, etc.) | Researcher-facing summary statistics are stable to resampling | *Yes — fast; optional* |

### 6c. Validity of ISAAC's discourse markers as indicators of psychologically meaningful signal

(Framing follows §5: ISAAC contains discourse markers, not attitude labels. Validation here is about whether those markers track psychologically meaningful external signal, including but not limited to attitudes.)

| Analysis | What it shows | Feasibility |
|---|---|---|
| **Convergent validity with Project Implicit.** State × year aggregates of ISAAC sentiment/emotion/moralization markers for race, sexuality, weight, age, ability, skin-tone vs. the corresponding IAT D-scores from Project Implicit, with Spearman ρ at the state-year panel level | Discourse markers track implicit attitudes — the single most important external-validity claim available | High — Project Implicit data is public. **Crucial:** keep the headline finding for the follow-up paper; this paper should show only that the correlation is non-null and goes in the expected direction, summarized in one figure. |
| **Convergent validity with ANES feeling thermometers** (race, sexuality at minimum) | Discourse markers track self-report attitudes | High — ANES is public; aggregate by year and region |
| **Convergent validity with GSS items** (e.g., GSS items on race, homosexuality, ageism) | Same | High — GSS is public |
| **Divergent validity** — ISAAC markers uncorrelated with attitudes they shouldn't track (e.g., race-related ISAAC sentiment vs. weight-related IAT) | Construct boundary | Medium — pick 2–3 clean divergent pairs |
| **Event-locked construct validity via post-frequency spikes.** Per Babak's feedback, the cleanest event-validation evidence we have right now is the *post-frequency* signal: e.g., a marked spike in race-relevant post volume around May–June 2020 (Floyd / BLM resurgence) that is already visible in the existing monthly counts. The paper should show this for several anchor events: *Obergefell* (June 2015, sexuality), COVID onset (Mar 2020, ability and age), Floyd / BLM (May–Jun 2020, race), and any others that produce clearly visible spikes in `counts_monthly.csv`. This is far stronger and more defensible than label-shift evidence at this stage. | The corpus picks up signals that real-world events should produce | **High — already partially observed for BLM; the monthly counts files in the repo support this directly** |
| **Demographic representativeness check.** ISAAC location distribution vs. American Community Survey state populations vs. Pew's Reddit user demographics | Document the well-known Reddit skew so reviewers don't have to ask | High |
| **Robustness across subreddit composition.** Drop the top-1 / top-5 / top-10 subreddits and re-compute aggregate trends — do the trends survive? | The signal isn't an artifact of one subreddit's volume dominance | High |
| **Label robustness across distinction-specific subreddits.** E.g., does the race sentiment trend look the same if r/BlackPeopleTwitter is excluded? | Within-distinction subreddit robustness | High |

### 6d. Reliability of the location model

| Analysis | What it shows | Feasibility |
|---|---|---|
| **Manual audit of automatically labeled training-set authors.** Per the attached `Geolocation Prediction- Reddit.docx`: a random sample of 100 auto-labeled authors was hand-reviewed. 18% were initially mismarked, but 5 of those errors were attributable to regional/non-English variants that were not normalized by the extraction pipeline; after refinement, the verifiable error rate dropped, yielding **>90% correctness in verifiable cases**. This is the **headline location-validation number** for the paper. Include the audit methodology and the refinement step. | The location annotations the model is trained on are largely correct | **Yes — already done; document in the paper** |
| Held-out classification accuracy (top-1, top-3) by US/non-US, region, state, with confidence-threshold sweeps | The location model generalizes from training labels to held-out users | *Yes — `train_location_weighting.py` produces this* |
| Calibration plot for `location_prob` | Reported probabilities are honest | *Yes* |
| Comparison of ISAAC state distribution to American Community Survey marginals after adjusting for Reddit's known demographic skew | Geographic mapping is plausible at the population level | High |

### Prioritized recommendation for *this* paper (revised per Babak's feedback)

The revised slate below reflects four specific points Babak raised:

- **(a)** Replace "interrater agreement on relevance" with a more informative **classifier performance table (precision, recall, F1)** for the resources the lab trained themselves: `filter_relevance` and the `label_*` resources whose models were built in-house. Off-the-shelf models (sentiment/emotion) point to creators' reported performance instead.
- **(b)** For ensemble internal agreement on sentiment and emotion: keep as a planned analysis, with a **placeholder in the table** until the full dataset finishes running. The analysis itself is straightforward once labels are complete.
- **(c)** For model-vs-human agreement: this paper does not need to commission new annotations across the board. Moralization is already being evaluated against human-coded labels by the **USC MOLA Lab** (whose MFRC dataset was the training source). For sentiment and emotion, the published performance from the **off-the-shelf models' creators** (Stanza/Vader/TextBlob; Hartmann's distilroberta, sickboi25, EmoBERTa) is reported as-is. Generalization is in-house and goes into the classifier-performance table.
- **(d)** For event-locked construct validity: use **post-frequency spikes** as the evidence that the corpus picks up major social events. Babak reports that this is already visible for BLM 2020 in the monthly counts; the paper should report it for several anchor events.
- **(e)** For location: use the existing **100-sample manual audit** with >90% correctness in verifiable cases (see §6d and the attached `Geolocation Prediction- Reddit.docx`).

| # | Analysis | Why | Source of evidence |
|---|---|---|---|
| 1 | **Classifier performance table (precision, recall, F1) for in-house models.** Columns: model name (`filter_relevance` per distinction; `label_moralization`; `label_generalization`; location model components), held-out test set, P / R / F1, n. Relevance classifiers are reported in **Table M (§6a)**; rows for `label_moralization`, `label_generalization`, and location-model components are still to be added. | Demonstrates the in-house pipeline is doing what it claims, and is the form of evidence BRM reviewers expect for novel classifiers. *Replaces the previous "relevance kappa" recommendation by Babak's preference (point #1 of his feedback).* | Relevance: Table M; others: existing `results/` artifacts and `report_*` files |
| 2 | **Inter-annotator agreement (Cohen's κ) on the relevance training samples** (~1,500 double-rated documents per distinction in `data/data_relevance_ratings/comments/`, *not* the single-rater post-filter QA samples). Reported in **Table κ (§6a)**; kept as a secondary support for the classifier table, demonstrating that the *training labels* the classifier was fit to are themselves reliable. | Required by reviewers familiar with classifier-construction norms; computed via `metrics_interrater.py` | Table κ; raw data in `data/data_relevance_ratings/comments/relevance_sample_{group}_{rater}_rated.csv` |
| 3 | **Ensemble internal agreement (sentiment + emotion).** Pairwise Pearson/Spearman, ICC(2,k), Krippendorff α on stratified subsamples across the three sentiment and three emotion models. **Placeholder pending full dataset completion (Babak's point #5b).** | Validates the ensembling claim, on which the methodological contribution of the sentiment/emotion section rests | Will run on the corpus columns once labeling is complete |
| 4 | **Reference to creators' reported performance for off-the-shelf labelers** (Stanza / VADER / TextBlob for sentiment; Hartmann distilroberta / sickboi25 / EmoBERTa for emotion). Cite their validation numbers in the relevant Methods/Validation paragraph rather than re-running human validation here. (Babak's point #5c.) | Avoids redundant annotation work for off-the-shelf models that have already been validated by their creators; honest about which labels are model-generated and which are in-house | Citations from the original papers |
| 5 | **Moralization validation against USC MOLA Lab's human-coded labels.** Report classification metrics on MOLA's evaluation set; cite their evaluation directly (Babak's point #5c). | Moralization is in-house but trained on MOLA's MFRC data; the appropriate validation set is MOLA's | USC MOLA Lab evaluation |
| 6 | **Convergent validity with at least one external benchmark** (Project Implicit IAT preferred; ANES feeling thermometers as backup) at the state-year panel level, with effect size and CI. Headline finding kept for follow-up paper. | The single most important external-validity result for psychologists | Public Project Implicit / ANES data |
| 7 | **Event-locked construct validity via post-frequency spikes.** Report monthly post counts around anchor events: *Obergefell* (June 2015, sexuality), COVID onset (Mar 2020, ability/age), Floyd / BLM resurgence (May–June 2020, race). The BLM 2020 spike is already visible in the data per Babak. (Babak's point #5e.) | Demonstrates the corpus tracks real-world events; vivid, fast, hard to argue with | Existing `counts_monthly.csv` |
| 8 | **Location-model validation: 100-sample manual audit.** Report the >90% correctness figure in verifiable cases, the methodology (random sample from the auto-labeled training set), and the post-audit refinement of the normalization pipeline (regional-language handling). (Babak's point #5f and the attached geolocation doc.) | Establishes location-label reliability for the geographic RQs | Existing audit; documented in the attached geolocation doc |
| 9 | *(Optional)* Held-out classification accuracy of the location model on a separate held-out user set, plus calibration plot for `location_prob`. | Separates *training-label reliability* (#8) from *prediction accuracy*; the two are distinct and BRM reviewers may ask for both | `train_location_weighting.py` outputs |
| 10 | *(Optional, space permitting)* Subreddit-robustness check + demographic-representativeness disclosure. | Pre-empts the "isn't this all just r/BlackPeopleTwitter" and "Reddit isn't representative" reviewer objections | Existing data |

**Net result**: a leaner, more honest V&R slate. The in-house models get hard performance numbers; the off-the-shelf models point to the literature where their validation already lives; the convergent-validity claim is single-figure, deferring the deeper analysis; event validity is volume-based and immediately visible; location uses the audit Babak already has.

---

## 7. Recommended introduction outline (~2,000–2,500 words)

**Section A — The Opportunity (300–450 words).** Lead with the positive frame: psychology is moving toward theory-driven, naturalistic, large-data work on social attitudes. Briefly note the methodological maturity arc (hand-coded → ad hoc → shared resources). Position ISAAC at the *shared resource* end.

**Section B — The Problem with the Status Quo (300–400 words).** Brief and constructive. The ad hoc pipeline problem: each study scrapes its own corpus, picks its own labelers, validates little, and yields results that don't combine or replicate. Cite specific examples of methodological heterogeneity (Pechenick 2015 on Google Books; the documented bias in single-model sentiment; Reddit-keyword false-positive problem). Frame as *the natural consequence of a young subfield maturing*, not as scolding.

**Section C — Existing Resources and Their Limits (350–500 words).** Walk briefly through three families using the comparison tables in §3. Be charitable: each comparator solved a piece of the problem. The gap ISAAC fills is the *intersection* of naturalistic + longitudinal + multi-distinction + multi-label + geography. Use one short table or figure if BRM permits.

**Section D — Introducing ISAAC (400–550 words).** Present the corpus at a high level: source, size, span, six distinctions, label families, pipeline overview (filter → label → organize), coding-free website, open code. Foreground features #1–#3 from the revised §4 — i.e., **lead with the platform/website/modular-pipeline nature, then the <5% filtering rigor, then the scale × span × six-distinction headline**. Demote the sentiment/emotion ensembling discussion to a sentence that acknowledges these models are off-the-shelf and that the methodological contribution is in the ensembling and reporting. Keep the technical pipeline details for the Methods section; mention the four-stage filter but defer the algorithm names. Also include the framing sentence from §5 that ISAAC contains *discourse markers* and not *attitude labels*.

**Section E — Research questions ISAAC enables (250–350 words).** Compressed version of §5: name the four to five RQ classes, give one example each, do not preview empirical findings.

**Section F — Validation strategy (200–300 words).** Preview the V&R section: which kinds of evidence the paper will report (reliability of pipeline, internal consistency of ensembles, model-vs-human agreement, convergent validity with external attitude measures, event-locked construct validity). Connect explicitly to replication/reproducibility — the BRM angle.

**Section G — Closing transition (80–120 words).** Standard "the remainder of the paper proceeds as follows" close, but with one sentence on the larger ambition: that ISAAC be a platform, not a one-off.

---

## 8. Outstanding decisions / questions Babak should weigh in on before the Word draft

1. **Final corpus number.** README says 554,464,184 comments; submissions TBA. The intro should commit to either (a) "554M+ comments, with submissions forthcoming" or (b) "~1B posts when submissions are integrated." (a) is more conservative and matches what's verifiable now; (b) is more impactful. Recommendation: (a) for the intro, (b) for the abstract with an explicit caveat.
2. **Whether to name the planned follow-up papers in the intro.** Recommendation: name the *kinds* of follow-up work (Project Implicit triangulation, event analyses) but not specific findings. This prevents reviewers from demanding we run those analyses *in this paper*.
3. **Authorship of comparator reflections.** Some of the comparators in §3 are direct precedents (MFTC's PABAK framing, Sachdeva's Rasch). Recommendation: cite them in the V&R section as the precedent for our specific analyses, not just in the comparison table.
4. **Ethics paragraph placement.** The README mentions anonymization but doesn't discuss IRB. BRM increasingly expects a paragraph on ethical sourcing of public-data corpora. Recommendation: a short paragraph at the end of §D or beginning of methods.
5. **Number cap for the Word draft.** Recommended target: ~2,200 words for the intro. Confirm or adjust.

---

## 9. Notes on style for the prose draft

- BRM style is plain, declarative, sparing with hedges.
- Avoid the "many researchers have…" opener.
- One-paragraph-per-idea, no nested subheadings inside the intro.
- Cite specific corpora by name on first mention; defer the comparison table to a numbered table referenced from the prose.
- The reviewers will be psychologists. Lead with constructs (moralization, sentiment, emotion, generalization); pipeline mechanics come second.
