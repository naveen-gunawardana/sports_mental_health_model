# Relevance rating — athlete mental health

Rate each comment in the `rating` column of `relevance_sample_athlete_{0,1}_rated.csv`.
Judge **only the text shown** — do not look at the key file (subreddit/author are hidden on purpose).

The question: **Does this comment refer to or imply something about a person's mental health?**
"Mental health" = any psychological difficulty, symptom, or state — e.g. depression, anxiety,
panic, disordered eating, sleep problems, burnout/loss of motivation, trauma, substance use,
suicidal thoughts, or coping with / seeking help for any of these — even in passing.

## DECIDED RULE (read this first)

Rate for **ANY mental health — not just sport-related mental health.** The sample is
already all athletes, so you do NOT need a sport connection in the comment. If it refers
to a real mental-health thing in any way, it's `1`:

- **Passing mentions count.** "...struggle with learning (thanks adhd)" -> 1
- **Advice to others counts.** "get some therapy my man" / "see a counselor" -> 1
- **Third-person counts.** "the tournament should acknowledge the mental health of its players" -> 1
- **Everyday anxiety/depression counts.** "i have social anxiety in stores" -> 1

Only `0` when there's no real mental-health content:
- keyword used in a non-MH sense ("not eating too much of it" = food) -> 0
- pure sport/game talk, nutrition, or physical injury with no MH -> 0
- figurative/joke ("ptsd from playing on double rims", a fanbase being "clinically depressed") -> 0

## Labels

- **1 — Relevant:** Explicitly refers to or implies something about one or more individuals'
  mental health (a difficulty, symptom, emotional struggle, coping, or help-seeking), even in
  passing. (The person can be the author or someone else.)

- **0 — Irrelevant:** Does not refer to or imply anything about anyone's mental health, even in
  passing. A keyword may appear in a non-mental-health sense (e.g. "not eating" about a recipe,
  "depressed" about the economy, "choke" about a game) — that is **0**.

- **x — Unclear:** You can't tell whether it's about mental health, usually because the comment
  is incoherent or leans on missing conversational context.

## Notes
- These are high-recall keyword matches, so expect false positives — the target is that **≤ ~10%**
  come back Irrelevant (0). If far more are 0, the keyword filter needs tightening.
- Rate independently. Do not discuss or compare until both files are complete.
- After both are rated, agreement is measured with **Cohen's Kappa** (goal ≥ 0.6) via
  `metrics_interrater.py` from Babak's repo (or an equivalent script).
