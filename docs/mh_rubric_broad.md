# `mh` labeling rubric — BROADENED (v2, 2026-07-20)

Decide whether a Reddit comment (from an athlete population) involves **mental health**,
**broadly defined**. This v2 rubric deliberately widens `mh` to include **sports/performance
psychology**, per the research decision that performance struggles are how most athletes first
experience mental-health difficulty and can point to a larger issue. When genuinely uncertain,
**lean YES**.

## mh = 1 (YES) if the comment touches ANY psychological or emotional experience, including:

- **Clinical / general mental health:** depression, anxiety, panic, eating disorders, self-harm,
  trauma, burnout, loneliness, low mood, grief, therapy or medication, loss of motivation.
- **Performance psychology (NOW IN SCOPE):** performance anxiety, nerves, choking under pressure,
  confidence, self-doubt, self-esteem tied to performance, fear of failure, mental blocks,
  overthinking, feeling watched or judged, and **pressure / expectations** — from oneself, parents,
  coaches, teammates, or crowds.
- **Emotional struggle affecting the person:** feeling worthless, hopeless, overwhelmed, or
  frustrated to the point of distress.

**Anchor example:** *"I don't play well when my parents are at the game"* → **1** (performance is
being affected by pressure/expectation — a psychological factor).

## mh = 0 (NO) only for:

- **Neutral / factual / technical talk:** game results, stats, technique, equipment, schedules,
  highlights, tactics.
- **One-off competitive emotion that resolves** and shows no inner struggle — e.g. *"gutted we lost
  but we'll bounce back"*, *"nervous but hit the winner"*.
- **Pure positive hype** with no psychological struggle — e.g. *"so hyped, crushed it today"*.

## Notes
- The **`sport` dimension is unchanged**; only `mh` widens.
- `relevant = mh AND sport` (recomputed after relabeling).
- Previous (narrow) rubric excluded performance psychology — those labels are preserved in
  `data/data_relevance_ratings/comments/_backup_narrow_rubric_20260720/` and in git history.
