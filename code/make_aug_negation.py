"""Negation-hardening augmentation for the mh model (v2 — refined).

v1 accidentally poisoned the bare-"confidence issues" signal by putting that exact phrase in a
mh=0 negation control. v2 fixes that:
  - NEG_ABSENCE (mh=0): negate ANXIETY/NERVES/STRESS only (never "confidence issues"); add more of
    the stubborn "no X, I feel calm" pattern.
  - NEG_DISTRESS (mh=1): negation that IS distress (incl. "no confidence left" -> reinforces that
    lost-confidence = mh).
  - CONF_POS (mh=1): bare performance-confidence positives, to recover the regression v1 caused.

Re-runnable: first deletes its own id ranges (992xxxx/993xxxx/994xxxx) from the mh files, then
re-appends, so editing this file and re-running gives a clean result.
"""
import csv

NEG_ABSENCE = [   # mh = 0  (no anxiety/nerves/stress — NOT mental health). No "confidence issues".
 "i don't get nervous or anxious before games at all",
 "no anxiety here, i feel totally calm when i step on the court",
 "i never feel depressed about losing, it's just a game to me",
 "honestly i'm not stressed about the tournament even a little",
 "zero nerves for me, i actually love the big moments",
 "not worried at all, just locked in and focused",
 "i've never dealt with performance anxiety, it just doesn't affect me",
 "no pressure gets to me, i stay relaxed the whole game",
 "i'm not burnt out, i feel great and motivated every day",
 "i don't overthink it, i just go out and play free",
 "i don't tense up, crowds honestly don't bother me one bit",
 "no stress about my parents watching, i actually play better",
 "i'm not anxious, i'm just excited to compete",
 "no nerves at all, i feel completely calm and ready out there",
 "not depressed or down, honestly i've never felt better",
 "no anxiety whatsoever, i'm calm and focused every single game",
 "i don't feel any pressure, just relaxed and having fun",
]

NEG_DISTRESS = [  # mh = 1  (negation IS the distress)
 "i can't shake this anxiety before every single game",
 "i don't feel like myself anymore, been so down lately",
 "i can't stop overthinking every play and it's wrecking me",
 "no matter what i do i can't calm my nerves at the line",
 "i don't have any confidence left after that slump",
 "i can't get out of my own head when the crowd is watching",
 "nothing feels worth it anymore, not even the sport i loved",
 "i just can't handle the pressure, it breaks me every time",
]

CONF_POS = [      # mh = 1  bare performance-confidence, to recover the v1 regression
 "i have confidence issues shooting the ball",
 "i've got major confidence issues on the court lately",
 "my confidence is a real issue every time i step up to shoot",
 "i really struggle with confidence when i play",
 "my confidence has been an issue all season",
]

D = "data/data_relevance_ratings/comments"
rows  = [(9920001 + i, t, 0) for i, t in enumerate(NEG_ABSENCE)]
rows += [(9930001 + i, t, 1) for i, t in enumerate(NEG_DISTRESS)]
rows += [(9940001 + i, t, 1) for i, t in enumerate(CONF_POS)]
MY_IDS = {str(i) for i, _, _ in rows} | {str(9920001 + i) for i in range(50)} \
         | {str(9930001 + i) for i in range(50)} | {str(9940001 + i) for i in range(50)}

for tgt in [f"{D}/relevance_sample_mh_0_rated.csv", f"{D}/relevance_sample_mh_1_rated.csv"]:
    # read, drop any prior rows from my id ranges, then append the current set
    kept = []
    with open(tgt, encoding="utf-8-sig", newline="") as fh:
        r = csv.reader(fh); header = next(r)
        for row in r:
            if row and row[0].strip() and row[0].strip() not in MY_IDS:
                kept.append(row)
    with open(tgt, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(kept)
        for i, t, lab in rows:
            w.writerow([i, t, lab])
    print(f"{tgt}: {len(kept)} base rows + {len(rows)} negation-aug rows")

print(f"neg-aug v2: {len(NEG_ABSENCE)} absence(0), {len(NEG_DISTRESS)} distress(1), {len(CONF_POS)} conf-pos(1)")
