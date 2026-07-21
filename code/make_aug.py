"""Targeted augmentation for the mh model: teach it that sports PERFORMANCE STRUGGLE
(tensing up, choking, nerves, lost confidence, pressure from people watching) IS mental health,
while neutral/positive performance talk is NOT. Appends rows to the two mh rater files.

Positives (mh=1): performance anxiety / pressure / confidence-loss with a struggle.
Controls  (mh=0): neutral or positive performance talk, no psychological struggle -- so the model
                  learns struggle-vs-not rather than just the words 'confidence'/'pressure'.
IDs use a 99xxxxx range to avoid colliding with the real 6-digit ids.
"""
import csv

POS = [
 "i don't play well when my parents are at the game because i tense up",
 "the second my dad shows up to watch i completely tense up and start missing easy shots",
 "whenever my family comes to watch i get in my head and play way worse",
 "i choke every time the game is close, my hands shake and i freeze up",
 "i tense up at the free throw line when it matters and airball it",
 "i've totally lost confidence in my shot, i keep replaying my misses and can't let go",
 "the pressure gets to me, i overthink every play and mess it up",
 "when the crowd is loud i get so nervous i forget the plays",
 "i get so anxious before big games i can barely warm up",
 "my confidence is completely gone after that last miss, i dread the next game",
 "i freeze under pressure, the bigger the moment the worse i play",
 "coach benched me and now i doubt every move i make on the court",
 "i psych myself out at the line, i tense up and my shot falls apart",
 "playing in front of scouts made me so nervous i couldn't perform",
 "i keep choking in the fourth quarter, the pressure just eats me alive",
 "ever since i shanked that penalty i've lost all confidence taking shots",
 "i overthink my serve so much now that i double fault under pressure",
 "the nerves before a race make my legs feel like jelly and i underperform",
 "i tense up on match point every time and hand the game away",
 "my hands get shaky at the free throw line when the game is on the line",
 "i can't shoot when people are watching, i get too in my own head",
 "the fear of missing makes me hesitate and i blow the layup",
 "i get so much anxiety before competition i throw up beforehand",
 "when my coach is yelling i tense up and completely lose my rhythm",
 "i lost my confidence after the injury and now i play scared",
 "every time the pressure is on i choke and let my team down",
 "i replay my mistakes for days and it kills my confidence for the next game",
 "the expectations from my parents make me so anxious i can't enjoy playing",
 "i freeze up whenever there's a big crowd, my mind goes blank",
 "i doubt myself so much now that i pass up open shots i used to make",
 "the pressure to perform has me stressed and nervous before every match",
 "i get in my head about my form and it wrecks my whole game",
 "i tense up in clutch moments and my shooting completely falls apart",
 "since coach started criticizing me i've been playing anxious and tight",
 "i'm terrified of failing in front of the crowd so i play it too safe",
 "i lose all focus when i'm nervous and make careless turnovers",
 "big games give me so much anxiety i can't sleep the night before",
 "i choke at the line, the pressure makes my whole body tense",
 "my nerves take over in tournaments and i never play to my level",
 "i've been so in my head about my slump that my confidence is shot",
]

# controls: neutral OR positive performance talk, NO psychological struggle -> mh = 0
NEG = [
 "i feel great out there, played my best game of the season today",
 "gained a ton of confidence this year and everything is clicking",
 "we won by twenty, pretty easy game honestly",
 "practiced my jumper all week and it's really paying off now",
 "hit a new PR on my squat this morning, feeling strong",
 "scored a hat trick, what a fun match that was",
 "my free throw percentage is way up after changing my form",
 "love playing in front of a big crowd, it pumps me up",
 "great team win, everyone moved the ball really well",
 "finally beat my rival, been working on my backhand for months",
 "the new cleats feel amazing, ran a fast 5k today",
 "coach put in a new play and it worked perfectly",
 "confident and locked in today, drained six threes",
 "solid practice, we ran drills and worked on spacing",
 "crushed my workout, added ten pounds to my bench",
 "we clinched the playoffs tonight, huge night for us",
 "my shot is falling and i feel unstoppable right now",
 "good scrimmage, defense looked sharp all around",
 "nailed my routine at the meet, so happy with the score",
 "played pickup for three hours, legs are tired but it was a blast",
]

D = "data/data_relevance_ratings/comments"
rows = [(9900001 + i, t, 1) for i, t in enumerate(POS)]
rows += [(9910001 + i, t, 0) for i, t in enumerate(NEG)]

for tgt in [f"{D}/relevance_sample_mh_0_rated.csv", f"{D}/relevance_sample_mh_1_rated.csv"]:
    # only append rows whose ids aren't already present (idempotent)
    existing = set()
    with open(tgt, encoding="utf-8-sig", newline="") as fh:
        r = csv.reader(fh); next(r)
        for row in r:
            if row and row[0].strip():
                existing.add(row[0].strip())
    add = [(i, t, lab) for (i, t, lab) in rows if str(i) not in existing]
    with open(tgt, "a", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        for i, t, lab in add:
            w.writerow([i, t, lab])
    print(f"{tgt}: appended {len(add)} rows (skipped {len(rows)-len(add)} already present)")

print(f"augmentation: {len(POS)} positives (mh=1), {len(NEG)} controls (mh=0)")
