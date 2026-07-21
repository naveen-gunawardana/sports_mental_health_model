"""Hard-negative augmentation for the SPORT model.

Problem: the sport model fires on sport-VOCABULARY presence, not sport TOPIC. Because the corpus was
keyword-matched, almost every comment contains sport words ("basketball", "practice", "injured",
"training", "coach"), so the model says sport=1 even for comments that merely mention those words
while being about something else ("i don't watch basketball"; "practice makes a better cook").

Fix: teach the boundary with HARD NEGATIVES (sport word present, but NOT about sport -> sport=0) and
clear POSITIVES (genuinely doing/playing/training in sport -> sport=1). Re-runnable (deletes its own
id ranges first).
"""
import csv

# HARD NEGATIVES: contain a sport-ish word but are NOT about athletic activity -> sport = 0
HARD_NEG = [
 "i don't even watch basketball, it's honestly boring to me",
 "i don't care that some basketball player got injured, i don't follow the sport",
 "creativity comes after practice, you'll be a better cook than most",
 "practice makes perfect when you're learning the piano",
 "i got injured at work, broke my arm falling off a ladder",
 "my coach at work has been really supportive during this rough patch",
 "the team at my office is great, we get along well",
 "i scored really well on my final exam this semester",
 "training for my new job has been mentally exhausting",
 "i don't follow any sports, never really got into them",
 "the game on tv last night was boring so i changed the channel",
 "he's a real team player at the office, always helps out",
 "i pulled a muscle just carrying groceries up the stairs",
 "my therapist said practice self-compassion every day",
 "we lost the pitch competition at work and i'm gutted",
 "just practicing my public speaking for the presentation",
 "my kid got a trophy at the science fair",
 "i injured my back sitting at my desk all day",
 "life is a marathon not a sprint, take it slow",
 "i'm training myself to wake up earlier, it's hard",
 "the coach on that reality tv show is so dramatic",
 "i don't have the energy to even watch football anymore",
 "practice gratitude and your mindset improves",
 "got benched from the project at work, feeling useless",
]

# POSITIVES: genuinely about doing/playing/training in sport -> sport = 1
SPORT_POS = [
 "benched 225 today, new personal record",
 "ran 10 miles this morning, my legs are completely dead",
 "we won the game, i scored the winning goal in overtime",
 "spent two hours practicing free throws at the gym",
 "my marathon training is going really well this block",
 "deadlift went up 20 pounds, feeling strong in the squat rack",
 "coach put me in at point guard for the fourth quarter",
 "did my long run today, 18 miles for marathon prep",
 "hit a new pr on my clean and jerk at practice",
 "our team made the playoffs after that overtime win",
 "i tweaked my knee during soccer practice yesterday",
 "swim practice was brutal, coach had us doing sprints",
 "working on my jump shot every day after school",
 "tennis match went three sets, i cramped up at the end",
 "leg day at the gym destroyed me, can barely walk",
 "warmed up, stretched, then ran my fastest 5k ever",
]

D = "data/data_relevance_ratings/comments"
rows  = [(9950001 + i, t, 0) for i, t in enumerate(HARD_NEG)]
rows += [(9960001 + i, t, 1) for i, t in enumerate(SPORT_POS)]
MY_IDS = {str(9950001 + i) for i in range(80)} | {str(9960001 + i) for i in range(80)}

for tgt in [f"{D}/relevance_sample_sport_0_rated.csv", f"{D}/relevance_sample_sport_1_rated.csv"]:
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
    print(f"{tgt}: {len(kept)} base + {len(rows)} aug rows")

print(f"sport aug: {len(HARD_NEG)} hard-negatives(0), {len(SPORT_POS)} positives(1)")
