"""Hard-negative augmentation for the MH model (v2 - expanded, category-targeted).

The mh model over-flags sport comments with no mental-health content (~40% of 'relevant' on audit).
v1's 28 negatives weren't enough or diverse enough. v2 targets the FP categories the audit exposed:
physical injury, training logistics, stats/commentary, MH-word jokes/jargon, nostalgia, mod
boilerplate, gear/nutrition. Plus MH positives to hold recall.

Re-runnable (clears its own id range first).
"""
import csv

SPORT_ONLY = [   # sport/fitness content, NO mental health -> mh = 0
 # --- training logistics / programs / routines ---
 "full body kettlebell workout with 30 seconds rest between sets, swings squats press rows 3x15",
 "switching up the bulgarian plan again, i liked a more structured approach after some videos",
 "double pr on squat and deadlift this week, taking a 3 day break from my cut",
 "i load water and carbs the day i work out, throw in 25g of carbs just before lifting",
 "running the first 14 weeks of sbs then base cycles, keep the overwarm singles",
 "my split is three heavy days and one deload week, bench went up five pounds",
 "cardio first or weights first, i always lift then do a short jog to finish",
 "switching from the bodyweight routine to lifting, any recommendations on where to start",
 "meal timing and grams of protein per meal doesn't actually matter that much for hypertrophy",
 "the ohp is an overrated lift outside of training the specific movement pattern",
 # --- physical injury / pain, no emotion ---
 "i tore my acl, mcl, and lcl playing basketball and had surgery to repair it with a donor graft",
 "slipped disk maybe, it feels like muscular pain until i get tingles down my leg",
 "pulled a hamstring sprinting, iced it and i'm taking two weeks off to let it heal",
 "my knee has been swelling after long runs, going to get a gait analysis at the running store",
 "rolled my ankle at practice, it's puffy but i can still put weight on it",
 "lower back is tight from deadlifts, foam rolling and stretching seems to help",
 # --- stats / commentary / game recaps ---
 "he made 73 unforced errors in two sets, that's a record for errors per minute",
 "he averaged 28 points and 9 assists, easy mvp season honestly",
 "zone defense shut them down in the second half, great adjustment by the coach",
 "the ref completely blew that call at the end of the fourth quarter",
 "great game last night, that overtime buzzer beater was insane",
 "petra is down a break but she has momentum, isner monfils is entertaining to watch",
 "trade deadline was wild, they gave up three first round picks for him",
 "who do you have winning the finals, i think it goes seven games",
 # --- MH-word jokes / jargon / hyperbole (word present, not real MH) ---
 "i started with suicide sprints today and my legs are absolutely destroyed",
 "we did a suicide set at the end of practice, brutal conditioning",
 "a suicide cut is when you drop weight way too fast before a show",
 "this game is killing me, i can't watch us blow another fourth quarter lead",
 "the raptors losing games two and three is giving me a heart attack, what a series",
 "pg gave hibbert depression lol, that block was disrespectful",
 "lebron caused ad's depression, that's the real story here haha",
 # --- nostalgia / player history / PR / off-topic ---
 "brandon was a special player, i still watch his comeback against the mavs all the time",
 "the spotlight is new to her so she's agreeable to autographs and press right now",
 "here's the full set of quotes from coach k in the duke press room after the game",
 "it brings out more fans and helps the supplement and clothing companies grow",
 # --- gear / nutrition / body composition, no MH ---
 "what running shoes do you recommend for flat feet and high weekly mileage",
 "anyone eat 3 or 4 meals a day, do you use supplements while bulking",
 "i went from 400 pounds to 175, ended up skinny so i started eating more for gains",
 "best pre workout meal is oatmeal and a banana about an hour before",
 # --- mod / removal boilerplate ---
 "this submission has been removed as it is requesting medical or psychological advice which users here are not qualified to provide",
 "removed: please see the rules in the sidebar, this belongs in the weekly questions thread",
 # --- encouragement / motivational, no MH ---
 "just keep doing it, when i see anyone running overweight or not i think oh yay another runner",
 "give yourself permission to suck, half your routine is better than nothing",
 "progress isn't linear, 30 is young, you'll try and fail a lot but keep going",
]

MH_KEEP = [   # sport + genuine mental health -> mh = 1 (preserve recall)
 "having an anxiety attack in the gym, pretending the heavy breathing is from leg day",
 "training for an ultramarathon while falling into depression, the meds are what helped",
 "i got laid off two weeks ago and dragging myself to the gym is the only thing keeping me okay",
 "i have adhd and exercise is the only thing that reliably boosts my dopamine",
 "lifting saved me when i was in a really dark place mentally last year",
 "my eating disorder makes tracking macros dangerous so i had to stop counting calories",
 "running gave me panic attacks after my injury, therapy helped me get back out there",
 "i feel worthless every time i miss a lift, the gym has become tied to my self-worth",
 "since my dad passed i can barely find the motivation to train, everything feels heavy",
 "the pressure before games gives me such bad anxiety i can't sleep the night before",
 "definitely consult a mental health professional if you're this anxious, exercise spikes my anxiety",
 "i fell into depression last lockdown when the gyms closed and i'm barely back to normal",
 "this sounds like textbook burnout, no enjoyment and getting depressed even with the miles",
 "crossfit helped but i was struggling with my mental health and the workout wasn't going well",
 "i got cut from the team and got really depressed for a while before i found motivation again",
]

D = "data/data_relevance_ratings/comments"
rows  = [(9970001 + i, t, 0) for i, t in enumerate(SPORT_ONLY)]
rows += [(9980001 + i, t, 1) for i, t in enumerate(MH_KEEP)]
MY_IDS = {str(9970001 + i) for i in range(120)} | {str(9980001 + i) for i in range(120)}

for tgt in [f"{D}/relevance_sample_mh_0_rated.csv", f"{D}/relevance_sample_mh_1_rated.csv"]:
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
print(f"mh sport-only aug v2: {len(SPORT_ONLY)} sport-only(0), {len(MH_KEEP)} mh-keep(1)")
