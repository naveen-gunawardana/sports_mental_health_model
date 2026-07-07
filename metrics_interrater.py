"""
Cohen's Kappa between two relevance raters.

Reads the two rated files, aligns rows by random_id, and reports:
  - observed agreement, chance agreement, Cohen's Kappa
  - per-rater relevant/irrelevant/unclear counts and the "unrelated rate"
  - a 2x2 (binary) confusion table

Binarization matches train_relevance.py: only "1" counts as relevant;
"0", "x", and blanks are treated as 0. A 3-category kappa (1/0/x) is also printed.

Usage:
  python metrics_interrater.py \
     data/data_relevance_ratings/comments/relevance_sample_athlete_0_rated.csv \
     data/data_relevance_ratings/comments/relevance_sample_athlete_1_rated.csv
"""
import csv, sys
from collections import Counter
csv.field_size_limit(2**31 - 1)

DEF0 = "data/data_relevance_ratings/comments/relevance_sample_athlete_0_rated.csv"
DEF1 = "data/data_relevance_ratings/comments/relevance_sample_athlete_1_rated.csv"


def load(path):
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rid = (row.get("random_id") or "").strip()
            rating = (row.get("rating") or "").strip().lower()
            if rid and rating != "":
                out[rid] = rating
    return out


def kappa(labels_a, labels_b, categories):
    """Cohen's kappa for paired label lists over the given category set."""
    n = len(labels_a)
    if n == 0:
        return None, 0.0, 0.0
    po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    ca = Counter(labels_a)
    cb = Counter(labels_b)
    pe = sum((ca.get(c, 0) / n) * (cb.get(c, 0) / n) for c in categories)
    k = (po - pe) / (1 - pe) if (1 - pe) > 0 else 1.0
    return k, po, pe


def main():
    p0 = sys.argv[1] if len(sys.argv) > 1 else DEF0
    p1 = sys.argv[2] if len(sys.argv) > 2 else DEF1
    a, b = load(p0), load(p1)

    common = sorted(set(a) & set(b))
    only_a, only_b = set(a) - set(b), set(b) - set(a)
    if not common:
        print("No overlapping *rated* rows yet. Rate the same rows in both files first.")
        print(f"  rater 0 has {len(a)} rated, rater 1 has {len(b)} rated, overlap 0.")
        return

    ra = [a[i] for i in common]
    rb = [b[i] for i in common]

    def unrel(vals):
        rel = sum(1 for v in vals if v == "1")
        return len(vals) - rel

    print(f"Rated in both files: {len(common)}  "
          f"(only in rater0: {len(only_a)}, only in rater1: {len(only_b)})")
    print()
    print(f"Rater 0: {Counter(ra)}  -> unrelated {unrel(ra)}/{len(ra)} = {100*unrel(ra)/len(ra):.1f}%")
    print(f"Rater 1: {Counter(rb)}  -> unrelated {unrel(rb)}/{len(rb)} = {100*unrel(rb)/len(rb):.1f}%")
    print()

    # binary: 1 vs not-1
    ba = ["1" if v == "1" else "0" for v in ra]
    bb = ["1" if v == "1" else "0" for v in rb]
    kb, po, pe = kappa(ba, bb, ["0", "1"])
    print("== Binary (relevant=1 vs not) ==")
    print(f"  observed agreement po = {po:.3f}")
    print(f"  chance agreement   pe = {pe:.3f}")
    print(f"  Cohen's kappa         = {kb:.3f}   (goal >= 0.60)")
    # 2x2 table
    tab = Counter(zip(ba, bb))
    print("           r1=0   r1=1")
    print(f"   r0=0    {tab[('0','0')]:>4}   {tab[('0','1')]:>4}")
    print(f"   r0=1    {tab[('1','0')]:>4}   {tab[('1','1')]:>4}")
    print()

    # 3-category: 1 / 0 / x
    k3, po3, pe3 = kappa(ra, rb, ["0", "1", "x"])
    print("== 3-category (1 / 0 / x) ==")
    print(f"  observed po = {po3:.3f}   chance pe = {pe3:.3f}   kappa = {k3:.3f}")

    if kb is not None and kb >= 0.6:
        print("\n>= 0.60: agreement is substantial. Rubric is reliable; safe to scale up rating.")
    else:
        print("\n< 0.60: agreement too low. Review disagreements, tighten the rubric, re-rate.")


if __name__ == "__main__":
    main()
