# NOTE: This script currently only applies to binary relevance ratings. If the ratings are on a different scale, or if there are multiple non-binary categories, this script would need to be updated. 

### Imports

# import functions and objects
from cli import get_args, DATA_DIR

# import python packages
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr
import csv
csv.field_size_limit(2**31 - 1)  # Increase the field size limit to handle larger fields
import itertools
import os

### Argument Handling

args = get_args()
group = args.group
type_ = args.type
num_annot = args.num_annotators  # CLI default = 2

### Path Handling

# Where to find the rated relevance samples. Defaults to the canonical location
# data/data_relevance_ratings/<type>/ where the original double-rated samples live.
if not args.input:
    ratings_path = DATA_DIR / "data_relevance_ratings" / type_
else:
    ratings_path = args.input

### Label binarization

# NOTE: The current assumption is that the relevant category is indicated by "1" and all other values (including empty strings) are treated as non-relevant (0).

def binarize(cell: str) -> int:
    return 1 if str(cell).strip() == "1" else 0

### Fleiss' kappa for N raters with binary labels.
# Generalises Cohen's kappa to >2 raters. Reduces to Cohen's kappa when N=2 and the two
# rater label vectors are aligned 1:1 on the same items.
# "`rating_vectors` is a list of N parallel 0/1 lists (one per rater), all the same length."
def fleiss_kappa_binary(rating_vectors):
    
    n_raters = len(rating_vectors)
    n_items = len(rating_vectors[0])
    if any(len(v) != n_items for v in rating_vectors):
        raise ValueError("All rater vectors must have the same length")
    if n_raters < 2 or n_items == 0:
        return float("nan")

    # For each item, count how many raters assigned each category (0 and 1).
    # P_i = (sum_k n_ik^2 - n) / (n * (n - 1)) is the per-item agreement.
    # Pbar = mean of P_i across items; Pe = sum_k (p_k)^2 where p_k is overall proportion.
    p_bar_sum = 0.0
    cat_totals = [0, 0]  # raters assigning 0, raters assigning 1, across all items
    for i in range(n_items):
        n_i0 = sum(1 for v in rating_vectors if v[i] == 0)
        n_i1 = n_raters - n_i0
        p_bar_sum += (n_i0 * n_i0 + n_i1 * n_i1 - n_raters) / (n_raters * (n_raters - 1))
        cat_totals[0] += n_i0
        cat_totals[1] += n_i1
    P_bar = p_bar_sum / n_items
    total_ratings = n_raters * n_items
    P_e = (cat_totals[0] / total_ratings) ** 2 + (cat_totals[1] / total_ratings) ** 2
    if P_e == 1.0:
        return 1.0
    return (P_bar - P_e) / (1.0 - P_e)


### Main Evaluation

# Per-rater dict: random_id -> 0/1
ratings = {i: {} for i in range(num_annot)}

for rater in range(num_annot):
    fname = os.path.join(
        ratings_path,
        f"relevance_sample_{group}_{rater}_rated.csv",
    )
    with open(fname, "r", encoding="utf-8-sig", errors="ignore") as f:
        reader = csv.reader(f)
        for idx, line in enumerate(reader):
            if idx == 0 or not line:
                continue
            rid_raw = line[0].strip()
            if not rid_raw:
                continue
            try:
                rid = int(rid_raw)
            except ValueError:
                raise Exception(
                    f"Error parsing annotator {rater}'s row {idx}: non-integer id={line[0]!r}"
                )
            v = binarize(line[2] if len(line) >= 3 else "")
            # If the same random_id appears more than once in the same file (a small
            # number of duplicates exist in some samples), OR the labels: any "1"
            # wins over "0".
            ratings[rater][rid] = max(ratings[rater].get(rid, 0), v)

# Restrict to documents EVERY rater rated. With filter_sample --perc-overlap < 1.0,
# the "every-rater" set is exactly the shared subset that all annotators received.
id_sets = [set(ratings[r].keys()) for r in range(num_annot)]
common_ids = sorted(set.intersection(*id_sets)) if id_sets else []

# Surface any id that one rater has but others don't.
all_ids = set().union(*id_sets) if id_sets else set()
for rid in sorted(all_ids - set(common_ids)):
    missing = [r for r in range(num_annot) if rid not in ratings[r]]
    have = [r for r in range(num_annot) if rid in ratings[r]]
    print(
        f"Warning! ID {rid} rated by annotator(s) {have} but missing from annotator(s) {missing}"
    )

# Build per-rater 0/1 vectors over common_ids
vectors = [[ratings[r][rid] for rid in common_ids] for r in range(num_annot)]
n = len(common_ids)

print(f"Group: {group} ({type_})")
print(f"Annotators: {num_annot}")
print(f"N (rated by all annotators): {n}")
if n == 0:
    print("No documents rated by every annotator; nothing to score.")
    raise SystemExit(0)

for r, v in enumerate(vectors):
    print(f"  Rater {r} relevant rate: {sum(v)/n:.3f}")

# Raw all-agree rate: fraction of items where every rater gave the same label.
all_agree = sum(1 for i in range(n) if len(set(v[i] for v in vectors)) == 1) / n
print(f"Raw all-rater agreement: {all_agree:.3f}")

# Multi-rater Fleiss' kappa (collapses to Cohen's kappa when num_annot == 2 + binary)
fleiss = fleiss_kappa_binary(vectors)
print(f"Fleiss' kappa ({num_annot} raters): {fleiss:.4f}")

# Pairwise Cohen's kappa and Pearson r for each (i, j) pair, for finer-grained diagnostics.
if num_annot >= 2:
    print("Pairwise metrics:")
    for i, j in itertools.combinations(range(num_annot), 2):
        kappa_ij = cohen_kappa_score(vectors[i], vectors[j])
        pear_ij = pearsonr(vectors[i], vectors[j])
        print(
            f"  raters {i} vs {j}:  Cohen kappa = {kappa_ij:+.4f}  "
            f"Pearson r = {pear_ij.statistic:+.4f} (p = {pear_ij.pvalue:.2e})"
        )
