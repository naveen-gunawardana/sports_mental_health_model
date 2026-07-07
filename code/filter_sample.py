### Imports

# import functions and objects
from cli import get_args, DATA_DIR
from utils import parse_range, groups, log_report, log_error

# import Python packages
import csv
csv.field_size_limit(2**31 - 1) # Increase the field size limit to handle larger fields
import random
import os
import time
import datetime
from pathlib import Path

### Argument Handling

# Extract and transform CLI arguments 
args = get_args()
years = parse_range(args.years)
type_ = args.type
group = args.group

### sampling hyper-parameters/initializations

num_annot = args.num_annotators           # CLI default = 2
perc_overlap = args.perc_overlap          # CLI default = 1.0

sample_size = getattr(args, "sample", 200) # PER-ANNOTATOR target document count
if not args.target: # determines the dataset stage to sample from
    target = "filter_keywords_adv" # default to post-filtering dataset
else:
    target = args.target # otherwise, set from the CLI arguments

# Spread a given total across years and the 3 sample types (top/bottom/random) as evenly as possible.
# Floor-dividing twice (total // years // 3) compounds rounding loss; distributing the remainder
# as +1s keeps the realized sum equal to `total` with at most a 1-sample gap between any two cells.
def compute_quotas(total, num_years, num_types=3):
    base_per_year, year_remainder = divmod(total, num_years)
    quotas = []
    for i in range(num_years):
        year_total = base_per_year + (1 if i < year_remainder else 0)
        base_per_type, type_remainder = divmod(year_total, num_types)
        quotas.append([base_per_type + (1 if j < type_remainder else 0) for j in range(num_types)])
    return quotas

# Each annotator receives `sample_size` docs. `shared_count` of those are identical across
# annotators (same random_id, enabling interrater agreement); the remaining `unique_per` differ
# per annotator. Total docs pulled from the corpus = shared_count + num_annot * unique_per.
# At perc_overlap = 1 this collapses to sample_size (current behaviour, no extra pull).
shared_count = round(sample_size * perc_overlap)
unique_per = sample_size - shared_count

shared_quotas_per_year = compute_quotas(shared_count, len(years))   # [top, bot, rand] shared sub-quota per year
unique_quotas_per_year = compute_quotas(unique_per, len(years))     # [top, bot, rand] per-annotator unique sub-quota per year

# Top/bottom keyword-count stratification only makes sense when the source files carry the
# keyword column (index 7) worth contrasting on — natively true at the filter_* stage. The
# 'auto' default keeps that stage heuristic: stratify for filter_* targets, fully random
# otherwise. '--stratify on' forces stratification regardless of target, which is what lets
# us draw filter-style top/bottom/random samples from label_ outputs that still descend from
# filter_keywords_adv and thus retain the keyword column. '--stratify off' forces fully random.
stratify_mode = getattr(args, "stratify", "auto")
if stratify_mode == "on":
    stratify = True
elif stratify_mode == "off":
    stratify = False
else:  # "auto"
    stratify = target.startswith("filter_")
if not stratify:
    shared_quotas_per_year = [[0, 0, sum(cells)] for cells in shared_quotas_per_year]
    unique_quotas_per_year = [[0, 0, sum(cells)] for cells in unique_quotas_per_year]

# Combined per-(year, type) reservoir size collected by filter_sample_year. The reservoir
# is then partitioned post-collection into the shared chunk and N annotator-unique chunks.
quotas_per_year = [
    [sy[k] + num_annot * uy[k] for k in range(3)]
    for sy, uy in zip(shared_quotas_per_year, unique_quotas_per_year)
]

sampling_mode_desc = (
    "stratified by keyword count (top / bottom / random thirds)"
    if stratify
    else "fully random (no keyword stratification)"
)

# Dictionary to store final samples for each annotator
all_samples = {i: [] for i in range(num_annot)}

# Module-level state shared across years so dedup spans the whole run.
seen_ids = set()         # original_ids already pulled into some reservoir
random_ids_used = set()  # blinded random_ids already issued (globally unique within this run)

### Path Handling

# set path variables

# Survey the input files and raise an error if an expected file is missing

# determine input folder
if not args.input: # assumes the default folder structure and naming conventions for the repository
    sample_path = os.path.join(DATA_DIR,"data_reddit_curated", group, type_, '{}ed_{}'.format(target.split('_')[0],"_".join(target.split('_')[1:])))
else:
    sample_path = args.input

# Organize input files by year
files_by_year = {year: [] for year in years}
if type_ == "comments":
    prefix = "RC"
elif type_ == "submissions":
    prefix = "RS"
elif type_ == "all":
    prefix = "ALL"
else:
    raise Exception("Wrong data type specified. Choose from [comments, submissions, all].")
for year in years:
    for month in range(1, 13):
        path_ = os.path.join(sample_path, f"{prefix}_{year}-{month:02d}.csv")
        if os.path.exists(path_):
            files_by_year[year].append(path_)
        else:
            raise Exception(
                f"Missing {prefix} file for year {year}, month {month}. Expected path: {path_}"
            )

# determine output folder
if not args.output:
    output_dir = os.path.join(DATA_DIR,
        "samples",
        group,
        type_
    )
else:
    output_dir = args.output

os.makedirs(output_dir, exist_ok=True)

# Report file path (placed in the project directory)
report_file_path = os.path.join(output_dir, f"Report_FilterSample.csv")

### Main Functions

# Extract and count unique social group-related keywords from the input string.
def get_unique_keywords(keyword_str, max_keywords=100):
    try:
        # Split by comma and clean each keyword
        keywords = keyword_str.replace('\t', ',').split(',')
        # Use a set for uniqueness
        cleaned_keywords = set()
        for kw in keywords:
            kw = kw.strip()
            # Example: 'fat:' or 'thin:' special logic, if needed
            if '{}:'.format(groups[args.group][0]) in kw or '{}:'.format(groups[args.group][1]) in kw:
                parts = kw.split(':')
                if len(parts) > 1:
                    cleaned_keywords.add(f"{parts[0].strip()}: {parts[1].strip()}")
            elif kw:
                cleaned_keywords.add(kw)
        unique_keywords = list(cleaned_keywords)[:max_keywords]
        return unique_keywords, len(unique_keywords)
    except Exception as e:
        log_report(report_file_path, f"Error processing keywords: {e}")
        return [], 0

# Shuffle a per-(year, type) reservoir and slice it into (shared_chunk, [per-annotator unique chunks]).
# On shortfall (reservoir shorter than the full target), fill the shared chunk first up to its
# quota, then distribute whatever's left across the annotator-unique chunks as evenly as possible.
# Shared-first prioritises the docs that matter most for interrater agreement.
def partition_reservoir(reservoir, shared_q, unique_q, n_annot):
    random.shuffle(reservoir)
    full_target = shared_q + n_annot * unique_q
    if len(reservoir) >= full_target:
        shared_chunk = reservoir[:shared_q]
        unique_chunks = [
            reservoir[shared_q + a * unique_q : shared_q + (a + 1) * unique_q]
            for a in range(n_annot)
        ]
        return shared_chunk, unique_chunks
    # Shortfall path: shared-first, then even split of the remainder.
    shared_chunk = reservoir[:min(shared_q, len(reservoir))]
    leftover = reservoir[len(shared_chunk):]
    base, rem = divmod(len(leftover), n_annot)
    unique_chunks = []
    cursor = 0
    for a in range(n_annot):
        sz = base + (1 if a < rem else 0)
        unique_chunks.append(leftover[cursor:cursor + sz])
        cursor += sz
    return shared_chunk, unique_chunks


# Issue a fresh 6-digit random_id that hasn't been used elsewhere in this run.
def fresh_random_id():
    while True:
        rid = random.randint(100000, 999999)
        if rid not in random_ids_used:
            random_ids_used.add(rid)
            return rid


# Oversample factor: each reservoir holds up to RESERVOIR_FUDGE × its true quota during the
# file-read loop so the post-dedup trim still has at least the quota's worth of unique docs.
# Without this buffer, the small fraction of docs that legitimately land in multiple reservoirs
# (e.g. a top-by-count doc that also gets picked into the random pool) would cause us to fall
# short of the per-annotator target. 1.5× is plenty for production-scale corpora (where the
# collision rate is effectively zero) and absorbs synthetic-test densities up to a few %.
RESERVOIR_FUDGE = 1.5


def _fudged(q):
    return int(q * RESERVOIR_FUDGE) + 1 if q > 0 else 0


# Process each year
def filter_sample_year(year, file_list_for_year,
                       top_quota_total, bottom_quota_total, random_quota_total,
                       shared_top, shared_bot, shared_rand,
                       unique_top, unique_bot, unique_rand):
    log_report(report_file_path, f"Started sampling documents for year {year} in group {args.group}.")
    print(f"\nSampling documents for the {args.group} social group from year {year}...")

    # Reservoirs (lists) for top, bottom, random — oversampled to FUDGE × the true (shared +
    # N*unique) per-type quota, so we have headroom for cross-reservoir dedup losses.
    top_fudged = _fudged(top_quota_total)
    bottom_fudged = _fudged(bottom_quota_total)
    random_fudged = _fudged(random_quota_total)
    top_reservoir = []
    bottom_reservoir = []
    random_reservoir = []

    total_docs = 0  # How many docs processed for this year

    # Iterate through each file for this year
    for file in file_list_for_year:
        print(f"Sampling from {Path(file).name}")
        try:
            with open(file, "r", encoding='utf-8-sig', errors='ignore') as input_file:
                reader = csv.reader(x.replace('\0', '') for x in input_file)
                for id_, line in enumerate(reader):
                    # Skip the header row
                    if id_ == 0:
                        continue
                    try:
                        # Basic row validation: must have at least 3 columns for text
                        if line and len(line) > 2 and line[2].strip():

                            # Extract original_id from first column
                            original_id = line[0].strip()
                            if original_id in seen_ids:
                                continue

                            seen_ids.add(original_id)
                            text = line[2].strip().replace("\n", " ")

                            # If there's a keywords column (index 7), parse it
                            if len(line) > 7:
                                keywords, unique_count = get_unique_keywords(line[7])
                            else:
                                keywords, unique_count = [], 0

                            total_docs += 1

                            #    TOP SAMPLES: more unique keywords
                            if len(top_reservoir) < top_fudged:
                                top_reservoir.append((unique_count, text, keywords, file, original_id))
                                top_reservoir.sort(key=lambda x: x[0], reverse=True)
                            elif top_fudged > 0 and unique_count > top_reservoir[-1][0]:
                                top_reservoir[-1] = (unique_count, text, keywords, file, original_id)
                                top_reservoir.sort(key=lambda x: x[0], reverse=True)

                            #    BOTTOM SAMPLES: fewer unique keywords
                            if len(bottom_reservoir) < bottom_fudged:
                                bottom_reservoir.append((unique_count, text, keywords, file, original_id))
                                bottom_reservoir.sort(key=lambda x: x[0])
                            elif bottom_fudged > 0 and unique_count < bottom_reservoir[-1][0]:
                                bottom_reservoir[-1] = (unique_count, text, keywords, file, original_id)
                                bottom_reservoir.sort(key=lambda x: x[0])

                            #    RANDOM SAMPLES
                            if len(random_reservoir) < random_fudged:
                                random_reservoir.append((unique_count, text, keywords, file, original_id))
                            elif random_fudged > 0:
                                s = random.randint(0, total_docs - 1)
                                if s < random_fudged:
                                    random_reservoir[s] = (unique_count, text, keywords, file, original_id)
                        else:
                            log_report(report_file_path, f"Skipping line {id_} in file {file}: insufficient columns ({len(line)} found)")
                    except Exception as e:
                        log_error("filter_sample_year", file, id_ + 1, str(line), e)
                        continue
        except Exception as e:
            log_error("filter_sample_year", file, 0, "File-level error", e) # Line number for sampling error defaults to 0.
            continue

    log_report(report_file_path, f"{total_docs} documents processed for year {year} in group {args.group}.")

    # Cross-reservoir deduplication. A doc can legitimately land in multiple reservoirs at
    # once (e.g., a high-keyword-count doc fills the top reservoir AND happens to get picked
    # into the random reservoir). Without this step the same original_id would be emitted
    # under multiple random_ids — breaking the shared/unique partition invariant that each
    # doc is either shared across all annotators or unique to one. Priority order top > bottom
    # > random keeps the more informative bucket assignment when a tie occurs.
    chosen_ids = set()
    def _dedup(reservoir):
        out = []
        for entry in reservoir:
            oid = entry[4]
            if oid in chosen_ids:
                continue
            chosen_ids.add(oid)
            out.append(entry)
        return out
    top_reservoir = _dedup(top_reservoir)
    bottom_reservoir = _dedup(bottom_reservoir)
    random_reservoir = _dedup(random_reservoir)

    # Trim each reservoir back to its true quota. Top is sorted desc and bottom asc by
    # unique_count, so slicing the head preserves the by-count semantic. Random was filled
    # by reservoir sampling, so a head-slice is itself a uniform random sub-sample of what
    # remained after dedup.
    top_reservoir = top_reservoir[:top_quota_total]
    bottom_reservoir = bottom_reservoir[:bottom_quota_total]
    random_reservoir = random_reservoir[:random_quota_total]

    # Warn if any reservoir came up short of its combined quota — partition_reservoir will
    # still produce sensible chunks (shared bucket filled first), but the user should know.
    shortfalls = [
        (label, len(reservoir), quota)
        for label, reservoir, quota in (
            ("top", top_reservoir, top_quota_total),
            ("bottom", bottom_reservoir, bottom_quota_total),
            ("random", random_reservoir, random_quota_total),
        )
        if len(reservoir) < quota
    ]
    for label, got, want in shortfalls:
        msg = (
            f"WARNING: year {year} {label}-sample pool short — got {got}/{want} "
            f"(only {total_docs} eligible doc(s) seen across all months); "
            f"shared bucket filled first, remainder split across {num_annot} annotator(s)."
        )
        print(f"  {msg}")
        log_report(report_file_path, msg)

    # Partition each reservoir into (shared, [per-annotator unique chunks]).
    shared_top_docs, unique_top_chunks = partition_reservoir(top_reservoir, shared_top, unique_top, num_annot)
    shared_bot_docs, unique_bot_chunks = partition_reservoir(bottom_reservoir, shared_bot, unique_bot, num_annot)
    shared_rand_docs, unique_rand_chunks = partition_reservoir(random_reservoir, shared_rand, unique_rand, num_annot)

    # Shared docs: one fresh random_id per doc, appended to every annotator's list with the
    # SAME record (so downstream interrater scoring can match them by random_id).
    for docs, sample_type in (
        (shared_top_docs, "top_sample"),
        (shared_bot_docs, "bottom_sample"),
        (shared_rand_docs, "random_sample"),
    ):
        for (_, text, keywords, file, original_id) in docs:
            record = {
                'random_id': fresh_random_id(),
                'text': text,
                'keywords': keywords,
                'file': file,
                'sample_type': sample_type,
                'original_id': original_id,
            }
            for annot in range(num_annot):
                all_samples[annot].append(record)

    # Annotator-unique docs: fresh random_id, appended to that one annotator only.
    for annot in range(num_annot):
        for docs, sample_type in (
            (unique_top_chunks[annot], "top_sample"),
            (unique_bot_chunks[annot], "bottom_sample"),
            (unique_rand_chunks[annot], "random_sample"),
        ):
            for (_, text, keywords, file, original_id) in docs:
                all_samples[annot].append({
                    'random_id': fresh_random_id(),
                    'text': text,
                    'keywords': keywords,
                    'file': file,
                    'sample_type': sample_type,
                    'original_id': original_id,
                })

# After processing all years, write output
def filter_sample_write(all_samples):

    # use numbered tags to prevent overwriting previous samples
    tag = 0
    while True:
        paths_exist = False
        for annot in range(num_annot):
            sample_file_path = os.path.join(output_dir, f"filter_sample_{annot}_v{tag}.csv")
            sample_key_file_path = os.path.join(output_dir, f"filter_sample_{annot}_v{tag}_key.csv")
            if os.path.isfile(sample_file_path) or os.path.isfile(sample_key_file_path):
                paths_exist = True
                break
        if not paths_exist:
            break
        tag += 1

    # write all annotator files using the same tag
    for annot in range(num_annot):
        sample_file_path = os.path.join(output_dir, f"filter_sample_{annot}_v{tag}.csv")
        sample_key_file_path = os.path.join(output_dir, f"filter_sample_{annot}_v{tag}_key.csv")

        with open(sample_file_path, "w", encoding='utf-8', newline='') as sample_file, \
             open(sample_key_file_path, "w", encoding='utf-8', newline='') as sample_file_key:
            
            writer = csv.writer(sample_file)
            writer_key = csv.writer(sample_file_key)
            
            # Write headers
            writer.writerow(["random_id", "text"])
            writer_key.writerow(["random_id", "file", "original_id", "keywords", "sample_type"])
            
            # Shuffle samples before writing so we don't group them year by year
            random.shuffle(all_samples[annot])
            
            # Write rows
            for data in all_samples[annot]:
                writer.writerow([data['random_id'], data['text'], "", "", ""])
                writer_key.writerow([
                    data['random_id'],
                    data['file'],
                    data['original_id'],
                    ",".join(data['keywords']),
                    data['sample_type']
                ])

### Main Execution

if __name__ == "__main__":
    start_time = time.time()
    # Announce the chosen sampling approach so the output is self-documenting.
    src_label = args.input if args.input else f"default folder for stage '{target}'"
    total_pull = shared_count + num_annot * unique_per
    startup_msg = (
        f"Sampling approach: {sampling_mode_desc}. "
        f"Source: {src_label}. "
        f"Per-annotator target: {sample_size}; num_annotators: {num_annot}; "
        f"perc_overlap: {perc_overlap:.3f} (shared={shared_count}, unique_per_annotator={unique_per}). "
        f"Total docs to pull from corpus: {total_pull} across {len(years)} year(s)."
    )
    print(startup_msg)
    log_report(report_file_path, startup_msg)
    # Process each year with only its corresponding files
    for year_idx, year in enumerate(years):
        file_list = files_by_year[year]
        top_q, bot_q, rand_q = quotas_per_year[year_idx]
        sh = shared_quotas_per_year[year_idx]
        un = unique_quotas_per_year[year_idx]
        filter_sample_year(year, file_list,
                           top_q, bot_q, rand_q,
                           sh[0], sh[1], sh[2],
                           un[0], un[1], un[2])
    filter_sample_write(all_samples)
    elapsed = (time.time() - start_time) / 60
    per_annot_counts = [len(all_samples[a]) for a in range(num_annot)]
    final_msg = (
        f"Reservoir sampling for the {group} social group from {args.years} finished in "
        f"{elapsed:.2f} minutes. Per-annotator sample counts: {per_annot_counts}."
    )
    print(final_msg)
    log_report(report_file_path, final_msg)