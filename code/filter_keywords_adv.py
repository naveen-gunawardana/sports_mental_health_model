### Imports

# Import functions and objects
from cli import get_args, DATA_DIR, PROJECT_ROOT
from utils import parse_range, log_report, log_error, load_terms, groups, check_reqd_files, get_last_source_row, detect_source_row, reraise_fatal

# Import Python packages
import os, time
import csv
csv.field_size_limit(2**31 - 1) # Increase the field size limit to handle larger fields
import hyperscan as hs
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import re

### Argument Handling

# Extract and transform CLI arguments
args = get_args()
years = parse_range(args.years)
type_ = args.type
group = args.group
files_per_job = getattr(args, "files_per_job", 1)
if files_per_job is None or files_per_job < 1:
    files_per_job = 1
array = args.array if args.array is not None else None

### Path Handling

# set the input folder
if not args.input:
    input_path = DATA_DIR / "data_reddit_curated" / group / type_ / "filtered_relevance"
else:
    input_path = Path(args.input)

# Load social group advanced regular experession sets
keyword_path = os.path.join(PROJECT_ROOT,"keywords")
marginalized_words = load_terms(os.path.join(keyword_path, f"{group}_{groups[group][0]}_adv.txt"))
privileged_words   = load_terms(os.path.join(keyword_path, f"{group}_{groups[group][1]}_adv.txt"))
all_words = marginalized_words + privileged_words

# set and survey the output folder
if not args.output:
    output_path = DATA_DIR / "data_reddit_curated" / group / type_ / "filtered_keywords_adv"
else:
    output_path = Path(args.output)
output_path.mkdir(parents=True, exist_ok=True)
file_list = check_reqd_files(years, input_path, type_)

# prepare the report file
report_file_path = os.path.join(output_path, "Report_filter_keywords_adv.csv")

### Filtering Hyperparameters/initializations

# CSV header index assumptions:
MATCHES_COL_INDEX = 7
BODY_COL_INDEX = 2

# Runtime Chimera detection for processing more complex patterns
CH = getattr(hs, "chimera", None)  # None if the installed wheel doesn't expose Chimera
ENGINE = "chimera" if CH is not None else "hyperscan_prefilter"

# Globals populated per process
db = None                  # Hyperscan/Chimera database
id2label = None            # pattern_id -> "Category: term"
compiled_re_by_id = None   # only for fallback: pattern_id -> Python re.Pattern

### Helper Functions

def _build_id2label(m_words, p_words, group_key):
    mapping = {}
    mid = len(m_words)
    for i, term in enumerate(m_words, start=1):
        mapping[i] = f"{groups[group_key][0]}: {term}"
    for j, term in enumerate(p_words, start=mid + 1):
        mapping[j] = f"{groups[group_key][1]}: {term}"
    return mapping


# Compile either a Chimera DB (if available) or a Hyperscan prefilter DB plus
# Python re confirmers for exact semantics.
def _compile_db(patterns):
    global ENGINE, compiled_re_by_id

    ids = list(range(1, len(patterns) + 1))

    if ENGINE == "chimera":
        db_local = CH.Database()
        db_local.compile(
            expressions=[p.encode("utf-8") for p in patterns],
            ids=ids,
            flags=[hs.HS_FLAG_CASELESS] * len(patterns),
        )
        compiled_re_by_id = None  # not needed
        return db_local

    # Fallback: pure Hyperscan with PREFILTER + Python re confirmation
    # PREFILTER allows unsupported constructs (lookarounds) by approximating them.
    db_local = hs.Database()
    db_local.compile(
        expressions=[p.encode("utf-8") for p in patterns],
        ids=ids,
        flags=[hs.HS_FLAG_CASELESS | hs.HS_FLAG_PREFILTER] * len(patterns),
    )

    # Build Python regexes for exact confirmation (PCRE-like features incl. lookahead)
    compiled = {}
    for i, pat in enumerate(patterns, start=1):
        try:
            compiled[i] = re.compile(pat, re.IGNORECASE)
        except re.error as e:
            # If a pattern can't compile in Python re (rare), fall back to a safe literal
            compiled[i] = re.compile(re.escape(pat), re.IGNORECASE)
            log_report(report_file_path, f"[WARN] Python re compile fallback for pattern ID {i}: {e}")
    compiled_re_by_id = compiled
    return db_local

# intialize the multiprocessing workers: load terms, build id2label, compile DB (Chimera or HS prefilter).
def _worker_init(group_key, keyword_path):
    global db, id2label, marginalized_words, privileged_words, all_words

    mfile = os.path.join(keyword_path, f"{group_key}_{groups[group_key][0]}_adv.txt")
    pfile = os.path.join(keyword_path, f"{group_key}_{groups[group_key][1]}_adv.txt")

    marginalized_words = load_terms(mfile)
    privileged_words = load_terms(pfile)
    all_words = marginalized_words + privileged_words

    id2label = _build_id2label(marginalized_words, privileged_words, group_key)
    db = _compile_db(all_words)

    # tiny warm-up
    def _noop_cb(*args, **kwargs):
        return 0

    db.scan(b"warmup", _noop_cb, context=[])

# Read a CSV file from input path, scan BODY_COL_INDEX with multi-regex, write matched rows to output_path.
def filter_keyword_adv_file(file_name):
    input_path = Path(file_name)
    out_stem = input_path.with_suffix("").name
    output_csv_file = output_path / f"{out_stem}.csv"

    log_report(report_file_path, f"[{ENGINE}] Started advanced regex filtering {input_path.name} for relevance to the {group} social group")
    total_lines = 0
    matched_lines = 0
    start_time = time.time()

    def on_match(pattern_id, from_, to_, flags_, context):
        context.append(pattern_id)
        return 0

    # Resume: skip rows already written under a prior run.
    last_processed = get_last_source_row(
        output_csv_file,
        report_file_path=report_file_path,
        file_for_log=str(input_path),
    )
    mode = "a" if last_processed >= 0 else "w"

    try:
        with open(input_path, "r", encoding="utf-8-sig", errors="ignore", newline="") as inp, \
             open(output_csv_file, mode, encoding="utf-8", newline="") as outp:

            reader = csv.reader(inp)
            writer = csv.writer(outp)

            try:
                in_header = next(reader)
            except StopIteration:
                # Empty input: still emit the expected fallback header on a fresh write.
                if mode == "w":
                    writer.writerow(["id", "parent_id", "body", "author", "created_utc", "subreddit", "score", "matches"])
                return 0, 0

            src_idx_in, has_input_source_row = detect_source_row(in_header)

            # Pass through input header verbatim if it has source_row;
            # otherwise append source_row so downstream resources can resume.
            if mode == "w":
                if has_input_source_row:
                    writer.writerow(in_header)
                else:
                    writer.writerow(list(in_header) + ["source_row"])

            for row_idx, row in enumerate(reader, start=1):
                total_lines += 1
                try:
                    if len(row) <= BODY_COL_INDEX:
                        continue

                    # Determine source_row for resume + output.
                    if has_input_source_row:
                        src_val = row[src_idx_in].strip() if len(row) > src_idx_in else ""
                        try:
                            src_num = int(src_val)
                        except ValueError:
                            src_num = None
                    else:
                        src_num = row_idx

                    if src_num is not None and src_num <= last_processed:
                        continue

                    text_str = row[BODY_COL_INDEX]
                    text_bytes = text_str.encode("utf-8", "ignore")

                    # Collect candidate IDs
                    match_ids = []
                    db.scan(text_bytes, on_match, context=match_ids)

                    if not match_ids:
                        continue

                    # If using HS prefilter, confirm each candidate ID with Python re
                    if ENGINE != "chimera":
                        uniq = sorted(set(match_ids))
                        confirmed = []
                        for pid in uniq:
                            regex = compiled_re_by_id.get(pid)
                            if regex is not None and regex.search(text_str):
                                confirmed.append(pid)
                        match_ids = confirmed

                    if match_ids:
                        matched_lines += 1
                        if has_input_source_row:
                            writer.writerow(row)
                        else:
                            writer.writerow(list(row) + [row_idx])

                except Exception as e:
                    log_error("filter_keyword_adv_file", str(input_path), total_lines, row, e)

    except Exception as e:
        log_report(report_file_path, f"Error filtering by advanced keywords in file {input_path.name}: {e}")

    elapsed_time = (time.time() - start_time) / 60
    log_report(
        report_file_path,
        f"[{ENGINE}] Filtered {input_path.name} for relevance to the {group} social group based on advanced regex in {elapsed_time:.2f} minutes. "
        f"Total lines: {total_lines}, matched lines: {matched_lines}"
    )
    return total_lines, matched_lines

def _select_files_for_this_run():
    if array is None:
        return file_list

    start = array * files_per_job
    end = min(start + files_per_job, len(file_list))
    if start >= len(file_list):
        raise RuntimeError(
            f"Array index {array} is out of range for {len(file_list)} files "
            f"(files_per_job={files_per_job})."
        )

    selected_files = file_list[start:end]
    log_report(
        report_file_path,
        f"Array task {array}: processing file_list[{start}:{end}] (files_per_job={files_per_job})."
    )
    for selected_file in selected_files:
        log_report(report_file_path, f"Selected input file: {Path(selected_file).name}")
    return selected_files

### Main Function

def filter_keyword_adv_parallel():
    total_lines = 0
    matched_lines = 0
    max_workers = min(3, os.cpu_count())
    log_report(report_file_path, f"Using {max_workers} processes for parallel processing. Engine: {ENGINE}")

    initargs = (group, keyword_path)
    selected_files = _select_files_for_this_run()

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_worker_init, initargs=initargs) as executor:
        futures = [executor.submit(filter_keyword_adv_file, str(file)) for file in selected_files]

        for future in futures:
            try:
                t, m = future.result()
                total_lines += t
                matched_lines += m
            except Exception as e:
                log_report(report_file_path, f"Error filtering by advanced keywords: {e}")

    # Only perform the whole-range file-count sanity check in non-array mode.
    if array is None:
        expected_file_count = len(file_list)
        actual_file_count = sum(
            1 for f in os.listdir(output_path)
            if f.endswith(".csv") and f != Path(report_file_path).name
        )
        if actual_file_count != expected_file_count:
            log_report(
                report_file_path,
                f"Warning: Expected {expected_file_count} output files, but generated {actual_file_count}."
            )

    log_report(report_file_path, f"Total lines processed: {total_lines}")
    log_report(report_file_path, f"Total matched lines: {matched_lines}")

# Main Execution

if __name__ == "__main__":
    overall_start_time = time.time()
    try:
        filter_keyword_adv_parallel()
    except Exception as e:
        reraise_fatal(report_file_path, f"filter_keywords_adv {group} {args.years} [{ENGINE}]", e)
    total_time = (time.time() - overall_start_time) / 60
    log_report(report_file_path, f"Advanced keyword filtering for {group} for {args.years} finished in {total_time:.2f} minutes [{ENGINE}]")
