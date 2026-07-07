### Imports

# package imports
import os
import csv
csv.field_size_limit(2**31 - 1) # Increase the field size limit to handle larger fields
import json
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import zstandard
import io
import re
import ahocorasick
from pathlib import Path

# Import functions and objects from local modules
from cli import get_args, PROJECT_ROOT,RAW_DIR, DATA_DIR
from utils import load_terms, groups, headers, parse_range, log_report, log_error, get_last_source_row, reraise_fatal

### Argument handling

# Extract and transform CLI arguments
args = get_args()
type_ = args.type
years = parse_range(args.years)
group = args.group
if isinstance(years, int):
    years = [years]

### Path handling

# Load social group keywords
keyword_path = os.path.join(PROJECT_ROOT,"keywords")
marginalized_words = load_terms(os.path.join(keyword_path, f"{group}_{groups[group][0]}.txt"))
privileged_words = load_terms(os.path.join(keyword_path, f"{group}_{groups[group][1]}.txt"))

# find the raw data folder
if not args.input:
    RAW_DIR = os.path.join(RAW_DIR,type_)
else:
    RAW_DIR = args.input

# Build an Aho-Corasick automaton for fast pattern matching of the keywords
automaton = ahocorasick.Automaton()
for term in marginalized_words:
    automaton.add_word(term.lower(), (groups[group][0], term))
for term in privileged_words:
    automaton.add_word(term.lower(), (groups[group][1], term))
automaton.make_automaton()

# Prepare and inspect the output path
if not args.output:
    output_path = os.path.join(
        DATA_DIR, "data_reddit_curated", group, type_, "filtered_keywords")
else:
    output_path = args.output
os.makedirs(output_path, exist_ok=True)

# Setup the report file (tab-separated format)
output_report_filename = "Report_filter_keywords.csv"
report_file_path = os.path.join(output_path, output_report_filename)

# Output schema: canonical headers + source_row (the 1-indexed JSON-line index
# inside the source .zst file). source_row is the canonical resume key for
# every downstream resource.
out_headers = headers + ["source_row"]

# Setup the report file (tab-separated format)
if not os.path.exists(report_file_path):
    mode = 'w'
else:
    mode = 'a'
with open(report_file_path, mode, encoding='utf-8', newline='') as report_file:
    writer = csv.writer(report_file, delimiter='\t')
    if mode == 'w':
        writer.writerow(["Timestamp", "Message"])

# Return a list of (year, month) tuples to process. 
# NOTE: Local run: all requested months. Slurm array task: only the chunk assigned to this task.

def build_requested_months(years_list):
    months = []
    for year in years_list:
        for month in range(1, 13):
            months.append((year, f"{month:02d}"))
    return months

def get_target_months(years_list, array_idx=None, files_per_job=1):
    all_months = build_requested_months(years_list)

    if array_idx is None:
        return all_months

    start_idx = array_idx * files_per_job
    # Without this guard, an out-of-range array index silently slices to []
    # and the Slurm task exits 0 having processed nothing — looks like success.
    if start_idx >= len(all_months):
        raise IndexError(
            f"--array {array_idx} with files-per-job {files_per_job} maps to "
            f"start_idx {start_idx}, which is out of range for {len(all_months)} requested months"
        )
    end_idx = start_idx + files_per_job
    return all_months[start_idx:end_idx]

files_per_job = getattr(args, "files_per_job", 1) or 1
try:
    target_months = get_target_months(years, args.array, files_per_job)
except IndexError as e:
    log_report(report_file_path, f"[error] {e}; aborting")
    raise SystemExit(2)
target_month_set = set(target_months)

### Main resource functions

# filters a single input file based on the provided keyword sets
def filter_keyword_file(file):

    # Process a single raw Reddit file by filtering for keyword matches.
    # Writes matching lines to an output CSV file.

    file_path = os.path.join(RAW_DIR, file)
    output_csv_file = os.path.join(output_path, f"{file.split('.zst')[0]}.csv")

    buffer = []
    buffer_size = 20
    total_lines = 0
    matched_lines = 0
    start_time = time.time()

    # Resume support: if a partial output exists with a source_row column,
    # pick up from the last source_row written. Legacy outputs without
    # source_row are skipped with a warning so we don't silently overwrite
    # them — the user can delete to force a clean re-run.
    last_processed = -1
    if os.path.exists(output_csv_file):
        last_processed = get_last_source_row(
            output_csv_file,
            report_file_path=report_file_path,
            file_for_log=file,
        )
        if last_processed < 0:
            with open(output_csv_file, "r", encoding="utf-8-sig", errors="ignore", newline="") as existing:
                hdr = next(csv.reader(existing), None)
            if hdr and "source_row" not in hdr:
                log_report(
                    report_file_path,
                    f"Skipping {Path(file).name}: existing output lacks source_row column "
                    f"(legacy file). Delete to force re-run."
                )
                return 0, 0

    mode = "a" if last_processed >= 0 else "w"

    try:
        with open(file_path, 'rb') as fh, \
            open(output_csv_file, mode, newline='', encoding='utf-8') as csv_file:

            writer = csv.writer(csv_file)
            if mode == "w":
                writer.writerow(out_headers)

            dctx = zstandard.ZstdDecompressor(max_window_size=2 ** 31)
            stream_reader = dctx.stream_reader(fh, read_across_frames=True)
            text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')


            for line in text_stream:
                total_lines += 1
                # Resume: skip JSON lines we've already processed in a prior run.
                if total_lines <= last_processed:
                    continue
                try:
                    contents = json.loads(line)
                    if type_ == "comments":
                        match_text = contents.get('body', '').strip().lower()
                    elif type_ == "submissions":
                        sub_title = contents.get('title','').strip().lower()
                        sub_body = contents.get('selftext','').strip().lower()
                        match_text = sub_title + "\n" + sub_body

                    matches = []
                    for _, (category, term) in automaton.iter(match_text):
                        matches.append(f"{category}: {term}")

                    if matches:
                        matched_lines += 1

                        if type_ == "comments":
                            parent_id = contents.get("parent_id","")
                        else:
                            parent_id = ""
                        buffer.append([
                            contents.get("id", ""),
                            parent_id,
                            match_text,
                            contents.get("author", ""),
                            datetime.fromtimestamp(int(contents.get("created_utc", 0))).strftime('%Y-%m-%d %H:%M:%S'),
                            contents.get("subreddit", ""),
                            contents.get("score", ""),
                            ', '.join(list(set(matches))),
                            total_lines,
                        ])

                        if len(buffer) >= buffer_size:
                            writer.writerows(buffer)
                            buffer.clear()

                except Exception as e:
                    log_error(
                    "filter_keyword_file",
                    file,
                    total_lines,
                    line,
                    e,
                    report_file_path=report_file_path,
                    output_path=output_path,
                )



            if buffer:
                writer.writerows(buffer)

    except Exception as e:
        log_report(report_file_path, f"Error filtering by keywords in file {Path(file).name}: {e}")

    return total_lines, matched_lines

# wrapper for filter_keyword_file
def filter_keyword_month(year, month, files):
    log_report(report_file_path, f"Started filtering files for {year}-{month} for relevance to the {group} social group based on keywords.")
    start_time = time.time()
    total_lines = 0
    matched_lines = 0

    for file in files:
        try:
            file_lines, file_matched = filter_keyword_file(file)
            total_lines += file_lines
            matched_lines += file_matched
        except Exception as e:
            log_report(report_file_path, f"Error filtering by keywords in file {Path(file).name}: {e}")

    elapsed_time = (time.time() - start_time) / 60
    log_report(report_file_path, f"Completed keyword filtering {year}-{month} for the {group} social group in {elapsed_time:.2f} minutes. Total lines: {total_lines}, matched lines: {matched_lines}")
    return total_lines, matched_lines

# Process files in parallel while checking for missing month files.
# NOTE: Logs warnings for missing months before processing if not running on the cluster.
def filter_keyword_parallel():
    total_lines = 0
    matched_lines = 0
    max_workers = min(3, os.cpu_count() or 1)
    log_report(report_file_path, f"Using {max_workers} processes for parallel processing.")

    # Group eligible raw files by requested (year, month)
    files_by_year_month = {}

    for file in sorted(os.listdir(RAW_DIR)):
        if not file.endswith(".zst"):
            continue

        # Simple parse based on existing filename convention
        try:
            year = int(file.split("_")[1].split("-")[0]) if "_" in file else int(file.split("-")[0])
            month = file.split("-")[1].split(".zst")[0]
        except Exception:
            # More robust fallback: search for YYYY-MM anywhere in filename
            m = re.search(r"(\d{4})-(\d{2})", file)
            if not m:
                continue
            year = int(m.group(1))
            month = m.group(2)

        if (year, month) not in target_month_set:
            continue

        files_by_year_month.setdefault((year, month), []).append(file)

    # Log warnings for target months that have no input files
    missing_input_months = sorted(target_month_set - set(files_by_year_month.keys()))
    for year, month in missing_input_months:
        log_report(report_file_path, f"Warning: Missing input file(s) for {year}-{month}")

    if not target_months:
        log_report(report_file_path, "No target months assigned to this run.")
        return

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_month = {}

        for year, month in target_months:
            files = files_by_year_month.get((year, month), [])
            if not files:
                continue
            future = executor.submit(filter_keyword_month, year, month, files)
            future_to_month[future] = (year, month)

        for future, (year, month) in future_to_month.items():
            try:
                month_lines, month_matched = future.result()
                total_lines += month_lines
                matched_lines += month_matched
            except Exception as e:
                log_report(report_file_path, f"Error filtering by keywords for {year}-{month}: {e}")

    # Check completeness only for the months this run was responsible for
    processed_months = set()
    for f in os.listdir(output_path):
        if not f.endswith(".csv") or f == output_report_filename:
            continue
        m = re.search(r"(\d{4})-(\d{2})", f)
        if m:
            processed_months.add((int(m.group(1)), m.group(2)))

    missing_output_months = sorted(target_month_set - processed_months)
    for year, month in missing_output_months:
        log_report(report_file_path, f"Warning: Missing output file for {year}-{month}")

    log_report(report_file_path, f"Total lines processed: {total_lines}")
    log_report(report_file_path, f"Total matched lines: {matched_lines}")

### Main execution

if __name__ == "__main__":
    overall_start_time = time.time()
    try:
        filter_keyword_parallel()
    except Exception as e:
        reraise_fatal(report_file_path, f"filter_keywords {group} {args.years}", e)

    total_time = (time.time() - overall_start_time) / 60

    if args.array is None:
        scope_msg = f"{args.years}"
    else:
        assigned = ", ".join(f"{y}-{m}" for y, m in target_months) if target_months else f"array task {args.array}"
        scope_msg = f"{args.years} (task scope: {assigned})"

    log_report(
        report_file_path,
        f"Keyword filtering for {group} for {scope_msg} finished in {total_time:.2f} minutes"
    )