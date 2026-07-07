### Imports

# import functions and objects
from cli import get_args, PROJECT_ROOT, DATA_DIR
from utils import (
    parse_range,
    default_resource,
    log_report,
    check_reqd_files,
    find_latest_resource_dir,
    validate_resource_dir,
    detect_reddit_folder_type,
    reraise_fatal,
)

# import Python packages
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # to prevent MKL crash with Torch
import csv
csv.field_size_limit(2**31 - 1) # Increase the field size limit to handle larger fields
import time
from datetime import datetime
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

### Argument Handling

# Extract and transform CLI arguments
args = get_args()
years = parse_range(args.years)
if isinstance(years, int):
    years = [years]

group = args.group

### Path Handling

# prepare the report file
# NOTE: Since the dataset should be complete after this integration, the report gets saved to the project root for easier review.
report_file_path = PROJECT_ROOT / "report_organize_types.csv"

# Set/survey the input folders
comments_base = DATA_DIR / "data_reddit_curated" / group / "comments"  # default
submissions_base = DATA_DIR / "data_reddit_curated" / group / "submissions"  # default

# Resolve input paths
if not args.input and not args.input_2:
    log_report(
        report_file_path,
        "No custom inputs provided. Finding the most advanced curated datasets for comments and submissions based on default pathing and resource order..."
    )

    input_comments = find_latest_resource_dir(comments_base, default_resource)
    input_submissions = find_latest_resource_dir(submissions_base, default_resource)

    log_report(
        report_file_path,
        f"{input_comments.name} identified as the most advanced curated dataset for {group} comments."
    )
    log_report(
        report_file_path,
        f"{input_submissions.name} identified as the most advanced curated dataset for {group} submissions."
    )

elif args.input and args.input_2:
    path_a = validate_resource_dir(args.input, default_resource)
    path_b = validate_resource_dir(args.input_2, default_resource)

    type_a = detect_reddit_folder_type(path_a)
    type_b = detect_reddit_folder_type(path_b)

    if type_a == type_b:
        raise ValueError(
            f"Both input folders appear to be '{type_a}'. "
            "One must be comments (RC) and the other submissions (RS)."
        )

    if type_a == "comments":
        input_comments, input_submissions = path_a, path_b
    else:
        input_comments, input_submissions = path_b, path_a

    log_report(
        report_file_path,
        f"{input_comments.name} identified as the curated dataset for {group} comments from custom input."
    )
    log_report(
        report_file_path,
        f"{input_submissions.name} identified as the curated dataset for {group} submissions from custom input."
    )

else:
    raise ValueError(
        "Provide either both --input and --input_2, or neither. "
        "Supplying only one is not supported."
    )

# Ensure both belong to same resource stage
if input_comments.name != input_submissions.name:
    raise ValueError(
        f"The comments and submissions curated folders do not belong to the same resource. "
        f"Comment: {input_comments.name}; Submissions: {input_submissions.name}"
    )

# generate input file lists
input_comments_file_list = sorted(
    check_reqd_files(years, input_comments, "comments"),
    key=lambda p: Path(p).name,
)
input_submissions_file_list = sorted(
    check_reqd_files(years, input_submissions, "submissions"),
    key=lambda p: Path(p).name,
)

# parse the output path
if not args.output:
    output_path = DATA_DIR / "data_reddit_curated" / group / "all" / input_comments.name
else:
    output_path = Path(args.output)

os.makedirs(output_path, exist_ok=True)

if args.output is not None and not output_path.is_dir():
    raise ValueError("The 'output' argument should be a directory path.")

### Integration Utilities

TIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%m/%d/%Y %H:%M",
)

def parse_time(value: str) -> datetime:
    value = value.strip()
    for fmt in TIME_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(
        f"Invalid time value {value!r}; expected one of: {', '.join(TIME_FORMATS)}"
    )

def find_matching_submission_file(comment_file: str | Path, submission_files: List[str | Path]) -> Path:
    """
    Match RC_YYYY-MM...csv -> RS_YYYY-MM...csv by swapping the prefix only.
    Assumes comments/submissions monthly files have the same suffix after RC/RS.
    """
    comment_file = Path(comment_file)
    expected_name = comment_file.name.replace("RC", "RS", 1)

    matches = [Path(f) for f in submission_files if Path(f).name == expected_name]
    if not matches:
        raise FileNotFoundError(
            f"No matching submissions file found for comment file {comment_file.name}. "
            f"Expected: {expected_name}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple matching submissions files found for {comment_file.name}: "
            f"{[m.name for m in matches]}"
        )
    return matches[0]


def validate_headers(comment_fieldnames: List[str], submission_fieldnames: List[str]) -> List[str]:
    """
    Require same columns in same order, except we will append 'type' on output.
    """
    if comment_fieldnames is None or submission_fieldnames is None:
        raise ValueError("Could not read CSV headers from one or both input files.")

    if len(comment_fieldnames) != len(submission_fieldnames):
        raise ValueError(
            f"Column count mismatch: comments={len(comment_fieldnames)}, "
            f"submissions={len(submission_fieldnames)}"
        )

    if comment_fieldnames != submission_fieldnames:
        raise ValueError(
            "CSV headers differ between comments and submissions files.\n"
            f"comments:    {comment_fieldnames}\n"
            f"submissions: {submission_fieldnames}"
        )

    if "time" not in comment_fieldnames:
        raise ValueError("Expected a 'time' column in input files.")

    if "type" in comment_fieldnames:
        raise ValueError("Input files already contain a 'type' column.")

    return comment_fieldnames + ["type"]


def next_row(reader: csv.DictReader) -> Optional[Dict[str, str]]:
    try:
        return next(reader)
    except StopIteration:
        return None


### Merge Functions

# NOTE: We assume the input files are already sorted from earliest to the timestamp. This is true of the data dump referenced in the repository.
# NOTE: If a comment and a submission are posted at the exact same time, the submission is written first

# Stream-merge two CSVs by ascending 'time', writing an added 'type' column.
# Writes to <output>.tmp first, then atomically renames to <output> on success.
# A crashed run leaves a .tmp behind, which the caller cleans up on startup.
def merge_comment_and_submission_file(
    comment_file: str | Path,
    submission_file: str | Path,
    output_file: str | Path,
) -> None:
    comment_file = Path(comment_file)
    submission_file = Path(submission_file)
    output_file = Path(output_file)
    tmp_file = output_file.with_suffix(output_file.suffix + ".tmp")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with (
        open(comment_file, "r", encoding="utf-8", errors="ignore", newline="") as comment_input,
        open(submission_file, "r", encoding="utf-8", errors="ignore", newline="") as submission_input,
        open(tmp_file, "w", encoding="utf-8", newline="") as out_f,
    ):
        comment_reader = csv.DictReader(comment_input)
        submission_reader = csv.DictReader(submission_input)

        output_fieldnames = validate_headers(
            comment_reader.fieldnames,
            submission_reader.fieldnames,
        )

        writer = csv.DictWriter(out_f, fieldnames=output_fieldnames)
        writer.writeheader()

        comment_row = next_row(comment_reader)
        submission_row = next_row(submission_reader)

        while comment_row is not None and submission_row is not None:
            comment_time = parse_time(comment_row["time"])
            submission_time = parse_time(submission_row["time"])

            if comment_time < submission_time:
                row_out = dict(comment_row)
                row_out["type"] = "comment"
                writer.writerow(row_out)
                comment_row = next_row(comment_reader)
            else:
                row_out = dict(submission_row)
                row_out["type"] = "submission"
                writer.writerow(row_out)
                submission_row = next_row(submission_reader)

        while comment_row is not None:
            row_out = dict(comment_row)
            row_out["type"] = "comment"
            writer.writerow(row_out)
            comment_row = next_row(comment_reader)

        while submission_row is not None:
            row_out = dict(submission_row)
            row_out["type"] = "submission"
            writer.writerow(row_out)
            submission_row = next_row(submission_reader)

    # Atomic publish: rename only after the merge has streamed to the end.
    os.replace(tmp_file, output_file)


# Worker wrapper so the parent process can log cleanly.
# Returns: (output_file, success, error_message)
def merge_one_pair(comment_file: str, submission_file: str, output_file: str) -> tuple[str, bool, str | None]:
    try:
        merge_comment_and_submission_file(
            comment_file=comment_file,
            submission_file=submission_file,
            output_file=output_file,
        )
        return output_file, True, None
    except Exception as e:
        return output_file, False, str(e)


### Batch/array utilities

# Return a list of (year, month) tuples to process.
# NOTE: Local run: all requested months. Slurm array task: only the chunk assigned to this task.
def build_requested_months(years_list: List[int]) -> List[Tuple[int, str]]:
    months = []
    for year in years_list:
        for month in range(1, 13):
            months.append((year, f"{month:02d}"))
    return months

# Build merge jobs as: (year, month, comment_file, submission_file, output_file)
def build_merge_jobs(
    comment_files: List[str | Path],
    submission_files: List[str | Path],
    output_dir: str | Path,
    processed_files: set[str],
    target_month_set: set[Tuple[int, str]],
) -> List[Tuple[int, str, str, str, str]]:
    output_dir = Path(output_dir)

    submission_lookup = {Path(f).name: str(f) for f in submission_files}
    jobs: List[Tuple[int, str, str, str, str]] = []

    for comment_file in sorted(comment_files, key=lambda p: Path(p).name):
        comment_file = Path(comment_file)
        m = re.search(r"(\d{4})-(\d{2})", comment_file.name)
        if not m:
            raise ValueError(f"Could not parse YYYY-MM from comment filename: {comment_file.name}")

        year = int(m.group(1))
        month = m.group(2)

        if (year, month) not in target_month_set:
            continue

        expected_submission_name = comment_file.name.replace("RC", "RS", 1)
        submission_file = submission_lookup.get(expected_submission_name)
        if submission_file is None:
            raise FileNotFoundError(
                f"No matching submissions file found for comment file {comment_file.name}. "
                f"Expected: {expected_submission_name}"
            )

        output_name = comment_file.name.replace("RC", "ALL", 1)
        output_stem = Path(output_name).stem

        # resumability feature: skip already-produced output files by stem
        if output_stem in processed_files:
            continue

        output_file = output_dir / output_name
        jobs.append((year, month, str(comment_file), str(submission_file), str(output_file)))

    return jobs


def get_target_jobs(
    all_jobs: List[Tuple[int, str, str, str, str]],
    array_idx: Optional[int] = None,
    files_per_job: int = 1,
) -> List[Tuple[int, str, str, str, str]]:
    if array_idx is None:
        return all_jobs

    start_idx = array_idx * files_per_job
    end_idx = start_idx + files_per_job
    return all_jobs[start_idx:end_idx]


### Main execution function

def organize_types_parallel() -> None:
    start_time = time.time()

    # Clean up any .tmp files left behind by a crashed prior run before we
    # build the processed_files set; otherwise their stems would falsely look
    # complete to build_merge_jobs.
    for f in os.listdir(output_path):
        if f.endswith(".csv.tmp"):
            try:
                (output_path / f).unlink()
                log_report(report_file_path, f"Removed stale temp file from prior crashed run: {f}")
            except OSError as e:
                log_report(report_file_path, f"Warning: could not remove stale temp file {f}: {e}")

    # Track already-produced month files by stem, e.g. ALL_2012-01
    processed_files = {
        Path(f).stem
        for f in os.listdir(output_path)
        if f.endswith(".csv")
    }

    files_per_job = getattr(args, "files_per_job", 1) or 1
    requested_months = build_requested_months(years)
    target_months = requested_months if args.array is None else requested_months[
        args.array * files_per_job : args.array * files_per_job + files_per_job
    ]
    target_month_set = set(target_months)

    if not target_months:
        log_report(report_file_path, "No target months assigned to this run.")
        return

    # Build independent jobs
    all_jobs = build_merge_jobs(
        input_comments_file_list,
        input_submissions_file_list,
        output_path,
        processed_files,
        target_month_set,
    )

    # Slice the relevant input files to assign to this array task
    target_jobs = get_target_jobs(all_jobs, None if args.array is None else 0, len(all_jobs)) \
        if args.array is None else all_jobs

    # The line above keeps behavior simple:
    # - non-array run: use all jobs
    # - array run: all_jobs is already restricted to this task's target months

    if not target_jobs:
        log_report(report_file_path, "No comment/submission file pairs were found to merge for this run.")
        return

    max_workers = min(4, os.cpu_count() or 1, len(target_jobs))
    log_report(report_file_path, f"Using {max_workers} processes to merge comment/submission file pairs.")

    completed = 0
    failed = 0

    # Log warnings for target months that have no pending output files to build
    job_months = {(year, month) for year, month, _, _, _ in target_jobs}
    missing_input_months = sorted(target_month_set - job_months)

    for year, month in missing_input_months:
        expected_output_stem = f"ALL_{year}-{month}"
        if expected_output_stem not in processed_files:
            log_report(report_file_path, f"Warning: Missing mergeable input file pair for {year}-{month}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {
            executor.submit(merge_one_pair, comment_file, submission_file, output_file): (
                year, month, comment_file, submission_file, output_file
            )
            for year, month, comment_file, submission_file, output_file in target_jobs
        }

        for future in as_completed(future_to_job):
            year, month, comment_file, submission_file, output_file = future_to_job[future]

            try:
                out_file, success, error_msg = future.result()
                if success:
                    completed += 1
                    log_report(
                        report_file_path,
                        f"Merged {Path(comment_file).name} + {Path(submission_file).name} -> {Path(out_file).name}"
                    )
                else:
                    failed += 1
                    log_report(
                        report_file_path,
                        f"Error merging {Path(comment_file).name} + {Path(submission_file).name}: {error_msg}"
                    )
            except Exception as e:
                failed += 1
                log_report(
                    report_file_path,
                    f"Worker crash while merging {Path(comment_file).name} + {Path(submission_file).name}: {e}"
                )

    # Check completeness only for the months this run was responsible for
    processed_months = set()
    for f in os.listdir(output_path):
        if not f.endswith(".csv"):
            continue
        m = re.search(r"(\d{4})-(\d{2})", f)
        if m:
            processed_months.add((int(m.group(1)), m.group(2)))

    missing_output_months = sorted(target_month_set - processed_months)
    for year, month in missing_output_months:
        log_report(report_file_path, f"Warning: Missing output file for {year}-{month}")

    elapsed_time = (time.time() - start_time) / 60
    log_report(
        report_file_path,
        f"Finished merging files. Successful: {completed}, Failed: {failed}, Total: {len(target_jobs)}"
    )

### Main Execution

if __name__ == "__main__":
    overall_start_time = time.time()
    try:
        organize_types_parallel()
    except Exception as e:
        reraise_fatal(report_file_path, "organize_types integration", e)

    total_time = (time.time() - overall_start_time) / 60

    if args.array is None:
        scope_msg = f"{args.years}"
    else:
        files_per_job = getattr(args, "files_per_job", 1) or 1
        requested_months = build_requested_months(years)
        assigned = requested_months[args.array * files_per_job : args.array * files_per_job + files_per_job]
        assigned_str = ", ".join(f"{y}-{m}" for y, m in assigned) if assigned else f"array task {args.array}"
        scope_msg = f"{args.years} (task scope: {assigned_str})"

    log_report(
        report_file_path,
        f"Type integration for {group} for {scope_msg} finished in {total_time:.2f} minutes"
    )