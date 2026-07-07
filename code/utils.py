import argparse
import csv
import random
import math
import os
import sys
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Literal, NamedTuple, Tuple, Optional, Any
import zstandard
import io
import json
import sqlite3


class StreamingScanYield(NamedTuple):
    """One yield from iter_author_feature_map_streaming. The cumulative dicts
    carry the running state across all yields so far; the per-file dicts hold
    only this yield's contribution. Overlap dicts are non-empty only when the
    scanner was given curated_seen_ids + target_month_basenames."""
    files_just_done: List[str]
    per_file_deltas: Dict[str, Tuple[Dict[str, Dict[str, int]], Dict[str, int], Dict[str, Dict[str, int]], Dict[str, int]]]
    cumulative_counts: Dict[str, Dict[str, int]]
    cumulative_seen: Dict[str, int]
    cumulative_overlap_counts: Dict[str, Dict[str, int]]
    cumulative_overlap_seen: Dict[str, int]
    files_remaining: int

## Shared constants used across the codebase

# The list of social groups. Marginalized groups always listed first.
groups = {
    "sexuality": ["gay", "straight"],
    "age": ["old", "young"],
    "weight": ["fat", "thin"],
    "ability": ["disabled", "abled"],
    "race": ["black", "white"],
    "skin_tone": ["dark", "light"],
    "mental_health": ["mental_health", "not_mental_health"],
}

# Information stored for each comment in ISAAC output files
headers = ["id", "parent id", "text", "author", "time", "subreddit", "score", "matched patterns"]

# default resource order
default_resource = ["filtered_keywords","filtered_language","filtered_relevance","filtered_keywords_adv","labeled_moralization","labeled_sentiment","labeled_generalization","labeled_emotion","labeled_location"]

# Basic parsing / validation helpers (used by multiple scripts)

# confirm that input year spec is valid
def validate_years(years_str: str, parser: argparse.ArgumentParser) -> None:
    """Validate a year spec.

    Accepts any comma-separated combination of single years ('YYYY') and
    contiguous ranges ('YYYY-YYYY'); e.g. '2019', '2019-2023', or
    '2007,2009,2011-2017'. All years must satisfy 2007 <= y <= 2023.
    Whitespace around commas/tokens is tolerated.
    """
    tokens = [t.strip() for t in years_str.split(",") if t.strip()]
    if not tokens:
        parser.error("--years must not be empty.")

    for tok in tokens:
        match = re.fullmatch(r"(\d{4})(?:-(\d{4}))?", tok)
        if not match:
            parser.error(
                "--years tokens must each be a 4-digit year or a range like "
                f"2010-2015; got '{tok}'."
            )
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start

        if not (2007 <= start <= 2023 and 2007 <= end <= 2023):
            parser.error(f"Years must be between 2007 and 2023; got '{tok}'.")
        if start > end:
            parser.error(f"Start year must be ≤ end year in '{tok}'.")

# process input year spec
def parse_range(value: str) -> List[int]:
    """Parse a year spec into a sorted, deduplicated list of years.

    Accepts a single year ('YYYY'), a contiguous range ('YYYY-YYYY'), or any
    comma-separated combination thereof (e.g. '2007,2009,2011-2017'). All
    years must satisfy 2007 <= y <= 2023. Overlapping or duplicate tokens are
    collapsed; the returned list is sorted ascending so it lines up with the
    chronological file ordering used downstream by check_reqd_files callers.
    """
    tokens = [t.strip() for t in value.split(",") if t.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError(
            f"Invalid value '{value}': must contain at least one year."
        )

    years: set = set()
    for tok in tokens:
        try:
            if "-" in tok:
                start, end = map(int, tok.split("-", 1))
                if start > end:
                    raise argparse.ArgumentTypeError(
                        f"Invalid range '{tok}': start must be ≤ end."
                    )
            else:
                start = end = int(tok)

            if start < 2007:
                raise argparse.ArgumentTypeError(
                    f"Invalid value '{tok}': years must be ≥ 2007."
                )
            if end > 2023:
                raise argparse.ArgumentTypeError(
                    f"Invalid value '{tok}': years must be ≤ 2023."
                )

            years.update(range(start, end + 1))
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid token '{tok}' in '{value}': must be an integer or a "
                f"range (e.g., 2007 or 2008-2010)."
            )

    return sorted(years)

# Calculate SLURM array span from a year spec.
def array_span_from_years(years_str: str) -> int:
    """Number of monthly array tasks needed to cover the given year spec.

    Equals 12 × (number of unique years parsed from `years_str`). Supports
    sparse specs like '2007,2009,2011-2017' as well as the legacy 'YYYY' /
    'YYYY-YYYY' forms.
    """
    return len(parse_range(years_str)) * 12

# Load newline-delimited keywords, lowercased; skip blanks.
def load_terms(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.lower().rstrip("\r\n") for line in f if line.strip()]

## Resume helpers (used by all filter_/label_/organize_ resources)

# Stream the existing output CSV forward with csv.reader and return the
# largest integer found in its `source_row` column. Returns -1 if:
#   - file doesn't exist
#   - file is empty / has only a header
#   - header lacks a `source_row` column
#   - no data row has a parseable source_row value
# Returns -1 (not 0) so callers can distinguish "no resume point" from
# "first data row already processed" (source_row=0). Callers should resume
# by skipping rows whose source_row <= returned value.
#
# We deliberately do NOT walk backward by byte to find the last row: many
# of our resources write fields (e.g., post body / title+body) that contain
# embedded newlines, so a single CSV record physically spans multiple
# lines. A byte-level reverse scan can't tell whether a `\n` is a record
# terminator or content inside a quoted field, and on submissions it almost
# always lands inside the body — producing a bogus partial "last row" and
# triggering a silent overwrite of a complete output. Streaming forward
# with csv.reader respects quoting and so is correct on multi-line records.
def get_last_source_row(output_file_path: str | Path,
                        report_file_path: Optional[str] = None,
                        file_for_log: Optional[str] = None) -> int:
    output_file_path = str(output_file_path)
    if not os.path.exists(output_file_path):
        return -1

    try:
        with open(output_file_path, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)

            if not header:
                return -1

            try:
                source_idx = header.index("source_row")
            except ValueError:
                if report_file_path and file_for_log:
                    log_report(
                        report_file_path,
                        f"Warning: Could not find 'source_row' column in existing output "
                        f"for {Path(file_for_log).name}. Restarting from beginning."
                    )
                return -1

            last_good_source_row = -1
            try:
                for row in reader:
                    if source_idx >= len(row):
                        continue
                    try:
                        src = int(row[source_idx])
                    except (ValueError, TypeError):
                        continue
                    # source_row is monotonic in our pipeline; max() is
                    # equivalent to last-seen but tolerant of any future
                    # writer that flushes rows out of order.
                    if src > last_good_source_row:
                        last_good_source_row = src
            except csv.Error as e:
                if report_file_path and file_for_log:
                    log_report(
                        report_file_path,
                        f"Warning: CSV parse error in existing output for {Path(file_for_log).name}; "
                        f"resuming from source_row={last_good_source_row}. If a prior run was killed "
                        f"mid-row, delete the file to force a clean re-run. ({e})"
                    )

            return last_good_source_row

    except Exception as e:
        if report_file_path and file_for_log:
            log_report(
                report_file_path,
                f"Warning: Could not determine resume position for {Path(file_for_log).name}. "
                f"Restarting from beginning. Error: {e}"
            )
        return -1


# Inspect an input CSV header. Returns (source_row_idx, has_source_row).
# When has_source_row is True, callers should propagate values from input
# instead of generating their own.
def detect_source_row(in_header: List[str]) -> Tuple[int, bool]:
    try:
        return in_header.index("source_row"), True
    except ValueError:
        return -1, False


## Logging helpers (used by multiple scripts)

def log_report(report_file_path: Optional[str] = None, message: Optional[str] = None) -> None:
    if message is None:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if report_file_path:
        os.makedirs(os.path.dirname(report_file_path), exist_ok=True)
        with open(report_file_path, "a", encoding="utf-8", newline="") as report_file:
            writer = csv.writer(report_file)
            writer.writerow([timestamp, message])
    print(f"{timestamp} - {message}")
    sys.stdout.flush()


def reraise_fatal(report_file_path: Optional[str], context: str, error: BaseException) -> None:
    """Log a fatal processing error and RE-RAISE so the task exits non-zero.

    Use this in a resource script's top-level processing handler instead of
    logging-and-continuing. Swallowing an error there is dangerous: an OS I/O
    error mid-write (e.g. ENOSPC / "Disk quota exceeded", errno 122) leaves a
    TRUNCATED output file on disk while the task still exits 0 and shows as
    COMPLETED in sacct -- silently corrupting that month's output. OS I/O
    errors get a specific, actionable message; everything else gets a full
    traceback. The log write may itself fail when the disk is full, so it is
    guarded; the original exception is re-raised regardless, and its traceback
    still reaches the Slurm .err file.

    `context` is a human-readable label for the unit of work (filename or
    group/years scope). Must be called from within an `except` block.
    """
    import traceback
    if isinstance(error, OSError):
        errno_val = getattr(error, "errno", None)
        detail = os.strerror(errno_val) if errno_val is not None else str(error)
        msg = (
            f"[fatal] OS I/O error while processing {context} "
            f"(errno={errno_val}: {detail}). Output is likely TRUNCATED and must "
            f"be re-run. Re-raising to fail the task."
        )
    else:
        msg = f"[fatal] Unexpected error while processing {context}:\n{traceback.format_exc()}"
    try:
        log_report(report_file_path, msg)
    except Exception:
        pass
    raise error


def log_error(
    function_name: str,
    file: str,
    line_number: int,
    line_content: str,
    error: Exception,
    report_file_path: Optional[str] = None,
    output_path: Optional[str] = None,  # kept for backward-compat; no longer used
) -> None:
    """
    Log a per-row recoverable error to the report CSV as a single line.

    Intended for data-level errors where the right action is "skip this row
    and continue" (e.g., malformed CSV row in filter_keywords). Does NOT
    create a separate file per error -- callers that previously relied on
    `output_path` to dump an `error_<...>.txt` should expect a single
    entry in the report CSV instead.

    For fatal infrastructure errors (OOM, CUDA, model failures), do NOT
    use this -- log via `log_report` with a clear "[fatal] ..." prefix
    and re-raise so the task exits non-zero.
    """
    try:
        resource_identifier = os.path.basename(file)
        snippet = line_content if len(line_content) <= 200 else line_content[:200] + "..."
        message = (
            f"[error] {function_name} | {resource_identifier} | line {line_number} | "
            f"{type(error).__name__}: {error} | content: {snippet}"
        )
        if report_file_path:
            log_report(report_file_path, message)
        else:
            # Fall back so the error isn't swallowed entirely.
            print(message, file=sys.stderr, flush=True)
    except Exception:
        pass

def f1_calculator(labels,predictions):

    metrics = {i:0 for i in ['tp','tn','fp','fn']}

    for idx,prediction in enumerate(predictions):
        
            if labels[idx] == 0:
                if prediction == 0:
                    metrics['tn'] += 1
                elif prediction == 1:
                    metrics['fp'] += 1
                else:
                    raise Exception
            elif labels[idx] == 1:
                if prediction == 0:
                    metrics['fn'] += 1
                elif prediction == 1:
                    metrics['tp'] += 1
                else:
                    raise Exception

    precision = float(metrics['tp']) / float(metrics['tp'] + metrics['fp'])
    recall = float(metrics['tp']) / float(metrics['tp'] + metrics['fn'])
    F_1 = 2 * float(precision * recall) / float(precision + recall)

    return precision, recall, F_1

## File discovery helpers

FolderType = Literal["comments", "submissions"]
def detect_reddit_folder_type(folder: str | Path) -> FolderType:
    folder = Path(folder)

    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder}")

    csv_files = [p.name for p in folder.iterdir() if p.is_file() and p.suffix == ".csv"]

    has_rc = any(name.startswith("RC") for name in csv_files)
    has_rs = any(name.startswith("RS") for name in csv_files)

    if has_rc and has_rs:
        raise ValueError(
            f"Folder {folder} contains both RC and RS CSV files. Provide an unambiguous input folder."
        )
    if has_rc:
        return "comments"
    if has_rs:
        return "submissions"

    raise FileNotFoundError(
        f"Folder {folder} contains no Reddit CSV files with RC or RS prefixes."
    )

def check_reqd_files(years: List[int], check_path: str | Path, type_: str) -> List[str]:
    PREFIX_MAP = {
        "comments": "RC",
        "submissions": "RS",
        "all": "ALL",
    }

    prefix = PREFIX_MAP.get(type_)
    if not prefix:
        raise ValueError(f"Invalid type_: {type_}")

    check_path = Path(check_path)
    if not check_path.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {check_path}")

    all_files = sorted(
        p for p in check_path.iterdir()
        if p.is_file() and p.suffix == ".csv" and p.name.startswith(prefix)
    )

    matched_files: List[str] = []
    files_by_year: Dict[str, set] = {str(y): set() for y in years}

    for p in all_files:
        m = re.search(r"(\d{4})-(\d{2})", p.name)
        if not m:
            continue

        year, month = m.groups()

        if year in files_by_year:
            files_by_year[year].add(month)
            matched_files.append(str(p))

    if not matched_files:
        raise FileNotFoundError(
            f"No files found in {check_path} for type_={type_} and years={years}"
        )

    expected_months = {f"{m:02d}" for m in range(1, 13)}
    missing_by_year: Dict[int, List[str]] = {}
    for y in years:
        missing = expected_months - files_by_year.get(str(y), set())
        if missing:
            missing_by_year[y] = sorted(missing)

    if missing_by_year:
        summary = "\n".join(
            f"  {y}: {', '.join(months)}"
            for y, months in sorted(missing_by_year.items())
        )
        raise FileNotFoundError(
            f"Missing required {type_} input files in {check_path}.\n"
            f"Strict mode: refusing to return a partial file_list because doing so "
            f"would silently shift the array-index -> month mapping in downstream "
            f"resource scripts. Each task computes file_list[task_id], so a gap at "
            f"month N causes every task index >= N to process the wrong month, and "
            f"tail tasks past the end of the compacted list to no-op as COMPLETED.\n"
            f"Missing months by year:\n{summary}\n"
            f"Fix the upstream pipeline stage that should have produced these months, "
            f"or restrict --years to a span with complete coverage."
        )

    return matched_files

def find_latest_resource_dir(base_dir: str | Path, default_resource: List[str]) -> Path:
    base_dir = Path(base_dir)
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Base curated directory does not exist: {base_dir}")

    curated_folders = {p.name for p in base_dir.iterdir() if p.is_dir()}

    for resource in reversed(default_resource):
        if resource in curated_folders:
            return base_dir / resource

    raise ValueError(
        f"No matching resource found in {base_dir}. "
        f"Expected one of: {', '.join(default_resource)}"
    )

def validate_resource_dir(path: str | Path, default_resource: List[str]) -> Path:
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"Input path is not a directory: {path}")

    if path.name not in default_resource:
        raise ValueError(
            f"{path.name} does not correspond to a proper curated dataset. "
            f"Choose from: {', '.join(default_resource)}"
        )

    return path

# for location model word features
def resolve_word_feature_src(type_arg: str) -> str:
    src = (type_arg or WORD_FEATURE_SRC).strip().lower()
    if src not in {"comments", "submissions", "all"}:
        raise ValueError("type / WORD_FEATURE_SRC must be one of: comments, submissions, all")
    return src

## Dataset splitting utilities

# splits data into train/test with given proportion
def dataset_split(users: List[Any], labels: List[Any], proportion: float, seed: Optional[int] = None):
    if seed is not None:
        random.seed(seed)

    n = len(users)
    k = math.floor(proportion * n)
    training_id = set(random.sample(range(n), k))
    test_id = [i for i in range(n) if i not in training_id]

    training_users, training_labels, test_users, test_labels = [], [], [], []
    for idx, u in enumerate(users):
        if idx in training_id:
            training_users.append(u)
            training_labels.append(labels[idx])
        else:
            test_users.append(u)
            test_labels.append(labels[idx])

    return training_users, test_users, training_labels, test_labels

# Write data splits to file
def split_dataset_to_file(file: str, items: List[Any]) -> None:
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, "w", encoding="utf-8", errors="ignore", newline="") as f:
        if ("users" in file) or ("text" in file):
            writer = csv.writer(f)
            for i in items:
                writer.writerow([i])
        elif "label" in file:
            for i in items:
                print(i, file=f)
        else:
            # fallback: csv
            writer = csv.writer(f)
            for i in items:
                writer.writerow([i])

# Read data splits from file
def split_dataset_from_file(file: str, label_cast: Optional[type] = None) -> List[Any]:
    items: List[Any] = []
    with open(file, "r", encoding="utf-8", errors="ignore") as f:
        if ("users" in file) or ("text" in file):
            reader = csv.reader(f)
            for row in reader:
                if row:
                    items.append(row[0])
        elif "label" in file:
            for line in f:
                v = line.strip()
                if label_cast is not None:
                    try:
                        v = label_cast(v)
                    except Exception:
                        pass
                items.append(v)
        else:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    items.append(row[0])
    return items

# Create or load an 80/10/10 split and make sure it persists.
def prepare_splits(users: List[Any], labels: List[Any], split_dir: str, description: str = ""):
    os.makedirs(split_dir, exist_ok=True)

    split_data = ["training", "validation", "test"]
    file_list: List[str] = []
    for cat in split_data:
        file_list.append(os.path.join(split_dir, f"users_{cat}.csv"))
        file_list.append(os.path.join(split_dir, f"label_{cat}.txt"))

    missing_file = any(not os.path.exists(f) for f in file_list)

    if missing_file:
        print(f"Creating {description} training, validation and test sets (80/10/10 split)")
        train_users, valid_users_init, train_labels, valid_labels_init = dataset_split(
            users, labels, proportion=0.8
        )
        valid_users, test_users, valid_labels, test_labels = dataset_split(
            valid_users_init, valid_labels_init, proportion=0.5
        )

        split_dataset_to_file(file_list[0], train_users)
        split_dataset_to_file(file_list[1], train_labels)
        split_dataset_to_file(file_list[2], valid_users)
        split_dataset_to_file(file_list[3], valid_labels)
        split_dataset_to_file(file_list[4], test_users)
        split_dataset_to_file(file_list[5], test_labels)
    else:
        print(f"Loading predetermined {description} training, validation and test sets (80/10/10 split)")
        train_users = split_dataset_from_file(file_list[0])
        train_labels = split_dataset_from_file(file_list[1])
        valid_users = split_dataset_from_file(file_list[2])
        valid_labels = split_dataset_from_file(file_list[3])
        test_users = split_dataset_from_file(file_list[4])
        test_labels = split_dataset_from_file(file_list[5])

    return train_users, train_labels, valid_users, valid_labels, test_users, test_labels

# summarize information about the data split
def summarize_split(name: str, users: List[Any], labels: List[Any]) -> None:
    print(f"Number of {name} documents: {len(users)}")
    print(f"Number of instances for each label in {name} data: {Counter(labels)}")

## Location labeling helpers

_token_re = re.compile(r"[a-z0-9']+")

# Simple tokenizer aligned with the location model's word feature style.
def tokenize(text: str) -> List[str]:
    return _token_re.findall((text or "").lower())

# Parse a timestamp string to hour [0..23]. Supports ISO-like and unix seconds.
def parse_time_to_hour(time_str: str) -> Optional[int]:
    if not time_str:
        return None
    s = str(time_str).strip()
    # unix seconds?
    if s.isdigit():
        try:
            return datetime.fromtimestamp(int(s)).hour
        except Exception:
            return None
    # common formats: 'YYYY-mm-dd HH:MM:SS' or ISO
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).hour
        except Exception:
            pass
    # try fromisoformat
    try:
        return datetime.fromisoformat(s).hour
    except Exception:
        return None

# update a user's sparse counts in-place from a single row
# word_vocab / subreddit_vocab are optional vocabularies; when supplied, tokens
# or subreddits not in the set are skipped. OOV features are dropped by the
# downstream model's word_index / struct_vectorizer anyway, so accumulating
# them just wastes memory during the raw .zst scan. Pass None to disable
# filtering (preserves the original behavior for callers that don't have a
# model vocab on hand, e.g. preprocessing/training pipelines).
def add_features_for_row(
    counts: Dict[str, int],
    text: str,
    subreddit: str,
    time_value: str,
    word_vocab: Optional[set] = None,
    subreddit_vocab: Optional[set] = None,
) -> None:
    if word_vocab is None:
        for tok in tokenize(text):
            k = f"w:{tok}"
            counts[k] = counts.get(k, 0) + 1
    else:
        for tok in tokenize(text):
            if tok not in word_vocab:
                continue
            k = f"w:{tok}"
            counts[k] = counts.get(k, 0) + 1

    if subreddit:
        s = subreddit.strip()
        if s and (subreddit_vocab is None or s in subreddit_vocab):
            k = f"s:{s}"
            counts[k] = counts.get(k, 0) + 1

    hr = parse_time_to_hour(time_value)
    if hr is not None:
        k = f"h:{hr:02d}"
        counts[k] = counts.get(k, 0) + 1

# First pass over an input CSV: aggregate sparse features per author.
def build_author_feature_map_from_csv(
    file_path: str | Path,
    author_col: str = "author",
    text_col: str = "text",
    subreddit_col: str = "subreddit",
    time_col: str = "time",
) -> Dict[str, Dict[str, int]]:
    author_to_counts: Dict[str, Dict[str, int]] = {}
    with open(file_path, "r", encoding="utf-8-sig", errors="ignore") as f:
        reader = csv.DictReader((line.replace("\x00", "") for line in f))
        if reader.fieldnames is None:
            return author_to_counts

        for row in reader:
            author = (row.get(author_col) or "").strip()
            if not author or author == "[deleted]":
                continue
            counts = author_to_counts.get(author)
            if counts is None:
                counts = {}
                author_to_counts[author] = counts
            add_features_for_row(
                counts,
                text=row.get(text_col, ""),
                subreddit=row.get(subreddit_col, ""),
                time_value=row.get(time_col, ""),
            )
    return author_to_counts

# Raw Reddit (.zst) reading helpers for location labeling

# Yield decoded JSON objects from a .zst file (one JSON per line).
def iter_zst_json_lines(file_path: str | Path):

    file_path = str(file_path)
    with open(file_path, "rb") as fh:
        dctx = zstandard.ZstdDecompressor(max_window_size=2 ** 31)
        stream_reader = dctx.stream_reader(fh, read_across_frames=True)
        text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8")
        for line in text_stream:
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

# Stream one or more raw .zst month files and build sparse features per author.
# NOTE: Only collects up to max_items_per_author posts/comments per author to cap work. For submissions, text is title + selftext.
# NOTE: Returns author -> counts dict with keys w:/s:/h:

def build_author_feature_map_from_raw_zst(
    raw_files: List[str | Path],
    target_authors: set[str],
    type_: str,
    max_items_per_author: int = 100,
    word_vocab: Optional[set] = None,
    subreddit_vocab: Optional[set] = None,
) -> Dict[str, Dict[str, int]]:
    """Wrapper returning only feature counts (without per-author seen counts)."""
    author_to_counts, _author_seen = build_author_feature_map_from_raw_zst_with_seen(
        raw_files=raw_files,
        target_authors=target_authors,
        type_=type_,
        max_items_per_author=max_items_per_author,
        word_vocab=word_vocab,
        subreddit_vocab=subreddit_vocab,
    )
    return author_to_counts

# Process one contiguous chunk of raw .zst files, returning partial (counts, seen) dicts.
# Used by the parallel scan path; no early-exit since chunks run concurrently.
def _scan_raw_file_chunk(
    file_chunk: List[str],
    target_authors,
    type_: str,
    max_items_per_author: int,
    word_vocab: Optional[set] = None,
    subreddit_vocab: Optional[set] = None,
    curated_seen_ids: Optional[Dict[str, set]] = None,
    target_month_basenames: Optional[set] = None,
    remaining_quota_per_author: Optional[Dict[str, int]] = None,
) -> Dict[str, Tuple[Dict[str, Dict[str, int]], Dict[str, int], Dict[str, Dict[str, int]], Dict[str, int]]]:
    """Scan a chunk of raw .zst files for the indicated authors.

    `target_authors` accepts either of:
      - set[str]: legacy mode -- scan every file in the chunk for the same
        author set (full spiral, no per-file targeting).
      - Dict[str, set[str]]: per-file targeting mode -- keys are file BASENAMES
        (e.g. "RC_2007-03.zst"), values are the set of authors we want to
        collect from that file. Files whose basename is absent from the dict
        are skipped entirely; an empty set value also skips the file.

    `curated_seen_ids` (Dict[author, set[post_id]]) plus `target_month_basenames`
    (set of file basenames) enable the per-post dedup correction: when scanning
    a target-month file, posts whose id is in curated_seen_ids[author] are
    counted into the per-author overlap dict in addition to the regular
    per-author file counts. The caller then subtracts overlap from
    regular-file-counts at inference time so each post is counted exactly once
    across (curated, raw) sources.

    Returns Dict[basename, (file_counts, file_seen, file_overlap_counts,
    file_overlap_seen)]. The third and fourth elements are empty dicts for
    non-target-month files.
    """
    per_file_mode = isinstance(target_authors, dict)
    curated_seen_ids = curated_seen_ids or {}
    target_month_basenames = target_month_basenames or set()
    remaining_quota_per_author = remaining_quota_per_author or {}
    # Chunk-cumulative seen across files in this chunk -- the across-files cap
    # is min(max_items_per_author, remaining_quota_per_author[author]) so the
    # author's total contribution from this chunk's scan respects the same
    # cumulative cap the original (pre-streaming) scanner enforced.
    chunk_seen: Dict[str, int] = {}
    out: Dict[str, Tuple[Dict[str, Dict[str, int]], Dict[str, int], Dict[str, Dict[str, int]], Dict[str, int]]] = {}
    for rf in file_chunk:
        basename = os.path.basename(rf)
        if per_file_mode:
            file_targets = target_authors.get(basename)
            if not file_targets:
                continue
        else:
            file_targets = target_authors
        is_target_month_file = basename in target_month_basenames
        file_counts: Dict[str, Dict[str, int]] = {}
        file_seen: Dict[str, int] = {}
        file_overlap_counts: Dict[str, Dict[str, int]] = {}
        file_overlap_seen: Dict[str, int] = {}
        for obj in iter_zst_json_lines(rf):
            author = (obj.get("author") or "").strip()
            if not author or author not in file_targets:
                continue
            author_cap = min(
                max_items_per_author,
                remaining_quota_per_author.get(author, max_items_per_author),
            )
            if chunk_seen.get(author, 0) >= author_cap:
                continue
            if file_seen.get(author, 0) >= max_items_per_author:
                continue
            if type_ == "comments":
                text = (obj.get("body") or "")
                subreddit = (obj.get("subreddit") or "")
            else:
                title = (obj.get("title") or "")
                body = (obj.get("selftext") or "")
                text = (title + "\n" + body).strip()
                subreddit = (obj.get("subreddit") or "")
            created_utc = obj.get("created_utc", "")
            counts = file_counts.get(author)
            if counts is None:
                counts = {}
                file_counts[author] = counts
            add_features_for_row(
                counts,
                text=text,
                subreddit=subreddit,
                time_value=str(created_utc),
                word_vocab=word_vocab,
                subreddit_vocab=subreddit_vocab,
            )
            file_seen[author] = file_seen.get(author, 0) + 1
            chunk_seen[author] = chunk_seen.get(author, 0) + 1
            if is_target_month_file:
                post_id = (obj.get("id") or "").strip()
                if post_id and post_id in curated_seen_ids.get(author, ()):  # tuple sentinel = empty set
                    overlap_dst = file_overlap_counts.get(author)
                    if overlap_dst is None:
                        overlap_dst = {}
                        file_overlap_counts[author] = overlap_dst
                    add_features_for_row(
                        overlap_dst,
                        text=text,
                        subreddit=subreddit,
                        time_value=str(created_utc),
                        word_vocab=word_vocab,
                        subreddit_vocab=subreddit_vocab,
                    )
                    file_overlap_seen[author] = file_overlap_seen.get(author, 0) + 1
        # Materialize even no-hit authors so callers can persist
        # "scanned-but-empty" rows: a file we opened for author X where X
        # didn't appear. These rows let the next overlapping-spiral month
        # know that f was already scanned for X and skip the file.
        for a in file_targets:
            if a not in file_counts:
                file_counts[a] = {}
                file_seen[a] = 0
        out[basename] = (file_counts, file_seen, file_overlap_counts, file_overlap_seen)
    return out


# Stream one or more raw .zst files and build sparse features per author.
# Generator variant of the raw-zst scanner. Yields (files_just_done, author_to_counts,
# author_seen, files_remaining) after each raw .zst file finishes (sequential path) or
# after each thread-pool chunk finishes (parallel path). The yielded dicts are the
# CUMULATIVE state -- mutating them outside breaks invariants; callers should snapshot
# anything they need to retain. Always yields at least once (the empty-target shortcut
# yields once with files_remaining == 0).
#
# Used by label_location_month() to flush rows for saturated authors mid-scan, so a
# task that crashes during the multi-day raw scan can resume from the last incremental
# flush instead of re-scanning every raw file from the top.
def iter_author_feature_map_streaming(
    raw_files: List[str | Path],
    target_authors,
    type_: str,
    max_items_per_author: int = 100,
    n_scan_workers: int = 1,
    word_vocab: Optional[set] = None,
    subreddit_vocab: Optional[set] = None,
    curated_seen_ids: Optional[Dict[str, set]] = None,
    target_month_basenames: Optional[set] = None,
    remaining_quota_per_author: Optional[Dict[str, int]] = None,
):
    """`target_authors` may be either:
      - set[str]: scan every raw_file for the same author set (legacy mode).
      - Dict[basename, set[str]]: per-file targeting. Keys are raw-file
        BASENAMES; values are the authors to look for in that file. Files
        whose basename is absent from the dict (or maps to an empty set) are
        skipped without opening them.

    `curated_seen_ids` and `target_month_basenames` enable the per-post
    dedup correction (see _scan_raw_file_chunk doc).

    Yields StreamingScanYield records (a NamedTuple).
    """
    per_file_mode = isinstance(target_authors, dict)
    if per_file_mode:
        all_target_authors: set = set()
        for s in target_authors.values():
            all_target_authors.update(s)
    else:
        all_target_authors = target_authors

    cumulative_counts: Dict[str, Dict[str, int]] = {}
    cumulative_seen: Dict[str, int] = {a: 0 for a in all_target_authors}
    cumulative_overlap_counts: Dict[str, Dict[str, int]] = {}
    cumulative_overlap_seen: Dict[str, int] = {}
    quota_per_author = remaining_quota_per_author or {}
    # Each author's effective across-the-scan cap is min(max_items_per_author,
    # quota_per_author[author]). Threaded chunks each enforce this cap
    # internally; the post-fold min() below makes the cumulative cap exact even
    # when multiple chunks individually filled up to the cap.

    raw_files = [str(f) for f in raw_files]
    total_files = len(raw_files)

    if not all_target_authors or total_files == 0:
        yield StreamingScanYield([], {}, cumulative_counts, cumulative_seen, cumulative_overlap_counts, cumulative_overlap_seen, 0)
        return

    def _fold_per_file(per_file: Dict[str, Tuple[Dict[str, Dict[str, int]], Dict[str, int], Dict[str, Dict[str, int]], Dict[str, int]]]) -> None:
        for _basename, (file_counts, file_seen, file_overlap_counts, file_overlap_seen) in per_file.items():
            for author, c in file_counts.items():
                if not c:
                    continue
                merged = cumulative_counts.get(author)
                if merged is None:
                    cumulative_counts[author] = dict(c)
                else:
                    for k, v in c.items():
                        merged[k] = merged.get(k, 0) + v
            for author, s in file_seen.items():
                cap = min(max_items_per_author, quota_per_author.get(author, max_items_per_author))
                cumulative_seen[author] = min(cap, cumulative_seen.get(author, 0) + s)
            for author, c in file_overlap_counts.items():
                if not c:
                    continue
                merged = cumulative_overlap_counts.get(author)
                if merged is None:
                    cumulative_overlap_counts[author] = dict(c)
                else:
                    for k, v in c.items():
                        merged[k] = merged.get(k, 0) + v
            for author, s in file_overlap_seen.items():
                cumulative_overlap_seen[author] = cumulative_overlap_seen.get(author, 0) + s

    if n_scan_workers > 1 and total_files > 1:
        from concurrent.futures import as_completed
        n_workers = min(n_scan_workers, total_files)
        chunks = [raw_files[i::n_workers] for i in range(n_workers)]

        def scan_chunk(chunk):
            return _scan_raw_file_chunk(
                chunk, target_authors, type_, max_items_per_author,
                word_vocab=word_vocab, subreddit_vocab=subreddit_vocab,
                curated_seen_ids=curated_seen_ids,
                target_month_basenames=target_month_basenames,
                remaining_quota_per_author=quota_per_author,
            )

        files_remaining = total_files
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            future_to_chunk = {pool.submit(scan_chunk, chunk): chunk for chunk in chunks}
            for fut in as_completed(future_to_chunk):
                chunk = future_to_chunk[fut]
                per_file_chunk = fut.result()
                _fold_per_file(per_file_chunk)
                files_remaining -= len(chunk)
                yield StreamingScanYield(list(chunk), per_file_chunk, cumulative_counts, cumulative_seen, cumulative_overlap_counts, cumulative_overlap_seen, files_remaining)
        return

    # Sequential path: scan one file at a time via the chunk scanner so the
    # per-file delta format is identical to the threaded path. Update the
    # remaining-quota dict between files so the per-author cumulative cap
    # exactly matches the original pre-streaming semantics.
    running_quota = {a: min(max_items_per_author, quota_per_author.get(a, max_items_per_author))
                     for a in all_target_authors}
    for i, rf in enumerate(raw_files):
        per_file_one = _scan_raw_file_chunk(
            [rf], target_authors, type_, max_items_per_author,
            word_vocab=word_vocab, subreddit_vocab=subreddit_vocab,
            curated_seen_ids=curated_seen_ids,
            target_month_basenames=target_month_basenames,
            remaining_quota_per_author=running_quota,
        )
        _fold_per_file(per_file_one)
        # Drain the quota by however much this file consumed.
        for _basename, (_fc, file_seen, _ovc, _ovs) in per_file_one.items():
            for author, s in file_seen.items():
                if s and author in running_quota:
                    running_quota[author] = max(0, running_quota[author] - s)
        files_remaining = total_files - (i + 1)
        yield StreamingScanYield([rf], per_file_one, cumulative_counts, cumulative_seen, cumulative_overlap_counts, cumulative_overlap_seen, files_remaining)


# NOTE: Collects up to max_items_per_author items per author. Returns (author_to_counts, author_seen).
# NOTE: n_scan_workers > 1 splits files across threads (zstd decompression releases the GIL).
#       Use only in single-process contexts (SLURM array tasks); not for multi-process local runs.
# NOTE: word_vocab / subreddit_vocab cap the feature space at scan time -- pass
#       the downstream model's known vocab to avoid building per-author dicts
#       full of OOV tokens that would be dropped during inference anyway.
#
# This non-streaming variant is the legacy one-shot scan; it now consumes the streaming
# generator above and returns the final cumulative state. Callers that don't need mid-scan
# checkpoints (anything outside label_location.py) keep their existing semantics unchanged.
def build_author_feature_map_from_raw_zst_with_seen(
    raw_files: List[str | Path],
    target_authors: set[str],
    type_: str,
    max_items_per_author: int = 100,
    n_scan_workers: int = 1,
    word_vocab: Optional[set] = None,
    subreddit_vocab: Optional[set] = None,
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    author_to_counts: Dict[str, Dict[str, int]] = {}
    author_seen: Dict[str, int] = {a: 0 for a in target_authors}
    for yld in iter_author_feature_map_streaming(
        raw_files=raw_files,
        target_authors=target_authors,
        type_=type_,
        max_items_per_author=max_items_per_author,
        n_scan_workers=n_scan_workers,
        word_vocab=word_vocab,
        subreddit_vocab=subreddit_vocab,
    ):
        # The generator mutates and re-yields the same dicts each step; the
        # final iteration's state is what we want. Legacy callers don't pass
        # curated_seen_ids/target_month_basenames so overlap dicts are empty.
        author_to_counts = yld.cumulative_counts
        author_seen = yld.cumulative_seen
    return author_to_counts, author_seen

# Find raw .zst files for a given year-month. Returns list of full paths.
def find_raw_month_files(raw_dir: str | Path, type_: str, year: int, month: str) -> List[str]:
    raw_dir = str(raw_dir)
    prefix = "RC" if type_ == "comments" else "RS"
    ym = f"{year}-{month}"
    out = []
    for fn in os.listdir(raw_dir):
        if not fn.endswith(".zst"):
            continue
        # common patterns: RC_YYYY-MM.zst, RC_YYYY-MM-*.zst, etc.
        if (prefix in fn) and (ym in fn):
            out.append(os.path.join(raw_dir, fn))
    return sorted(out)

## Persistent author -> location cache (SQLite)

# Run a no-arg callable, retrying on sqlite3.OperationalError 'database is
# locked' with exponential backoff and small jitter. SQLite's busy_timeout
# already handles most lock contention, but startup bursts of many array tasks
# briefly hammer the DB during table-touching reads/writes; this gives those
# operations a few extra chances before they surface as a fatal error. Returns
# the callable's result; non-lock OperationalErrors and other exceptions
# propagate immediately.
def _sqlite_retry_on_locked(operation, *, max_attempts: int = 8, base_delay: float = 0.5, max_delay: float = 8.0):
    last_err: Optional[sqlite3.OperationalError] = None
    for attempt in range(max_attempts):
        try:
            return operation()
        except sqlite3.OperationalError as e:
            if "locked" not in str(e).lower():
                raise
            last_err = e
            delay = min(max_delay, base_delay * (2 ** attempt)) + random.uniform(0.0, 0.5)
            time.sleep(delay)
    if last_err is not None:
        raise last_err

# Initialize the SQLite cache for author->location mapping.
# NOTE: Uses WAL journal mode for better concurrent read/write behavior.
# NOTE: location_prob is stored alongside location to support confidence-aware
#       overwrite in cache_put_locations (only replace cached rows when the new
#       prob beats the cached prob).
def init_location_cache(db_path: str) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    def _do() -> None:
        conn = sqlite3.connect(db_path, timeout=60)
        try:
            cur = conn.cursor()
            # DELETE journal mode (not WAL): the cache DB lives on NFS, and WAL's
            # shared-memory coordination file (-shm) requires real mmap-based
            # shared memory that NFS does not provide. Under concurrent writes
            # from multiple array tasks across nodes, WAL silently corrupts the
            # DB. DELETE mode uses POSIX file locks, which NFS implements
            # correctly (just more slowly). Combined with batching writes to
            # one per label_location_month, the contention is low enough that
            # the slower lock path is irrelevant.
            cur.execute("PRAGMA journal_mode=DELETE;")
            cur.execute("PRAGMA synchronous=NORMAL;")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS author_location (
                    author TEXT PRIMARY KEY,
                    location TEXT NOT NULL,
                    location_prob REAL,
                    updated_at INTEGER NOT NULL
                );
                """
            )
            conn.commit()
        finally:
            conn.close()

    _sqlite_retry_on_locked(_do)

# Initialize the detail table (per-author top/contender labels with probs and tier).
# NOTE: Pre-create alongside init_location_cache from a single process before launching
# parallel Slurm array tasks to avoid "database is locked" races on first WAL setup.
def init_location_detail_cache(db_path: str) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    def _do() -> None:
        conn = sqlite3.connect(db_path, timeout=60)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS author_location_detail (
                    author TEXT PRIMARY KEY,
                    location TEXT NOT NULL,
                    location_prob REAL,
                    contender_location TEXT,
                    contender_location_prob REAL,
                    top_location TEXT,
                    top_location_prob REAL,
                    top_contender_location TEXT,
                    top_contender_location_prob REAL,
                    tier TEXT,
                    seen_count INTEGER,
                    updated_at INTEGER NOT NULL
                );
                """
            )
            conn.commit()
        finally:
            conn.close()

    _sqlite_retry_on_locked(_do)

# Fetch cached locations for a set of authors. Wrapped in a retry loop because
# concurrent Slurm array tasks can briefly contend on schema-touching writes
# at startup, occasionally surfacing 'database is locked' on reads despite WAL.
def cache_get_locations(db_path: str, authors: set[str]) -> Dict[str, str]:
    if not authors:
        return {}
    def _do() -> Dict[str, str]:
        conn = sqlite3.connect(db_path, timeout=60)
        try:
            cur = conn.cursor()
            out: Dict[str, str] = {}
            # sqlite has a variable limit; chunk in 900s to be safe
            author_list = list(authors)
            for i in range(0, len(author_list), 900):
                chunk = author_list[i:i+900]
                qmarks = ",".join(["?"] * len(chunk))
                cur.execute(f"SELECT author, location FROM author_location WHERE author IN ({qmarks})", chunk)
                for a, loc in cur.fetchall():
                    out[a] = loc
            return out
        finally:
            conn.close()
    return _sqlite_retry_on_locked(_do)

# Upsert many author->location mappings with confidence-aware overwrite.
# Each detail dict must contain at least 'location' (str) and 'location_prob'
# (float | None). Cached rows are atomically overwritten only when the new
# location_prob is strictly greater than the cached one (or the cached one is
# NULL); a new prob of None never overwrites an existing labeled row.
def cache_put_locations(db_path: str, details_by_author: Dict[str, Dict[str, Any]]) -> None:
    if not details_by_author:
        return

    def _do() -> None:
        conn = sqlite3.connect(db_path, timeout=60)
        try:
            cur = conn.cursor()
            now = int(time.time())
            rows = []
            for author, d in details_by_author.items():
                if not author:
                    continue
                location = d.get("location")
                if not location:
                    continue
                rows.append((author, str(location), d.get("location_prob"), now))
            if not rows:
                return
            cur.executemany(
                """
                INSERT INTO author_location(author, location, location_prob, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(author) DO UPDATE SET
                    location = excluded.location,
                    location_prob = excluded.location_prob,
                    updated_at = excluded.updated_at
                WHERE excluded.location_prob IS NOT NULL
                  AND (author_location.location_prob IS NULL
                       OR excluded.location_prob > author_location.location_prob)
                """,
                rows,
            )
            conn.commit()
        finally:
            conn.close()

    _sqlite_retry_on_locked(_do)


## Persistent per-(author, raw_file) feature-counts cache (SQLite, separate
## table from the location-decision cache above)
#
# Purpose: amortize raw-zst scan work across runs. Each row records the
# features collected for one author from one raw .zst file. Because the raw
# data is the same physical input regardless of which social group, year, or
# concurrent task triggered the scan, identical (author, file) tuples yield
# identical counts. Writes use INSERT OR IGNORE so concurrent tasks scanning
# the same (author, file) tuple converge on whichever row commits first --
# the second scan's work is wasted but never corrupting (no double-counts,
# no lost-write races).
#
# Aggregation happens at read time: cache_get_author_file_counts returns,
# per author, the union of all cached basenames plus the summed counts and
# seen across those files. Callers then subtract the union from the current
# spiral to compute which files still need scanning.
#
# Storage: roughly 60-200 bytes per row after zstd compression (counts blob
# is sparse with ~5-50 features; seen, timestamps, raw_file basename round
# out the row). A fully-populated corpus row count is bounded by
# unique_authors x average_files_per_author -- order of magnitude ~100M rows
# for 2M authors x 50 files, or ~6-12 GB.


def init_author_file_counts_cache(db_path: str) -> None:
    """Create the author_file_counts table if missing. Idempotent; the inner
    _sqlite_retry_on_locked wrap absorbs brief 'database is locked' contention
    when many array tasks initialize the shared NFS-backed DB simultaneously."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    def _do() -> None:
        conn = sqlite3.connect(db_path, timeout=60)
        try:
            cur = conn.cursor()
            # DELETE journal mode (not WAL): the cache DB lives on NFS, and WAL's
            # shared-memory coordination file (-shm) requires real mmap-based
            # shared memory that NFS does not provide. Under concurrent writes
            # from multiple array tasks across nodes, WAL silently corrupts the
            # DB. DELETE mode uses POSIX file locks, which NFS implements
            # correctly (just more slowly). Combined with batching writes to
            # one per label_location_month, the contention is low enough that
            # the slower lock path is irrelevant.
            cur.execute("PRAGMA journal_mode=DELETE;")
            cur.execute("PRAGMA synchronous=NORMAL;")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS author_file_counts (
                    author TEXT NOT NULL,
                    raw_file TEXT NOT NULL,
                    counts_blob BLOB NOT NULL,
                    seen_count INTEGER NOT NULL,
                    scanned_at REAL NOT NULL,
                    PRIMARY KEY (author, raw_file)
                );
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_author_file_counts_author ON author_file_counts(author);"
            )
            conn.commit()
        finally:
            conn.close()

    _sqlite_retry_on_locked(_do)


# ---------------------------------------------------------------------------
# Year-sharded layout for the location cache.
#
# The author_location / author_location_detail label tables stay in ONE DB
# (keyed by author, which is year-independent -- splitting by year would destroy
# the cross-year dedup that gives the cache its hit rate). The large, regenerable
# author_file_counts table is sharded into one DB per year of raw_file, because
# a file_counts row is intrinsically tied to one raw_file = one year. This spreads
# write contention across year DBs and limits a corruption's blast radius to a
# single (regenerable) year.
# ---------------------------------------------------------------------------

_RAW_FILE_YEAR_RE = re.compile(r"(\d{4})-\d{2}")


def year_of_raw_file(raw_file: str) -> Optional[str]:
    """Extract the YYYY year from a raw-file basename like 'RC_2018-01.zst'."""
    m = _RAW_FILE_YEAR_RE.search(raw_file or "")
    return m.group(1) if m else None


def location_label_db_path(cache_dir: str, rtype: str) -> str:
    """Path to the single DB holding author_location + author_location_detail."""
    return os.path.join(cache_dir, f"author_location_label_{rtype}.sqlite")


def author_file_counts_db_path(cache_dir: str, rtype: str, year) -> str:
    """Path to the per-year author_file_counts DB."""
    return os.path.join(cache_dir, f"author_file_counts_{rtype}_{year}.sqlite")


def init_author_file_counts_caches(cache_dir: str, rtype: str, years) -> None:
    """Pre-create the per-year author_file_counts DBs from a single process so
    parallel array tasks don't race on first-touch CREATE TABLE."""
    for y in years:
        init_author_file_counts_cache(author_file_counts_db_path(cache_dir, rtype, str(y)))


def cache_get_author_file_counts_sharded(
    cache_dir: str,
    rtype: str,
    years,
    authors,
    exclude_basenames: Optional[set] = None,
) -> Dict[str, Tuple[set, Dict[str, int], int]]:
    """Read and merge per-author file-counts across the given year-sharded DBs.

    Each raw_file lives in exactly one year DB, so summing the per-DB aggregates
    never double-counts a file. Missing year DBs (no rows scanned yet) are
    skipped. Return shape matches cache_get_author_file_counts."""
    if not authors:
        return {}
    merged: Dict[str, Tuple[set, Dict[str, int], int]] = {}
    for y in years:
        p = author_file_counts_db_path(cache_dir, rtype, str(y))
        if not os.path.exists(p):
            continue
        part = cache_get_author_file_counts(p, authors, exclude_basenames)
        for a, (scanned, counts, seen) in part.items():
            if a not in merged:
                merged[a] = (set(), {}, 0)
            ms, mc, msn = merged[a]
            ms |= scanned
            for k, v in counts.items():
                mc[k] = mc.get(k, 0) + v
            merged[a] = (ms, mc, msn + seen)
    return merged


def cache_put_author_file_counts_sharded(cache_dir: str, rtype: str, rows) -> int:
    """Route (author, raw_file, counts, seen) rows to per-year DBs by raw_file
    year, lazily creating a year DB on first write. Rows whose raw_file has no
    parseable year are dropped. Returns the number of rows written."""
    rows = list(rows)
    if not rows:
        return 0
    buckets: Dict[str, list] = defaultdict(list)
    for entry in rows:
        y = year_of_raw_file(entry[1])
        if not y:
            continue
        buckets[y].append(entry)
    written = 0
    for y, yrows in buckets.items():
        p = author_file_counts_db_path(cache_dir, rtype, y)
        if not os.path.exists(p):
            init_author_file_counts_cache(p)
        cache_put_author_file_counts(p, yrows)
        written += len(yrows)
    return written


def _zstd_compress_json(obj: Any) -> bytes:
    raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    cctx = zstandard.ZstdCompressor(level=3)
    return cctx.compress(raw)


def _zstd_decompress_json(blob: bytes) -> Any:
    dctx = zstandard.ZstdDecompressor()
    raw = dctx.decompress(blob)
    return json.loads(raw)


# Returns Dict[author, (scanned_files: set[str], aggregated_counts: dict,
# aggregated_seen: int)] for authors with at least one cached row. The
# scanned_files set is the union of basenames cached for that author across
# any prior run (or any group); counts and seen are the row-wise sums.
# Authors without cached rows are absent from the result dict (caller treats
# them as needing a full spiral scan).
#
# `exclude_basenames`: optional set of raw-file basenames whose rows should
# NOT contribute to the aggregated counts/seen, but whose presence still
# counts toward scanned_files. label_location's caller passes the target
# month's basenames here so the aggregated cached_counts excludes features
# from any prior run's scan of THIS run's target month -- otherwise we'd
# double-count the target month's content (once via prior-run cached counts,
# once via this run's forced target-month re-scan).
def cache_get_author_file_counts(
    db_path: str,
    authors,
    exclude_basenames: Optional[set] = None,
) -> Dict[str, Tuple[set, Dict[str, int], int]]:
    if not authors:
        return {}
    exclude_basenames = exclude_basenames or set()

    def _do() -> Dict[str, Tuple[set, Dict[str, int], int]]:
        conn = sqlite3.connect(db_path, timeout=60)
        try:
            cur = conn.cursor()
            out: Dict[str, Tuple[set, Dict[str, int], int]] = {}
            author_list = list(authors)
            for i in range(0, len(author_list), 900):
                chunk = author_list[i:i + 900]
                qmarks = ",".join(["?"] * len(chunk))
                cur.execute(
                    f"SELECT author, raw_file, counts_blob, seen_count FROM author_file_counts WHERE author IN ({qmarks})",
                    chunk,
                )
                for author, raw_file, counts_blob, seen in cur.fetchall():
                    if author not in out:
                        out[author] = (set(), {}, 0)
                    scanned, agg_counts, agg_seen = out[author]
                    scanned.add(raw_file)
                    if raw_file in exclude_basenames:
                        # Row is in the cache (scanned_files reflects it) but
                        # we deliberately do NOT fold its counts/seen into the
                        # aggregate -- the caller will get a fresh scan of
                        # this file and add those counts directly.
                        out[author] = (scanned, agg_counts, agg_seen)
                        continue
                    try:
                        cnts = _zstd_decompress_json(counts_blob)
                    except Exception:
                        # Corrupt row -- treat as scanned-but-empty so we don't
                        # re-scan, and don't add bogus counts.
                        cnts = {}
                    for k, v in cnts.items():
                        agg_counts[k] = agg_counts.get(k, 0) + int(v)
                    out[author] = (scanned, agg_counts, agg_seen + int(seen))
            return out
        finally:
            conn.close()

    return _sqlite_retry_on_locked(_do)


# INSERT OR IGNORE one row per (author, raw_file). `rows` is an iterable of
# (author, raw_file_basename, counts_dict, seen_count). A row already in the
# table for the same (author, raw_file) keeps its existing contents -- this
# is the race-safety guarantee: two concurrent tasks scanning the same
# (author, file) tuple commit identical work, so whichever lands first wins
# and the other is silently dropped.
def cache_put_author_file_counts(
    db_path: str,
    rows,
) -> None:
    rows = list(rows)
    if not rows:
        return

    def _do() -> None:
        conn = sqlite3.connect(db_path, timeout=60)
        try:
            cur = conn.cursor()
            now = time.time()
            db_rows = []
            for entry in rows:
                author, raw_file, counts, seen_count = entry
                if not author or not raw_file:
                    continue
                db_rows.append((
                    author,
                    raw_file,
                    _zstd_compress_json(counts or {}),
                    int(seen_count or 0),
                    now,
                ))
            if not db_rows:
                return
            cur.executemany(
                """
                INSERT OR IGNORE INTO author_file_counts(author, raw_file, counts_blob, seen_count, scanned_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                db_rows,
            )
            conn.commit()
        finally:
            conn.close()

    _sqlite_retry_on_locked(_do)
