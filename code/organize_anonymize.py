### Imports

import csv
csv.field_size_limit(2**31 - 1) # Increase the field size limit to handle larger fields
import os
import re
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# import functions and objects
from cli import get_args, PROJECT_ROOT, DATA_DIR
from utils import (
    parse_range,
    check_reqd_files,
    default_resource,
    log_report,
    find_latest_resource_dir,
    validate_resource_dir,
    get_last_source_row,
    reraise_fatal,
)

### Argument Handling

# Extract and transform CLI arguments
args = get_args()
years = parse_range(args.years)
if isinstance(years, int):
    years = [years]

group = args.group
type_ = args.type

if type_ not in {"comments", "submissions", "all"}:
    raise ValueError(f"Unsupported 'type' argument: {type_}")

### Path Handling

# SQLite file storing author -> anonymized numeric ID mapping
USER_CACHE = DATA_DIR / "user_map.sqlite3"
# NOTE: this is where the key connecting user IDs to numbers lives. Keep safe.
# NOTE: label_location should be used before anonymizing, as it requires access to user IDs.

# prepare the report file
# NOTE: Since the dataset should be complete after this integration, the report gets saved to the project root for easier review.
report_file_path = PROJECT_ROOT / "report_organize_anonymize.csv"

# Set/survey the input folders
data_base = DATA_DIR / "data_reddit_curated" / group / type_  # default

if not args.input:
    log_report(
        report_file_path,
        f"No custom input path provided. Finding the most advanced curated datasets of type '{type_}' based on default pathing and resource order..."
    )
    input_path = find_latest_resource_dir(data_base, default_resource)

    log_report(
        report_file_path,
        f"{input_path.name} identified as the most advanced curated dataset for {group} entries of type '{type_}'."
    )
else:
    input_path = validate_resource_dir(args.input, default_resource)

file_list = check_reqd_files(years=years, type_=type_, check_path=input_path)
file_list = sorted(file_list, key=lambda p: Path(p).name)

# parse the output path
if not args.output:
    output_path = PROJECT_ROOT / "data" / "data_reddit_curated" / group / type_ / f"{Path(input_path).name}_anon"
else:
    output_path = Path(args.output)

os.makedirs(output_path, exist_ok=True)

if args.output is not None and not output_path.is_dir():
    raise ValueError("The 'output' argument should be a directory path.")

### Slurm/array helpers

def build_requested_months(years_list: List[int]) -> List[Tuple[int, str]]:
    months: List[Tuple[int, str]] = []
    for year in years_list:
        for month in range(1, 13):
            months.append((year, f"{month:02d}"))
    return months

def month_from_filename(file_path: str | Path) -> Tuple[int, str]:
    m = re.search(r"(\d{4})-(\d{2})", Path(file_path).name)
    if not m:
        raise ValueError(f"Could not parse YYYY-MM from filename: {Path(file_path).name}")
    return int(m.group(1)), m.group(2)

def select_target_files(
    all_files: List[Path],
    years_list: List[int],
    array_idx: Optional[int],
    files_per_job: int,
) -> List[Path]:
    requested_months = build_requested_months(years_list)

    if array_idx is None:
        target_months = requested_months
    else:
        start_idx = array_idx * files_per_job
        end_idx = start_idx + files_per_job
        target_months = requested_months[start_idx:end_idx]

    target_month_set = set(target_months)
    if not target_month_set:
        return []

    out: List[Path] = []
    for p in all_files:
        ym = month_from_filename(p)
        if ym in target_month_set:
            out.append(Path(p))

    return sorted(out, key=lambda p: p.name)

### SQLite cache helpers

def open_cache(db_path: str | Path) -> sqlite3.Connection:
    db_path = str(db_path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(
        db_path,
        timeout=120,
        isolation_level=None,  # explicit transactions
    )
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA busy_timeout=120000;")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS author_map (
            author TEXT PRIMARY KEY,
            anon_id TEXT NOT NULL UNIQUE,
            created_at INTEGER NOT NULL
        );
        """
    )
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_author_map_anon_id ON author_map(anon_id);")
    conn.commit()
    return conn

# Generate a random numeric string with a fixed width.
def generate_candidate_id(num_digits: int = 12) -> str:
    if num_digits < 6:
        raise ValueError("num_digits should be at least 6")
    lower = 10 ** (num_digits - 1)
    upper = (10 ** num_digits) - 1
    return str(secrets.randbelow(upper - lower + 1) + lower)

# Batch-fetch existing IDs for a group of authors.
def get_cached_ids(conn: sqlite3.Connection, authors: List[str]) -> Dict[str, str]:

    if not authors:
        return {}

    out: Dict[str, str] = {}
    cur = conn.cursor()

    # SQLite variable limit safety
    for i in range(0, len(authors), 900):
        chunk = authors[i:i + 900]
        qmarks = ",".join(["?"] * len(chunk))
        cur.execute(
            f"SELECT author, anon_id FROM author_map WHERE author IN ({qmarks})",
            chunk
        )
        for author, anon_id in cur.fetchall():
            out[author] = anon_id

    return out

# Concurrency-safe get-or-create: returns existing ID if present, otherwise assigns exactly one new unique random numeric ID
def get_or_create_author_id(conn: sqlite3.Connection, author: str, num_digits: int = 12) -> str:
    if not author:
        return author

    cur = conn.cursor()

    # Fast path: read first without taking a write lock
    cur.execute("SELECT anon_id FROM author_map WHERE author = ?", (author,))
    row = cur.fetchone()
    if row is not None:
        return row[0]

    while True:
        try:
            # Serialize writers so two processes cannot assign two IDs to the same author
            cur.execute("BEGIN IMMEDIATE;")

            # Check again after acquiring the write lock
            cur.execute("SELECT anon_id FROM author_map WHERE author = ?", (author,))
            row = cur.fetchone()
            if row is not None:
                conn.commit()
                return row[0]

            candidate = generate_candidate_id(num_digits=num_digits)
            now = int(time.time())

            cur.execute(
                "INSERT INTO author_map(author, anon_id, created_at) VALUES (?, ?, ?)",
                (author, candidate, now),
            )
            conn.commit()
            return candidate

        except sqlite3.IntegrityError:
            # Either anon_id collided or another process inserted first.
            conn.rollback()

            cur.execute("SELECT anon_id FROM author_map WHERE author = ?", (author,))
            row = cur.fetchone()
            if row is not None:
                return row[0]

            # Otherwise anon_id collision; retry with a fresh random number.
            continue

        except sqlite3.OperationalError as e:
            conn.rollback()
            msg = str(e).lower()
            if "locked" in msg or "busy" in msg:
                time.sleep(0.1)
                continue
            raise

### Anonymization helpers

# Preserve common non-user Reddit placeholders as-is.
def should_preserve_author(author_value: str) -> bool:
    if author_value is None:
        return True

    author_clean = str(author_value).strip()
    return author_clean in {"", "[deleted]", "[removed]"}

### Main Anonymization Functions

# Stream one CSV to another while anonymizing the 'author' column. Returns (rows_written, new_ids_created_in_this_file).
# Resumes from the last source_row already written to the output if any.
def anonymize_one_file(
    input_file: str | Path,
    output_file: str | Path,
    db_path: str | Path,
) -> Tuple[int, int]:
    input_file = Path(input_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    last_processed = get_last_source_row(output_file)
    mode = "a" if last_processed >= 0 else "w"

    conn = open_cache(db_path)
    local_cache: Dict[str, str] = {}
    rows_written = 0
    new_ids_created = 0

    try:
        with (
            open(input_file, "r", encoding="utf-8-sig", errors="ignore", newline="") as in_f,
            open(output_file, mode, encoding="utf-8", newline="") as out_f,
        ):
            reader = csv.DictReader((line.replace("\x00", "") for line in in_f))
            if reader.fieldnames is None:
                raise ValueError(f"Could not read CSV header from {input_file.name}")

            if "author" not in reader.fieldnames:
                raise ValueError(f"Input file {input_file.name} does not contain an 'author' column.")

            has_source_row = "source_row" in reader.fieldnames

            writer = csv.DictWriter(out_f, fieldnames=reader.fieldnames)
            if mode == "w":
                writer.writeheader()

            for row in reader:
                # Resume: skip rows already anonymized in a prior run.
                if has_source_row and last_processed >= 0:
                    sval = (row.get("source_row") or "").strip()
                    try:
                        if int(sval) <= last_processed:
                            continue
                    except ValueError:
                        pass

                author = row.get("author", "")

                if not should_preserve_author(author):
                    author = str(author).strip()

                    anon_id = local_cache.get(author)
                    if anon_id is None:
                        # Check DB in case it already exists from earlier runs / other tasks
                        cached = get_cached_ids(conn, [author])
                        anon_id = cached.get(author)

                        if anon_id is None:
                            before = get_cached_ids(conn, [author]).get(author)
                            anon_id = get_or_create_author_id(conn, author)
                            after = anon_id
                            if before is None and after is not None:
                                new_ids_created += 1

                        local_cache[author] = anon_id

                    row["author"] = anon_id

                writer.writerow(row)
                rows_written += 1

    finally:
        conn.close()

    return rows_written, new_ids_created

### Main execution

def organize_anonymize() -> None:
    start_time = time.time()

    files_per_job = getattr(args, "files_per_job", 1) or 1
    target_files = select_target_files(
        all_files=file_list,
        years_list=years,
        array_idx=args.array,
        files_per_job=files_per_job,
    )

    if not target_files:
        log_report(report_file_path, "No target files assigned to this run.")
        return

    log_report(
        report_file_path,
        f"Preparing to anonymize {len(target_files)} file(s) for group={group}, type={type_}."
    )

    processed = 0
    skipped = 0
    failed = 0
    total_rows = 0
    total_new_ids = 0

    for input_file in target_files:
        output_file = output_path / input_file.name

        try:
            rows_written, new_ids_created = anonymize_one_file(
                input_file=input_file,
                output_file=output_file,
                db_path=USER_CACHE
            )
            processed += 1
            total_rows += rows_written
            total_new_ids += new_ids_created

            log_report(
                report_file_path,
                f"Anonymized {input_file.name} -> {output_file.name}; rows={rows_written}, new_ids={new_ids_created}"
            )

        except Exception as e:
            failed += 1
            log_report(
                report_file_path,
                f"Error anonymizing {input_file.name}: {e}"
            )

    elapsed_time = (time.time() - start_time) / 60.0
    log_report(
        report_file_path,
        f"Finished anonymization. Successful: {processed}, Skipped: {skipped}, Failed: {failed}, "
        f"Rows written: {total_rows}, New IDs assigned: {total_new_ids}, Time: {elapsed_time:.2f} minutes"
    )

if __name__ == "__main__":
    overall_start_time = time.time()
    try:
        organize_anonymize()
    except Exception as e:
        reraise_fatal(report_file_path, "organize_anonymize", e)

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
        f"Anonymization for {group} / {type_} for {scope_msg} finished in {total_time:.2f} minutes"
    )