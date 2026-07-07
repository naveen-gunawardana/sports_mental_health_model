### Imports

# Import functions and objects
from cli import get_args, MODELS_DIR, DATA_DIR
from utils import parse_range, headers, check_reqd_files, log_report, log_error, get_last_source_row, detect_source_row, reraise_fatal

# fasttext-wheel 0.9.2 calls np.array(probs, copy=False) inside its predict()
# path, which raises ValueError under numpy 2.x. Patch np.array to fall back
# to np.asarray when copy=False is requested. Must run BEFORE `import fasttext`
# so the patched function is in place when fasttext binds it later.
import numpy as _np
_orig_np_array = _np.array
def _np_array_copy_false_compat(obj, dtype=None, *, copy=True, **kw):
    if copy is False:
        return _np.asarray(obj, dtype=dtype, **kw)
    return _orig_np_array(obj, dtype=dtype, copy=copy, **kw)
_np.array = _np_array_copy_false_compat

# Import Python packages
import fasttext
import os
import csv
csv.field_size_limit(2**31 - 1) # Increase the field size limit to handle larger fields
import time
import re
from datetime import datetime  # For timestamping
import traceback
from pathlib import Path

### Argument Handling

# Extract and transform CLI arguments 
args = get_args()
type_ = args.type
group = args.group
years = parse_range(args.years)

### Path Handling

# Load the fastText language identification model
model_path = os.path.join(MODELS_DIR,"filter_language.bin")
model = fasttext.load_model(str(model_path))

# Define a function that applies the fastText model to a given text
def detect_language(text):
    predictions = model.predict(text)
    # The language code is returned with a prefix "__label__", which we remove
    return predictions[0][0].replace('__label__', '')

# Survey the input files and raise an error if an expected file within the requested range is missing
if not args.input:
    input_path = os.path.join(DATA_DIR, "data_reddit_curated", group, type_, "filtered_keywords"
    )
else:
    input_path = args.input
file_list = check_reqd_files(years=years, type_=type_, check_path=input_path)
file_list = sorted(file_list, key=lambda p: Path(p).name)

# Allow each Slurm array task to process only its assigned slice of files
array_index = getattr(args, "array", None)
files_per_job = getattr(args, "files_per_job", 1)

if array_index is not None:
    total_files = len(file_list)
    start = array_index * files_per_job
    end = min(start + files_per_job, total_files)

    if start >= total_files:
        print(
            f"No files to process for array index {array_index} "
            f"(start={start}, total_files={total_files}). Exiting."
        )
        exit(0)

    file_list = file_list[start:end]
    print(
        f"Array index {array_index}: processing files "
        f"{start}..{end-1} (of {total_files})"
    )

# Prepare and survey the output path
if args.output:
    output_path = args.output
else:
    output_path = os.path.join(DATA_DIR,
        "data_reddit_curated", group, type_, "filtered_language"
    )
os.makedirs(output_path, exist_ok=True)

# The report file is used to log messages (tab-separated: timestamp and message)
report_file_path = os.path.join(output_path, "Report_filter_language.csv")

### Main functions

# Function for language filtering a single file
def filter_language_file(file):

    function_name = "filter_language_file"
    log_report(report_file_path, f"Started language filtering for the {group} content in {Path(file).name}.")

    try:
        output_file_path = os.path.join(output_path, Path(file).name)

        # Resume: pick up from the last source_row already written to the output.
        # If there's no resume point (no output yet, or it lacks source_row),
        # last_processed = -1 and we open in 'w' mode for a fresh start.
        last_processed = get_last_source_row(
            output_file_path,
            report_file_path=report_file_path,
            file_for_log=file,
        )
        mode = "a" if last_processed >= 0 else "w"

        with open(file, "r", encoding='utf-8-sig', errors='ignore') as input_file, \
             open(output_file_path, mode, encoding='utf-8', errors='ignore', newline='') as output_file:
            start_time = time.time()

            error_counter = 0
            filtered_counter = 0
            passed_counter = 0

            # Replace NUL characters with empty strings before passing to csv.reader
            reader = csv.reader((line.replace('\0', '') for line in input_file))
            writer = csv.writer(output_file)

            try:
                in_header = next(reader)
            except StopIteration:
                return 0, 0, 0

            src_idx_in, has_input_source_row = detect_source_row(in_header)

            # Output header: pass through input header (which already has
            # source_row when produced by an updated upstream resource).
            # If input lacks source_row, generate it from the input row index
            # so downstream resources can still resume.
            if mode == "w":
                if has_input_source_row:
                    writer.writerow(in_header)
                else:
                    writer.writerow(list(in_header) + ["source_row"])

            for id_, line in enumerate(reader, start=1):
                try:
                    # Determine this row's source_row (for both the resume check
                    # and the value we'll write).
                    if has_input_source_row:
                        src_val = line[src_idx_in].strip() if len(line) > src_idx_in else ""
                        try:
                            src_num = int(src_val)
                        except ValueError:
                            src_num = None
                    else:
                        src_num = id_

                    # Resume: skip rows we've already written.
                    if src_num is not None and src_num <= last_processed:
                        continue

                    line[2] = line[2].strip().replace("\n", " ")
                    if detect_language(line[2]) == 'en':
                        if has_input_source_row:
                            writer.writerow(line)
                        else:
                            writer.writerow(list(line) + [id_])
                        passed_counter += 1
                except IndexError as e:
                    log_error(
                        function_name,
                        file,
                        id_ + 1,
                        str(line),
                        e,
                        report_file_path=report_file_path,
                        output_path=output_path,
                    )
                    error_counter += 1
                    continue
                filtered_counter += 1

            elapsed_minutes = (time.time() - start_time) / 60
            log_report(
                report_file_path,
                f"Finished filtering {Path(file).name} for relevance to the {group} social group based on language in {elapsed_minutes:.2f} minutes. "
                f"# of evaluations: {filtered_counter}, # of English posts: {passed_counter}, # of errors: {error_counter}"
            )
            # Return counters for overall statistics
            return filtered_counter, passed_counter, error_counter
    except Exception as e:
        # Fatal error mid-processing leaves a TRUNCATED output file; never
        # swallow it (that would exit 0 / COMPLETED with partial data). Log an
        # actionable message (OS I/O errors get errno detail) and re-raise so
        # the task fails and any --dependency=afterok gate trips.
        reraise_fatal(report_file_path, Path(file).name, e)

### Main execution

if __name__ == "__main__":
    target_files = list(file_list)

    overall_start_time = time.time()
    total_filtered = 0
    total_passed = 0
    total_errors = 0

    # Process each file and accumulate statistics
    for file in file_list:
        counters = filter_language_file(file)
        if counters:
            filtered_counter, passed_counter, error_counter = counters
            total_filtered += filtered_counter
            total_passed += passed_counter
            total_errors += error_counter

    overall_elapsed = (time.time() - overall_start_time) / 60

    if array_index is None:
        scope_msg = f"{args.years}"
    else:
        scope_msg = f"{args.years} (array index {array_index})"

    log_report(
        report_file_path,
        f"Language filtering for the {args.group} social group for {scope_msg} finished in {overall_elapsed:.2f} minutes"
    )

    # Missing-month checks: evaluate output completeness only after processing
    processed_months = {}
    for file in os.listdir(output_path):
        if file.endswith(".csv") and file not in [os.path.basename(report_file_path), "Final_report_filter_language.csv"]:
            m = re.search(r"(\d{4})-(\d{2})", file)
            if m:
                year, month = m.groups()
                processed_months.setdefault(year, set()).add(month)

    if array_index is None:
        # Full-run completeness check
        for year in years:
            year_str = str(year)
            expected_months = set(f"{m:02d}" for m in range(1, 13))
            actual_months = processed_months.get(year_str, set())
            missing = expected_months - actual_months
            if missing:
                log_report(
                    report_file_path,
                    f"Warning: For year {year_str}, missing output files for months: {sorted(list(missing))}"
                )

        # Final summary report only for full runs
        final_report = [
            ["Timestamp", "Social Group", "Years", "Total Evaluations", "Total Relevant Posts", "Total Missing Lines", "Elapsed Time (minutes)"],
            [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), args.group, args.years, total_filtered, total_passed, total_errors, f"{overall_elapsed:.2f}"]
        ]
        final_report_file = os.path.join(output_path, "Final_report_filter_language.csv")
        with open(final_report_file, "w", encoding="utf-8", newline="") as rf:
            writer = csv.writer(rf)
            writer.writerows(final_report)
        log_report(report_file_path, f"Final report saved to: {final_report_file}")

    else:
        # Array-task completeness check: only for months assigned to this task
        expected_by_year = {}
        for file in target_files:
            m = re.search(r"(\d{4})-(\d{2})", Path(file).name)
            if m:
                year, month = m.groups()
                expected_by_year.setdefault(year, set()).add(month)

        for year_str, expected_months in expected_by_year.items():
            actual_months = processed_months.get(year_str, set())
            missing = expected_months - actual_months
            if missing:
                log_report(
                    report_file_path,
                    f"Warning: For array index {array_index}, year {year_str}, missing output files for months: {sorted(list(missing))}"
                )