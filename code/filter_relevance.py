### Imports

# import functions and objects
from cli import get_args, DATA_DIR, MODELS_DIR
from utils import parse_range, headers, log_report, check_reqd_files, log_error, get_last_source_row, detect_source_row

# import Python packages
import os
import csv
csv.field_size_limit(2**31 - 1) # Increase the field size limit to handle larger fields
import time
import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
import datetime
import re
from pathlib import Path
import traceback
import sys
import io
import threading
from queue import Queue

### Argument Handling

# Extract and transform CLI arguments 
args = get_args()
years = parse_range(args.years)
group = args.group
type_ = args.type
batch_size = args.batchsize

# Set relevance filtering hyperparameters
max_length = 512
if group == "skin_tone" or group == "race":
    thresholding = True # If True, the model will use a confidence threshold (set below) to determine the class of a document. If False, it will always return the most probable class.
else:
    thresholding = False # thresholding is only applied to the filter_relevance model for skin_tone
threshold_class = 1 # the class that needs a probability passing the threshold (set below) to be picked as the answer. Only matters if thresholding = True.
threshold = 0.6 # The confidence threshold for the rarest class. If the model's confidence in a class is below this value, it will not return that class. Only matters if thresholding=True and the value is greater than >.50 given the two main labels. 

### Path Handling

# set path variables

# Survey the input files
if args.input:
    input_path = os.path.abspath(args.input)
else:
    input_path = os.path.join(
        DATA_DIR, "data_reddit_curated", group, type_, "filtered_language"
    )

file_list = check_reqd_files(years, input_path, type_)

# Prepare and survey the output path
if args.output:
    output_path = os.path.abspath(args.output)
else:
    output_path = os.path.join(
        DATA_DIR, "data_reddit_curated", group, type_, "filtered_relevance"
    )

os.makedirs(output_path, exist_ok=True)
# prepare the report file
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use CUDA if available
report_file_path = os.path.join(output_path, "Report_filter_relevance.csv")
log_report(report_file_path, f"Using device: {device}")

# the following portion allows each slurm process to process multiple files if files_per_job is set in the command line arguments.
array_index   = getattr(args, "array", None)
files_per_job = getattr(args, "files_per_job", 1)

if array_index is not None:
    total_files = len(file_list)
    start = array_index * files_per_job
    end   = min(start + files_per_job, total_files)

    if start >= total_files:
        msg = (
            f"No files to process for array index {array_index} "
            f"(start={start}, total_files={total_files}). Exiting."
        )
        log_report(report_file_path, msg)
        sys.exit(0)

    file_list = file_list[start:end]
    log_report(
        report_file_path,
        f"Array index {array_index}: processing files {start}..{end-1} (of {total_files})"
    )

# Load relevance model
model_path = os.path.join(MODELS_DIR,
                          f"filter_relevance_{group}")
tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path).to(device)
if torch.cuda.device_count() > 1: # if more than one GPU is available
    model = torch.nn.DataParallel(model) # parallelize
model.eval() # set model to evaluation mode

# Log GPU memory usage
def log_gpu_memory():
    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(device=device)
        used_bytes = total_bytes - free_bytes
        log_report(
            report_file_path,
            f"GPU memory: {used_bytes / (1024 ** 3):.2f} GiB / {total_bytes / (1024 ** 3):.2f} GiB used"
        )

### Main functions

# CPU-side tokenization (runs on the producer thread so it can overlap with
# GPU inference happening on the main thread).
def tokenize_texts(texts):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    if device.type == "cuda":
        enc = {k: v.pin_memory() for k, v in enc.items()}
    else:
        enc = {k: v for k, v in enc.items()}
    return enc

# GPU-side inference on an already-tokenized batch.
@torch.no_grad()
def predict_tokenized(tokenized, threshold_class=threshold_class, threshold=threshold, thresholding=thresholding):
    inputs = {k: v.to(device, non_blocking=True) for k, v in tokenized.items()}
    with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
        outputs = model(**inputs)
        probs = outputs[0].softmax(1)

    predictions = []
    for prob in probs:
        if thresholding and prob[threshold_class] > threshold:
            predictions.append(threshold_class)
        else:
            if thresholding:
                masked_probs = prob.clone()
                masked_probs[threshold_class] = -1
                predictions.append(masked_probs.argmax().item())
            else:
                predictions.append(prob.argmax().item())

    return predictions

# Propagate `source_row` from input when present; generate it from the input
# row index when absent (legacy filter_language outputs).
def filter_relevance_file(file):
    function_name = "filter_relevance_file"
    log_report(
        report_file_path,
        f"Started neural network filtering {Path(file).name} for relevance to the {group} social group."
    )

    try:
        start_time = time.time()
        output_file_path = os.path.join(output_path, Path(file).name)

        error_counter = 0
        evaluated_counter = 0
        passed_counter = 0

        # Determine resume position if output file already exists.
        last_processed = get_last_source_row(
            output_file_path,
            report_file_path=report_file_path,
            file_for_log=file,
        )
        mode = "a" if last_processed >= 0 else "w"

        with open(file, "r", encoding="utf-8-sig", errors="ignore") as input_file, \
             open(output_file_path, mode, encoding="utf-8-sig", errors="ignore", newline="") as output_file:

            reader = csv.reader((line.replace('\x00', '') for line in input_file))
            writer = csv.writer(output_file)

            try:
                in_header = next(reader)
            except StopIteration:
                return 0, 0, 0

            src_idx_in, has_input_source_row = detect_source_row(in_header)
            keywords_idx = in_header.index("matched patterns") if "matched patterns" in in_header else headers.index("matched patterns")

            # Output header: propagate input header verbatim if it already has
            # source_row; otherwise append source_row so downstream resources
            # can resume.
            if mode == "w":
                if has_input_source_row:
                    writer.writerow(in_header)
                else:
                    writer.writerow(list(in_header) + ["source_row"])

            # Producer/consumer with length-bucketing. The producer accumulates
            # a bucket of BUCKET_MULTIPLIER * batch_size rows, sorts them by
            # estimated token length, and greedily packs them into sub-batches
            # respecting a fixed token-slot budget (batch_size * max_length).
            # Effect: short docs batch together with little padding; long docs
            # are processed in smaller sub-batches so peak GPU memory stays
            # bounded by the original worst case rather than growing past it.
            # The consumer collects outputs per bucket and re-sorts them by
            # input row order before writing, so resume safety is preserved.
            BUCKET_MULTIPLIER = 8
            bucket_target = batch_size * BUCKET_MULTIPLIER
            token_budget = batch_size * max_length

            batch_queue: "Queue" = Queue(maxsize=2)
            SENTINEL = object()
            producer_state = {"evaluated_counter": 0, "error_counter": 0}

            def flush_bucket(buf):
                if not buf:
                    return
                buf.sort(key=lambda t: t[2])
                sub_batches = []
                current = []
                current_max = 0
                for triple in buf:
                    length = triple[2]
                    new_max = max(current_max, length)
                    if current and (len(current) + 1) * new_max > token_budget:
                        sub_batches.append(current)
                        current = [triple]
                        current_max = length
                    else:
                        current.append(triple)
                        current_max = new_max
                if current:
                    sub_batches.append(current)

                for i, sb in enumerate(sub_batches):
                    is_last = (i == len(sub_batches) - 1)
                    try:
                        lines = [t[1] for t in sb]
                        ids = [t[0] for t in sb]
                        texts = [l[2].strip().replace("\n", " ") for l in lines]
                        tokenized = tokenize_texts(texts)
                        batch_queue.put((ids, lines, tokenized, is_last))
                    except Exception as e:
                        log_error(
                            function_name,
                            file,
                            int(sb[0][0]) if sb else -1,
                            f"tokenize sub-batch ({len(sb)} rows)",
                            e,
                            report_file_path=report_file_path,
                            output_path=output_path,
                        )
                        producer_state["error_counter"] += len(sb)
                        # Still need to mark the bucket boundary so the consumer flushes.
                        if is_last:
                            batch_queue.put(([], [], None, True))

            def producer():
                pending_pairs = []
                try:
                    for id_, line in enumerate(reader, start=1):
                        try:
                            if len(line) < 3:
                                raise IndexError(f"insufficient columns ({len(line)} found)")

                            if has_input_source_row:
                                src_val = line[src_idx_in].strip() if len(line) > src_idx_in else ""
                                try:
                                    src_num = int(src_val)
                                except ValueError:
                                    src_num = None
                            else:
                                src_num = id_

                            if src_num is not None and src_num <= last_processed:
                                continue

                            if has_input_source_row:
                                line_to_add = line
                            else:
                                line_to_add = list(line) + [id_]

                            est_len = min(max_length, max(1, len(line_to_add[2]) // 4 + 1))
                            pending_pairs.append((id_, line_to_add, est_len))
                            producer_state["evaluated_counter"] += 1

                            if len(pending_pairs) >= bucket_target:
                                flush_bucket(pending_pairs)
                                pending_pairs = []

                        except Exception as e:
                            log_error(
                                function_name,
                                file,
                                id_ + 1,
                                str(line),
                                e,
                                report_file_path=report_file_path,
                                output_path=output_path,
                            )
                            producer_state["error_counter"] += 1
                            continue

                    if pending_pairs:
                        flush_bucket(pending_pairs)
                        pending_pairs = []
                finally:
                    batch_queue.put(SENTINEL)

            producer_thread = threading.Thread(target=producer, daemon=True)
            producer_thread.start()

            pending_writes = []  # (input_id, output_row) within current bucket
            while True:
                item = batch_queue.get()
                if item is SENTINEL:
                    break
                ids, batch_lines, tokenized, is_last = item

                predictions = None
                if tokenized is not None:
                    # Inference failures (CUDA OOM, model errors, etc.) are
                    # infrastructure issues, not per-row data problems. Swallowing
                    # them produces silently-incomplete output with the Slurm task
                    # still exiting 0 -- the exact failure mode that triggered this
                    # hardening. Log one line to the report CSV for forensics, then
                    # re-raise so the task exits non-zero and we know to fix and
                    # resubmit (likely with a smaller --batchsize or a larger GPU).
                    try:
                        predictions = predict_tokenized(tokenized)
                    except Exception as e:
                        log_report(
                            report_file_path,
                            f"[fatal] inference failed on sub-batch "
                            f"({len(batch_lines)} rows) in {os.path.basename(file)} "
                            f"at first input id={int(ids[0]) if ids else -1}: "
                            f"{type(e).__name__}: {e}"
                        )
                        raise

                if predictions is not None:
                    for idx, pred in enumerate(predictions):
                        if pred == 1:
                            row = batch_lines[idx]
                            row[keywords_idx] = ",".join(set(row[keywords_idx].split(",")))
                            pending_writes.append((ids[idx], row))
                            passed_counter += 1

                if is_last and pending_writes:
                    pending_writes.sort(key=lambda x: x[0])
                    writer.writerows([row for _, row in pending_writes])
                    pending_writes.clear()

            producer_thread.join()
            if pending_writes:
                pending_writes.sort(key=lambda x: x[0])
                writer.writerows([row for _, row in pending_writes])
                pending_writes.clear()

            evaluated_counter = producer_state["evaluated_counter"]
            error_counter = producer_state["error_counter"]

        elapsed_minutes = (time.time() - start_time) / 60
        log_report(
            report_file_path,
            f"Finished neural network filtering {Path(file).name} for relevance to the {group} social group in {elapsed_minutes:.2f} minutes. "
            f"# of evaluations: {evaluated_counter}, # of relevant posts: {passed_counter}, # of errors: {error_counter}"
        )
        log_gpu_memory()

        return evaluated_counter, passed_counter, error_counter

    except Exception:
        tb_str = traceback.format_exc()
        log_report(
            report_file_path,
            f"Fatal error during relevance filtering for {Path(file).name}:\n{tb_str}"
        )
        # Per-row data errors are caught deeper in the producer/flush_bucket,
        # so what reaches this handler is infrastructure (CUDA OOM, model
        # crashes, fs errors). Re-raise so the Slurm task exits non-zero
        # instead of silently exiting 0 with truncated or missing output.
        raise

### Main Execution

if __name__ == "__main__":
    target_files = list(file_list)

    overall_start_time = time.time()
    total_evaluated = 0
    total_passed = 0
    total_errors = 0

    for file in file_list:
        counters = filter_relevance_file(file)
        if counters:
            evaluated_counter, passed_counter, error_counter = counters
            total_evaluated += evaluated_counter
            total_passed += passed_counter
            total_errors += error_counter

    overall_elapsed = (time.time() - overall_start_time) / 60

    if array_index is None:
        scope_msg = f"{args.years}"
    else:
        scope_msg = f"{args.years} (array index {array_index})"

    log_report(
        report_file_path,
        f"Relevance filtering for the {group} social group for {scope_msg} finished in {overall_elapsed:.2f} minutes"
    )

    # Check for missing outputs
    processed_months = {}
    for file in os.listdir(output_path):
        if file.endswith(".csv") and file not in [
            os.path.basename(report_file_path),
            "Final_Report_FilterRelevance.csv",
        ]:
            m = re.search(r"(\d{4})-(\d{2})", file)
            if m:
                year, month = m.groups()
                processed_months.setdefault(year, set()).add(month)

    if array_index is None: # on local runs
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

        final_report = [
            ["Timestamp", "Social Group", "Years", "Total Evaluations", "Total Relevant Posts", "Total Errors", "Elapsed Time (minutes)"],
            [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), group, args.years, total_evaluated, total_passed, total_errors, f"{overall_elapsed:.2f}"]
        ]
        final_report_file = os.path.join(output_path, "Final_Report_FilterRelevance.csv")
        with open(final_report_file, "w", encoding="utf-8", newline="") as rf:
            writer = csv.writer(rf)
            writer.writerows(final_report)
        log_report(report_file_path, f"Final report saved to: {final_report_file}")

    else: # if running on a cluster
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
