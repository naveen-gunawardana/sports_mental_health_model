### Imports

# import functions and objects
from cli import get_args, MODELS_DIR, DATA_DIR
from utils import parse_range, log_report, check_reqd_files, get_last_source_row

# import Python packages
import os, sys
import csv
csv.field_size_limit(2**31 - 1) # Increase the field size limit to handle larger fields
import time
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
import datetime
import re
from pathlib import Path
import threading
from queue import Queue

### Argument Handling

# Extract and transform CLI arguments
args = get_args()
years = parse_range(args.years)
group = args.group
type_ = args.type
batch_size = args.batchsize
files_per_job = getattr(args, "files_per_job", 1)
if files_per_job is None or files_per_job < 1:
    files_per_job = 1
if args.array is not None:
    array = args.array

### Path Handling

# set path variables

# input/output path processing
model_path = MODELS_DIR / "label_moralization"
if not args.input:
    input_path = DATA_DIR / "data_reddit_curated" / group / type_ / "filtered_keywords_adv"
else:
    input_path = Path(args.input)

# Build file_list organized by year and raise an error if an expected file is missing
file_list = check_reqd_files(years, input_path, type_)
file_list = sorted(file_list, key=lambda p: Path(p).name)

if not args.output:
    output_path = DATA_DIR / "data_reddit_curated" / group / type_ / "labeled_moralization"
else:
    output_path = Path(args.output)
output_path.mkdir(parents=True, exist_ok=True)

# prepare the report file
report_file_path = os.path.join(output_path, f"report_label_moralization.csv")

### Model Preparation

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_report(report_file_path, f"Using device: {device}")

# Load moralization model
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(
    model_path,
    device_map=None,   # or "auto" if you want
    use_safetensors=True
).to(device)
if torch.cuda.device_count() > 1:  # if more than one GPU is available
    model = torch.nn.DataParallel(model)  # parallelize
model.eval()  # set model to evaluation mode

# Log GPU memory usage
def log_gpu_memory():
    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(device=device)
        used_bytes = total_bytes - free_bytes
        log_report(
            report_file_path,
            f"GPU memory: {used_bytes / (1024 ** 3):.2f} GiB / {total_bytes / (1024 ** 3):.2f} GiB used"
        )

### Main Functions

# CPU-side tokenization (runs on the producer thread).
def tokenize_texts(texts, max_length=512):
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
def predict_tokenized(tokenized):
    inputs = {k: v.to(device, non_blocking=True) for k, v in tokenized.items()}
    with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        predictions = probs.argmax(dim=1).tolist()
    return predictions

def label_moralization_file(file):

    missing_lines_count = 0
    log_report(report_file_path, f"Started labeling {Path(file).name} from the {group} social group for moralization.")
    start_time = time.time()

    # Build output file path using the relative part from the input file.
    output_file_path = os.path.join(output_path, Path(file).name)

    # Resume from the last source_row in the existing output (if any).
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

        # Read input header and locate source_row column by name
        try:
            in_header = next(reader)
        except StopIteration:
            return 0  # empty input

        try:
            src_idx_in = in_header.index("source_row")
        except ValueError:
            raise RuntimeError("Input file is missing required 'source_row' column.")

        # If starting a new output, write header = input header + Moralization
        if mode == "w":
            new_headers = in_header + ["Moralization"]
            writer.writerow(new_headers)

        # Producer/consumer with length-bucketing. See filter_relevance.py for
        # design notes. Producer accumulates BUCKET_MULTIPLIER * batch_size
        # rows, sorts by est. token length, and packs into sub-batches under a
        # fixed token-slot budget. Consumer sorts outputs back to input-row
        # order per bucket before writing, preserving resume safety.
        BUCKET_MULTIPLIER = 8
        bucket_target = batch_size * BUCKET_MULTIPLIER
        MAX_LENGTH = 512
        token_budget = batch_size * MAX_LENGTH

        batch_queue: "Queue" = Queue(maxsize=2)
        SENTINEL = object()
        producer_state = {"total_lines": 0, "missing_lines_count": 0}

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
                lines = [t[1] for t in sb]
                ids = [t[0] for t in sb]
                texts = [l[2].strip().replace("\n", " ") for l in lines]
                tokenized = tokenize_texts(texts)
                batch_queue.put((ids, lines, tokenized, is_last))

        def producer():
            pending_pairs = []
            try:
                for id_, line in enumerate(reader, start=1):
                    if len(line) < 3:
                        producer_state["missing_lines_count"] += 1
                        continue

                    src_val = line[src_idx_in].strip()
                    src_num = int(src_val) if src_val.isdigit() else None
                    if src_num is not None and src_num <= last_processed:
                        continue

                    est_len = min(MAX_LENGTH, max(1, len(line[2]) // 4 + 1))
                    pending_pairs.append((id_, line, est_len))
                    producer_state["total_lines"] += 1

                    if len(pending_pairs) >= bucket_target:
                        flush_bucket(pending_pairs)
                        pending_pairs = []

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
            predictions = predict_tokenized(tokenized)
            for idx, pred in enumerate(predictions):
                row_out = batch_lines[idx] + ["Moralized" if pred else "Non-Moralized"]
                pending_writes.append((ids[idx], row_out))

            if is_last and pending_writes:
                pending_writes.sort(key=lambda x: x[0])
                writer.writerows([row for _, row in pending_writes])
                pending_writes.clear()

        producer_thread.join()
        if pending_writes:
            pending_writes.sort(key=lambda x: x[0])
            writer.writerows([row for _, row in pending_writes])
            pending_writes.clear()

        total_lines = producer_state["total_lines"]
        missing_lines_count = producer_state["missing_lines_count"]

    # generate processing report
    elapsed_minutes = (time.time() - start_time) / 60
    log_report(
        report_file_path,
        f"Finished moralization labeling {Path(file).name} for the {group} social group in {elapsed_minutes:.2f} minutes. "
        f"Processed rows: {total_lines}"
    )
    log_gpu_memory()

    if missing_lines_count:
        missing_records_file = os.path.join(output_path, 'missing_records.csv')
        need_header = not os.path.exists(missing_records_file)
        with open(missing_records_file, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if need_header:
                w.writerow(['Filename', 'MissingLinesCount', 'Timestamp'])
            w.writerow([str(file), missing_lines_count, datetime.datetime.now().isoformat(timespec="seconds")])

    return total_lines

### Main Execution

# process each file and aggregate stats
if __name__ == "__main__":

    start_time = time.time()

    overall_docs = 0

    # Process each file from the file_list (global mode)
    if args.array is not None:  # for batch processing (Slurm array task)
        # args.array is the *job index*; each job processes a chunk of monthly files.
        start = array * files_per_job
        end = min(start + files_per_job, len(file_list))
        if start >= len(file_list):
            raise RuntimeError(
                f"Array index {array} is out of range for {len(file_list)} files (files_per_job={files_per_job})."
            )
        log_report(report_file_path, f"Array task {array}: processing file_list[{start}:{end}] (files_per_job={files_per_job}).")
        for file in file_list[start:end]:
            overall_docs += label_moralization_file(file)

    else:  # for sequential processing
        for file in file_list:
            overall_docs += label_moralization_file(file)

        # Check for missing monthly outputs
        for year in years:
            expected_months = set(f"{m:02d}" for m in range(1, 13))
            processed_months = set()
            for file in os.listdir(output_path):
                m = re.search(r'RC_' + str(year) + r'-(\d{2})\.csv', file)
                if m:
                    processed_months.add(m.group(1))
            missing = expected_months - processed_months
            if missing:
                log_report(report_file_path, f"Warning: For year {year}, missing output files for months: {sorted(list(missing))}")

        # time logging
        overall_elapsed = (time.time() - start_time) / 60
        log_report(
            report_file_path,
            f"Moralization labeling for the {group} social group for {args.years} finished in {overall_elapsed:.2f} minutes. "
            f"Total processed rows: {overall_docs}"
        )

        # Aggregate overall statistics and save final summary report
        final_report = [
            ["Timestamp", "Social Group", "Years", "Total Processed Rows", "Total Elapsed Time (min)"],
            [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), group, args.years, overall_docs, f"{overall_elapsed:.2f}"]
        ]
        final_report_file = os.path.join(output_path, "final_report_label_moralization.csv")
        with open(final_report_file, "a+", encoding="utf-8", newline="") as rf:
            writer = csv.writer(rf)
            writer.writerows(final_report)
        log_report(report_file_path, f"Final summary report saved to: {final_report_file}")
