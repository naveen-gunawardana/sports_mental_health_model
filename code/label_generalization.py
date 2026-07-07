### Imports

# import functions and objects
from cli import get_args, DATA_DIR, MODELS_DIR
from utils import parse_range, log_report, check_reqd_files, get_last_source_row

# import Python packages
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # to prevent MKL crash with Torch
import csv
csv.field_size_limit(2**31 - 1) # Increase the field size limit to handle larger fields
import time
import torch
from transformers import RobertaTokenizerFast, AutoModelForTokenClassification, RobertaForSequenceClassification
import datetime
import re
import spacy
import numpy as np
from copy import deepcopy
from pathlib import Path
import threading
from queue import Queue

### Argument Handling

# Extract and transform CLI arguments 
args = get_args()
years = parse_range(args.years)
files_per_job = getattr(args, "files_per_job", 1)
if files_per_job is None or files_per_job < 1:
    files_per_job = 1
group = args.group
type_ = args.type
batch_size = args.batchsize
if args.array is not None:
    array = args.array

### Path Handling

# set path variables

model_path = MODELS_DIR / "label_generalization"

if not args.input:
    input_path = DATA_DIR / "data_reddit_curated" / group / type_ / "labeled_sentiment"
else:
    input_path = Path(args.input)

# Build file_list organized by year and raise an error if an expected file is missing
file_list = check_reqd_files(years, input_path, type_)
file_list = sorted(file_list, key=lambda p: Path(p).name)

if not args.output:
    output_path = DATA_DIR / "data_reddit_curated" / group / type_ / "labeled_generalization"
else:
    output_path = Path(args.output)
output_path.mkdir(parents=True, exist_ok=True)

# prepare the report file
report_file_path = os.path.join(output_path, f"report_label_generalization.csv")

### Model Preparation

# Set moralization labeling hyperparameters
max_length=512

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_report(report_file_path,f"Using device: {device}")

# load the necessary models and move them to the device being used
nlp = spacy.load("en_core_web_sm")
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base",add_prefix_space=True)  
clause_model = AutoModelForTokenClassification.from_pretrained("{}/label_generalization_segmentation".format(model_path), num_labels=3).to(device)
generalization_model = RobertaForSequenceClassification.from_pretrained("{}/label_generalization".format(model_path), num_labels=18).to(device)

if torch.cuda.device_count() > 1: # if more than one GPU is available
    clause_model = torch.nn.DataParallel(clause_model) # parallelize
    generalization_model = torch.nn.DataParallel(generalization_model)

clause_model.eval() # set model to evaluation mode
generalization_model.eval() # set model to evaluation mode

# Log GPU memory usage
def log_gpu_memory():
    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(device=device)
        used_bytes = total_bytes - free_bytes
        log_report(
            report_file_path,
            f"GPU memory: {used_bytes / (1024 ** 3):.2f} GiB / {total_bytes / (1024 ** 3):.2f} GiB used"
        )


# define the mapping between clause labels and each of the three composing features
labels2attrs = {
    "##BOUNDED EVENT (SPECIFIC)": ("specific", "dynamic", "episodic"),
    "##BOUNDED EVENT (GENERIC)": ("generic", "dynamic", "episodic"),
    "##UNBOUNDED EVENT (SPECIFIC)": ("specific", "dynamic", "static"),  # This should be (static, or habitual)
    "##UNBOUNDED EVENT (GENERIC)": ("generic", "dynamic", "static"),
    "##BASIC STATE": ("specific", "stative", "static"),
    "##COERCED STATE (SPECIFIC)": ("specific", "dynamic", "static"),
    "##COERCED STATE (GENERIC)": ("generic", "dynamic", "static"),
    "##PERFECT COERCED STATE (SPECIFIC)": ("specific", "dynamic", "episodic"),
    "##PERFECT COERCED STATE (GENERIC)": ("generic", "dynamic", "episodic"),
    "##GENERIC SENTENCE (DYNAMIC)": ("generic", "dynamic", "habitual"),   
    "##GENERIC SENTENCE (STATIC)": ("generic", "stative", "static"),  # The car is red now (static)
    "##GENERIC SENTENCE (HABITUAL)": ("generic", "stative", "habitual"),   # I go to the gym regularly (habitual)
    "##GENERALIZING SENTENCE (DYNAMIC)": ("specific", "dynamic", "habitual"),
    "##GENERALIZING SENTENCE (STATIVE)": ("specific", "stative", "habitual"),
    "##QUESTION": ("NA", "NA", "NA"),
    "##IMPERATIVE": ("NA", "NA", "NA"),
    "##NONSENSE": ("NA", "NA", "NA"),
    "##OTHER": ("NA", "NA", "NA"),
}

# create dictionaries for ease of translating between clause labels and attributes
label2index = {l:i for l,i in zip(labels2attrs.keys(), np.arange(len(labels2attrs)))}
index2label = {i:l for l,i in label2index.items()}

### Helper Function

# Splits longer text inputs at the end of sentences into parts that the neural networks can label without truncation.
def auto_split(text):
    doc = nlp(text)
    current_len = 0
    snippets = []
    current_snippet = ""
    for sent in doc.sents:
        text = sent.text
        words = text.split()
        if current_len + len(words) > 200:
            snippets.append(current_snippet)
            current_snippet = text
            current_len = len(words)
        else:
            current_snippet += " " + text
            current_len += len(words)
    snippets.append(current_snippet) # the leftover part. 
    return snippets

### Main Function

# CPU-side prep: spaCy auto_split + word-level tokenization. Pulled out so the
# producer thread can run it in parallel with GPU work on a prior batch.
def prepare_pipeline_inputs(texts):
    if isinstance(texts, str):
        texts = [texts]

    all_snippets = []
    snippet_map = []
    for doc_id, text in enumerate(texts):
        snippets = auto_split(text)
        all_snippets.extend(snippets)
        snippet_map.extend([doc_id] * len(snippets))

    tokenized_snippets = [s.strip().split() for s in all_snippets]
    return texts, all_snippets, snippet_map, tokenized_snippets

# Runs the segmentation + generalization pipeline on a batch of texts. Takes str or list of str as input.
# When `prepared` is provided (the tuple returned by prepare_pipeline_inputs),
# the spaCy step is skipped — used by the producer/consumer fast path.
def run_pipeline(texts, model_batch_size=32, max_length=max_length, prepared=None):

    if prepared is None:
        texts, all_snippets, snippet_map, tokenized_snippets = prepare_pipeline_inputs(texts)
    else:
        texts, all_snippets, snippet_map, tokenized_snippets = prepared

    # Predict segmentation labels in batch
    all_labels = get_pred_clause_labels(
        all_snippets,
        tokenized_snippets,
        model_batch_size=model_batch_size,
        max_length=max_length,
    )

    # Reconstruct clauses
    all_clauses = []
    clause_map = []  # keep track of which text each clause came from
    for snip_id, (words, labels) in enumerate(zip(tokenized_snippets, all_labels)):
        clauses = reconstruct_clauses(words, labels)
        all_clauses.extend(clauses)
        clause_map.extend([snippet_map[snip_id]] * len(clauses))

    # Predict generalization labels in batch
    clause2labels = get_pred_generalization_labels(
        all_clauses, model_batch_size=model_batch_size, max_length=max_length
    )

    # Organize outputs back per input text
    results = [[] for _ in texts]
    results_with_labels = [[] for _ in texts]

    for i, clause in enumerate(all_clauses):
        doc_id = clause_map[i]
        results[doc_id].append((clause, str(len(results[doc_id]) + 1)))

    for s, l in clause2labels:
        doc_id = clause_map[all_clauses.index(s)]
        results_with_labels[doc_id].append((s, l))

    return results, results_with_labels

# extracts generalization labels from clause-segmented input
@torch.no_grad()
def get_pred_generalization_labels(clauses, model_batch_size=32, max_length=256):
    if not clauses:
        return []

    # Length-bucket: sort indices by clause length so each mini-batch contains
    # clauses of similar length, minimizing padding waste. Unsort at the end.
    order = sorted(range(len(clauses)), key=lambda i: len(clauses[i]))
    sorted_clauses = [clauses[i] for i in order]

    sorted_labels = [None] * len(sorted_clauses)
    for i in range(0, len(sorted_clauses), model_batch_size):
        batch_examples = sorted_clauses[i : i + model_batch_size]

        model_inputs = tokenizer(
            batch_examples,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            outputs = generalization_model(**model_inputs)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits

        pred_labels = logits.argmax(-1).cpu().numpy()
        pred_labels = [index2label[l] for l in pred_labels]

        for j, lab in enumerate(pred_labels):
            sorted_labels[i + j] = lab

    # Unsort back to original clause order
    clause2labels = [None] * len(clauses)
    for sorted_idx, orig_idx in enumerate(order):
        clause2labels[orig_idx] = (clauses[orig_idx], str(sorted_labels[sorted_idx]))

    return clause2labels

# generates clause segmentations from input
@torch.no_grad()
def get_pred_clause_labels(texts, tokenized_texts, model_batch_size=32, max_length=256):
    if not tokenized_texts:
        return []

    # Length-bucket: sort indices by word count so each mini-batch contains
    # snippets of similar length, minimizing padding waste. Unsort at the end.
    order = sorted(range(len(tokenized_texts)), key=lambda i: len(tokenized_texts[i]))
    sorted_words = [tokenized_texts[i] for i in order]

    sorted_labels = [None] * len(sorted_words)
    for i in range(0, len(sorted_words), model_batch_size):
        batch_words = sorted_words[i : i + model_batch_size]

        encoding = tokenizer(
            batch_words,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            outputs = clause_model(**encoding)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits

        pred_ids = logits.argmax(-1).cpu().numpy()

        for b_idx, words in enumerate(batch_words):
            word_ids = encoding.word_ids(batch_index=b_idx)
            labels = []
            seen = set()
            for j, word_idx in enumerate(word_ids):
                if word_idx is None or word_idx in seen:
                    continue
                labels.append(pred_ids[b_idx][j])
                seen.add(word_idx)
            sorted_labels[i + b_idx] = labels

    # Unsort back to original snippet order
    all_labels = [None] * len(tokenized_texts)
    for sorted_idx, orig_idx in enumerate(order):
        all_labels[orig_idx] = sorted_labels[sorted_idx]

    return all_labels

# Reconstructs segmented clauses from words + predicted labels.
def reconstruct_clauses(words, labels):

    segmented_clauses = []
    prev_label = 2
    current_clause = None
    for cur_token, cur_label in zip(words, labels):
        if prev_label == 2:
            current_clause = []
        if current_clause is not None:
            current_clause.append(cur_token)

        if cur_label == 2:
            if prev_label in [0, 1]:
                segmented_clauses.append(deepcopy(current_clause))
                current_clause = None
        prev_label = cur_label

    if current_clause is not None and len(current_clause) != 0:
        segmented_clauses.append(deepcopy(current_clause))

    return [" ".join(clause) for clause in segmented_clauses if clause is not None]

# Generates and writes labels for an entire month's worth of documents. If the output file already exists, we check the last processed row number and resume from there.
def label_generalization_file(file):

    missing_lines_count = 0  
    log_report(report_file_path, f"Started labeling {Path(file).name} from the {group} social group for generalization.")
    start_time = time.time()

    # Build output file path using the relative part from the input file.
    output_file_path = os.path.join(output_path, Path(file).name)

    # Determine resume position from existing output (if any).
    last_processed = get_last_source_row(
        output_file_path,
        report_file_path=report_file_path,
        file_for_log=file,
    )
    mode = "a" if last_processed >= 0 else "w"

    # Open input & output
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

        # If starting a new output file, write header as: input header + generalization columns
        if mode == "w":
            new_headers = in_header + [
                "clauses",
                "generalization_clause_labels",
                "genericity_generic_count","genericity_specific_count",
                "eventivity_stative_count","eventivity_dynamic_count",
                "boundedness_static_count","boundedness_episodic_count","habitual_count","NA_count",
                "genericity_generic_proportion","genericity_specific_proportion",
                "eventivity_stative_proportion","eventivity_dynamic_proportion",
                "boundedness_static_proportion","boundedness_episodic_proportion",
                "habitual_proportion","NA_proportion"
            ]
            writer.writerow(new_headers)

        # Aggregate result rows for a single batch and append to relevant_lines.
        def process_batch_result(batch_lines, result, relevant_lines):
            counts = {}
            individual_labels = {}
            clauses = {}
            props = {}

            for i_doc, text_result in enumerate(result):
                individual_labels[i_doc] = []
                clauses[i_doc] = []
                counts[i_doc] = {
                    "generic":0,"specific":0,"stative":0,"dynamic":0,
                    "static":0,"episodic":0,"habitual":0,
                    "NA genericity":0,"NA eventivity":0,"NA boundedness":0
                }
                for clause in text_result:
                    clauses[i_doc].append(clause[0])
                    individual_labels[i_doc].append(clause[1])
                    label_triplet = labels2attrs[clause[1]]
                    for j, feature in enumerate(label_triplet):
                        if "NA" not in feature:
                            counts[i_doc][feature] += 1
                        elif j == 0:
                            counts[i_doc]["NA genericity"] += 1
                        elif j == 1:
                            counts[i_doc]["NA eventivity"] += 1
                        else:
                            counts[i_doc]["NA boundedness"] += 1

                props[i_doc] = []
                gen_tot = counts[i_doc]['generic'] + counts[i_doc]['specific'] + counts[i_doc]['NA genericity']
                props[i_doc] += [
                    (counts[i_doc]['generic']/gen_tot) if gen_tot else 0.0,
                    (counts[i_doc]['specific']/gen_tot) if gen_tot else 0.0,
                ]
                eve_tot = counts[i_doc]['stative'] + counts[i_doc]['dynamic'] + counts[i_doc]['NA eventivity']
                props[i_doc] += [
                    (counts[i_doc]['stative']/eve_tot) if eve_tot else 0.0,
                    (counts[i_doc]['dynamic']/eve_tot) if eve_tot else 0.0,
                ]
                bou_tot = (counts[i_doc]['static'] + counts[i_doc]['episodic'] +
                           counts[i_doc]["habitual"] + counts[i_doc]['NA boundedness'])
                props[i_doc] += [
                    (counts[i_doc]['static']/bou_tot) if bou_tot else 0.0,
                    (counts[i_doc]['episodic']/bou_tot) if bou_tot else 0.0,
                    (counts[i_doc]['habitual']/bou_tot) if bou_tot else 0.0,
                    (counts[i_doc]['NA boundedness']/bou_tot) if bou_tot else 0.0,
                ]

            for i_doc in counts.keys():
                ind_clause = "\n".join(clauses[i_doc])
                ind_labels = "\n".join(individual_labels[i_doc])
                row_out = (batch_lines[i_doc] + [ind_clause, ind_labels,
                          counts[i_doc]['generic'], counts[i_doc]['specific'],
                          counts[i_doc]['stative'], counts[i_doc]['dynamic'],
                          counts[i_doc]['static'], counts[i_doc]['episodic'],
                          counts[i_doc]['habitual'], counts[i_doc]['NA boundedness']] + props[i_doc])
                relevant_lines.append(row_out)

        # Prefetch batches from a producer thread: while the GPU runs the
        # segmentation + generalization forwards for batch N, the producer
        # reads the CSV and runs spaCy auto_split for batch N+1 on the CPU.
        # spaCy is the dominant CPU cost here, so this overlap is the big win.
        # Queue depth 1 keeps memory bounded.
        batch_queue: "Queue" = Queue(maxsize=2)
        SENTINEL = object()
        producer_state = {"total_lines": 0, "missing_lines_count": 0}

        def producer():
            pending_lines = []
            try:
                for row_idx, line in enumerate(reader):
                    if len(line) < 3:
                        log_report(report_file_path, f"Skipping line {row_idx}: insufficient columns ({len(line)} found)")
                        producer_state["missing_lines_count"] += 1
                        continue

                    src_val = line[src_idx_in].strip()
                    src_num = int(src_val) if src_val.isdigit() else None
                    if src_num is not None and src_num <= last_processed:
                        continue

                    pending_lines.append(line)
                    producer_state["total_lines"] += 1

                    if len(pending_lines) == batch_size:
                        texts = [l[2].strip().replace("\n", " ") for l in pending_lines]
                        prepared = prepare_pipeline_inputs(texts)
                        batch_queue.put((pending_lines, prepared))
                        pending_lines = []

                if pending_lines:
                    texts = [l[2].strip().replace("\n", " ") for l in pending_lines]
                    prepared = prepare_pipeline_inputs(texts)
                    batch_queue.put((pending_lines, prepared))
            finally:
                batch_queue.put(SENTINEL)

        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        relevant_lines = []
        while True:
            item = batch_queue.get()
            if item is SENTINEL:
                break
            batch_lines, prepared = item
            _, result = run_pipeline(None, prepared=prepared)
            process_batch_result(batch_lines, result, relevant_lines)
            if relevant_lines:
                writer.writerows(relevant_lines)
                relevant_lines.clear()

        producer_thread.join()
        total_lines = producer_state["total_lines"]
        missing_lines_count = producer_state["missing_lines_count"]

    # generate processing report
    elapsed_minutes = (time.time() - start_time) / 60
    log_report(report_file_path, f"Finished generalization labeling {Path(file).name} for the {group} social group in {elapsed_minutes:.2f} minutes. Processed rows: {total_lines}")
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

### Main execution

# process each file and aggregate stats
if __name__ == "__main__":
    start_time = time.time()
    overall_docs = 0

    if args.array is not None: # for batch processing (Slurm array task)
        start = array * files_per_job
        end = min(start + files_per_job, len(file_list))
        if start >= len(file_list):
            raise RuntimeError(
                f"Array index {array} out of range for {len(file_list)} files (files_per_job={files_per_job})."
            )
        for file in file_list[start:end]:
            overall_docs += label_generalization_file(file)

    else: # for sequential processing
        for file in file_list:        
            overall_docs += label_generalization_file(file)

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

        overall_elapsed = (time.time() - start_time) / 60
        log_report(report_file_path, f"Generalization labeling for the {group} social group for {args.years} finished in {overall_elapsed:.2f} minutes. Total processed rows: {overall_docs}")

        # Aggregate overall statistics and save final summary report
        final_report = [
            ["Timestamp", "Social Group", "Years", "Total Processed Rows", "Total Elapsed Time (min)"],
            [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), group, args.years, overall_docs, f"{overall_elapsed:.2f}"]
        ]
        final_report_file = os.path.join(output_path, "final_report_label_generalization.csv")
        with open(final_report_file, "a+", encoding="utf-8", newline="") as rf:
            writer = csv.writer(rf)
            writer.writerows(final_report)
        log_report(report_file_path, f"Final summary report saved to: {final_report_file}")
