### Imports

# import functions and objects
from cli import get_args, DATA_DIR
from utils import parse_range, log_report, check_reqd_files, get_last_source_row
import os
import csv
csv.field_size_limit(2**31 - 1) # Increase the field size limit to handle larger fields
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import stanza
from textblob import TextBlob
import numpy as np
import time
import datetime
import re
from pathlib import Path

### Argument Handling

# Extract and transform CLI arguments 
args = get_args()
years = parse_range(args.years)
group = args.group
type_ = args.type
batch_size = args.batchsize
if args.array is not None:
    array = args.array
files_per_job = getattr(args, "files_per_job", 1)
if files_per_job is None or files_per_job < 1:
    files_per_job = 1

### Path Handling

# set path variables

if not args.input:
    input_path = DATA_DIR / "data_reddit_curated" / group / type_ / "labeled_moralization"
else:
    input_path = Path(args.input)

if not args.output:
    output_path = DATA_DIR / "data_reddit_curated" / group / type_ / "labeled_sentiment"
else:
    output_path = Path(args.output)

output_path.mkdir(parents=True, exist_ok=True)

# prepare the report file
report_file_path = os.path.join(output_path, f"report_label_sentiment.csv")
log_report(report_file_path)

# Build file_list organized by year and raise an error if an expected file is missing 
file_list = check_reqd_files(years, input_path, type_)
file_list = sorted(file_list, key=lambda p: Path(p).name)

### Main Functions

## Model inclusion rationale:
# VADER: to include a rule-based one focused on social media data
# TextBlob: for a neural continuous measure (trained on movie reviews)
# Stanza: for a neural categorical measure (broader training data)

# generates labels for an entire month's worth of documents. It resumes labeling if it comes across incomplete output.
def label_sentiment_file(file):

    missing_lines_count = 0
    first_error_logged = False
    log_report(report_file_path, f"Started labeling {Path(file).name} from the {group} social group for sentiment.")
    start_time = time.time()

    # Compute output path that mirrors input directory structure
    output_file_path = os.path.join(output_path, Path(file).name)

    # determine resume position from any existing output
    last_processed = get_last_source_row(
        output_file_path,
        report_file_path=report_file_path,
        file_for_log=file,
    )
    mode = "a" if last_processed >= 0 else "w"

    # sentiment tools
    try:
        _ = nlp, analyzer  # type: ignore  # these are set later in your script
    except NameError:
        stanza.download('en', processors='tokenize,sentiment', verbose=False)
        globals()["nlp"] = stanza.Pipeline(lang='en', processors='tokenize,sentiment')
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        globals()["analyzer"] = SentimentIntensityAnalyzer()

    # batching
    try:
        BATCH = int(batch_size)  # use your global if present
    except Exception:
        BATCH = 1200

    # process input
    total_lines_written = 0
    with open(file, "r", encoding="utf-8-sig", errors="ignore", newline="") as f_in, \
         open(output_file_path, mode, encoding="utf-8", newline="") as f_out:

        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        # Read input header and find its 'source_row'
        try:
            in_header = next(reader)
        except StopIteration:
            # Empty input file
            return 0

        try:
            source_row_in_idx = in_header.index("source_row")
        except ValueError:
            raise RuntimeError("Input file is missing required 'source_row' column.")

        # If we're starting a brand-new output file, write headers
        if mode == "w":
            new_headers = in_header + [
                "Sentiment_Stanza_pos", "Sentiment_Stanza_neu", "Sentiment_Stanza_neg",
                "Sentiment_Vader_compound",
                "Sentiment_TextBlob_Polarity", "Sentiment_TextBlob_Subjectivity"
            ]
            writer.writerow(new_headers)

        # Prepare batch buffers
        batch_lines = []
        batch_input_rows = []  # keep the original input rows in parallel

        def flush_batch():
            nonlocal total_lines_written
            if not batch_lines:
                return
            # Collect sentiments for each line in the batch
            out_rows = []
            for line in batch_lines:
                try:
                    # Stanza sentence-level aggregation
                    doc = nlp(line[2].strip().replace("\n", " "))
                    stanza_counts = {"pos": 0, "neu": 0, "neg": 0}
                    stanza_labels = {0: "neg", 1: "neu", 2: "pos"}
                    vader_scores = []
                    for sentence in doc.sentences:
                        stanza_counts[stanza_labels[sentence.sentiment]] += 1
                        vader_scores.append(analyzer.polarity_scores(sentence.text)["compound"])

                    # TextBlob document-level
                    tb = TextBlob(line[2].strip().replace("\n", " "))

                    # Append only new columns; keep the original row intact
                    out_row = line + [
                        stanza_counts["pos"], stanza_counts["neu"], stanza_counts["neg"],
                        (np.mean(vader_scores) if vader_scores else 0.0),
                        tb.sentiment.polarity, tb.sentiment.subjectivity
                    ]
                    out_rows.append(out_row)
                except Exception as e:
                    # Count and skip any bad line, but keep going. Log the first
                    # failure of the file with traceback + source-row context so
                    # silent counts don't hide a systemic problem.
                    nonlocal missing_lines_count, first_error_logged
                    missing_lines_count += 1
                    if not first_error_logged:
                        first_error_logged = True
                        try:
                            src_id = line[source_row_in_idx]
                        except Exception:
                            src_id = "?"
                        import traceback as _tb
                        tb = "".join(_tb.format_exception(type(e), e, e.__traceback__))
                        log_report(
                            report_file_path,
                            f"[warn] {Path(file).name}: first sentiment-batch failure at source_row={src_id}: {e}\n{tb}",
                        )

            if out_rows:
                writer.writerows(out_rows)
                total_lines_written += len(out_rows)

            # clear buffers
            batch_lines.clear()
            batch_input_rows.clear()

        # Iterate the input, skip header
        for row_idx, row in enumerate(reader, start=2):  # start=2 because header is row 1
            # Guard against short/blank lines (expect at least 3 columns given usage of row[2] for text)
            if len(row) < 3:
                missing_lines_count += 1
                if not first_error_logged:
                    first_error_logged = True
                    log_report(
                        report_file_path,
                        f"[warn] {Path(file).name}: first malformed row at line {row_idx} (only {len(row)} columns); skipping",
                    )
                continue

            # Use the input's source_row to decide skipping/resume
            src_value = row[source_row_in_idx].strip()
            if not src_value.isdigit():
                # If malformed (e.g. empty), treat as missing and process anyway to avoid skipping the whole file accidentally.
                src_num = None
            else:
                src_num = int(src_value)

            if src_num is not None and src_num <= last_processed:
                continue  # already processed in previous run

            batch_lines.append(row)
            batch_input_rows.append(row)

            if len(batch_lines) >= BATCH:
                flush_batch()

        # Flush any remainder
        flush_batch()

    # generate processing report
    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    log_report(report_file_path, f"Finished sentiment labeling {Path(file).name} for the {group} social group in {elapsed_minutes:.2f} minutes. Processed rows: {total_lines_written}")

    if missing_lines_count > 0:
        missing_records_file = os.path.join(output_path, 'missing_records.csv')
        need_header = not os.path.exists(missing_records_file)
        with open(missing_records_file, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if need_header:
                w.writerow(['Filename', 'MissingLinesCount', 'Timestamp'])
            w.writerow([str(file), missing_lines_count, datetime.datetime.now().isoformat(timespec="seconds")])

    return total_lines_written

### Main execution

# process each file and aggregate stats
if __name__ == "__main__":

    start_time = time.time()
    overall_docs = 0

    # Process each file from the file_list (global mode)

    # create the analyzer objects
    nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment')
    analyzer = SentimentIntensityAnalyzer()

    # Process each file from the file_list (global mode)
    if args.array is not None: # for batch processing (Slurm array task)
        start = array * files_per_job
        end = min(start + files_per_job, len(file_list))
        if start >= len(file_list):
            raise RuntimeError(
                f"Array index {array} out of range for {len(file_list)} files (files_per_job={files_per_job})."
            )
        for file in file_list[start:end]:
            overall_docs += label_sentiment_file(file)

    else: # for sequential processing
        for file in file_list:        
            overall_docs += label_sentiment_file(file)

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
        log_report(report_file_path, f"Sentiment labeling for the {group} social group for {args.years} finished in {overall_elapsed:.2f} minutes. Total processed rows: {overall_docs}")

        # Aggregate overall statistics and save final summary report 
        final_report = [
            ["Timestamp", "Social Group", "Years", "Total Processed Rows", "Total Elapsed Time (min)"],
            [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), group, args.years, overall_docs, f"{overall_elapsed:.2f}"]
        ]
        final_report_file = os.path.join(output_path, "final_report_label_sentiment.csv")
        with open(final_report_file, "a+", encoding="utf-8", newline="") as rf:
            writer = csv.writer(rf)
            writer.writerows(final_report)
        log_report(report_file_path, f"Final summary report saved to: {final_report_file}")