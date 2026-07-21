### Imports

## helper functions, kept in a separate file for readability
from cli import get_args, MODELS_DIR, DATA_DIR
from utils import dataset_split, split_dataset_to_file, split_dataset_from_file, f1_calculator

## needed python packages and functions
import torch
from transformers.file_utils import is_torch_available
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
import numpy as np
import csv
csv.field_size_limit(2**31 - 1) # Increase the field size limit to handle larger fields
import random
from sklearn.utils import compute_class_weight
from collections import Counter
from datetime import datetime
import os
from pathlib import Path

### Argument Handling
args = get_args()
group = args.group          # e.g. "race"
type_ = args.type           # "comments" or "submissions"

### Run settings / modes

# NOTE: Exactly one of these three should be active at a time:

trial = 2   # identifies training run trial number. Used only when `use_trial_suffix_in_path = True`.
use_trial_suffix_in_path = False  # If True, model_path = filter_relevance_{group}_{model_name}_{trial}. If False, use the bare folder.

## 1) Train a model from the original data and evaluate on original test set
training = True

## 2) Just evaluate inference performance on the original test set
#    (change thresholding params etc. and set training = False, retraining = False)
#    --> This mode uses the model saved under `model_path`.
#    NOTE: if both training and retraining are False, this is the default mode.

## 3) Retrain (fine-tune) an already-trained model on additional data
#    and evaluate on a held-out retraining test set.
retraining = False
retrain_trial = 8 # identifies the retraining run

### Hyperparameters

## Training
max_length = 512           # max tokens per document
train_batch_size = 8       # base-sized models (roberta-base / twitter-roberta) fit fine at 8
retrain_batch_size = 8     # batch size for retraining
epochs = 20                # high ceiling; early stopping (F1, patience 2) cuts it when val F1 drops twice
custom_training = False    # use penalty for specific mistakes if True (see the next line)
penalize_confusion = "0_to_1" # which direction in mistakes is more important to fix: [true label]_to_[wrong_classification]
penalty_weight = 0.5       # only matters if custom_training is True
num_annot = 2              # number of annotators in original data

## Thresholding (used only at inference time)
thresholding = True       # if True, apply custom thresholding logic
threshold_class = 1        # which class gets thresholding
threshold = 0.6            # probability threshold

## Sanity: don’t allow “train” and “retrain” simultaneously
if training and retraining:
    raise ValueError("Set at most one of `training` or `retraining` to True.")

### Logging

# records the parameters for logging purposes
def collect_run_params():
    return {
        # Run identity
        "group": group,
        "type": type_,
        "trial": trial,
        "training": training,
        "retraining": retraining,
        "retrain_trial": retrain_trial if retraining else None,

        # Model
        "model_name": model_name,
        "max_length": max_length,

        # Training hyperparameters
        "train_batch_size": train_batch_size,
        "retrain_batch_size": retrain_batch_size,
        "epochs": epochs,
        "custom_training": custom_training,
        "penalize_confusion": penalize_confusion,
        "penalty_weight": penalty_weight,
        "num_annot": num_annot,

        # Thresholding
        "thresholding": thresholding,
        "threshold_class": threshold_class,
        "threshold": threshold,
    }

### Path Handling

# NOTE: Model loading assumes the default pathing.
if group in ("mh", "sport"):
    # DOMAIN MATCH is the winning lever: twitter-roberta is pretrained on informal social-media
    # text (much closer to Reddit than roberta's books/wiki). For `mh` this broke the ceiling:
    # P(mh)max 0.57->1.00, gate F1 0.68->0.79. roberta-large (3x capacity) did NOT help (0.66).
    # Applying the same fix to `sport` (previously roberta-base, P(sport)max stuck at 0.68).
    model_name = "cardiffnlp/twitter-roberta-base"
elif group in ("skin_tone", "mental_health"):
    model_name = "roberta-base"
else:
    model_name = "roberta-large"

if use_trial_suffix_in_path:
    model_path = os.path.join(
        MODELS_DIR, f"filter_relevance_{group}_{model_name}_{trial}",
    )
else:
    model_path = os.path.join(MODELS_DIR, f"filter_relevance_{group}")

retrain_path = os.path.join(
    MODELS_DIR,
    f"retrain_relevance_{group}_{model_name}_{retrain_trial}",
)

# where to find the rated relevance samples
ratings_path = DATA_DIR / "data_relevance_ratings" / type_
reratings_path = DATA_DIR / "data_relevance_QAratings"

### Utilities

## Label mappings
title_label = {"Irrelevant": 0, "Relevant": 1}
label_title = {0: "Irrelevant", 1: "Relevant"}
num_labels = len(label_title)

## Random seed for reproducability
def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(1)

## tokenizer is shared across all modes
tokenizer = AutoTokenizer.from_pretrained(model_name)

## device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Performing computations on {device}")

### Data loading helpers

# Load the original ratings for training/validation/test splitting.
# NOTE: Assumes two annotators with files: relevance_sample_{group}_{rater}_rated.csv and uses "x" / numeric rating in column 3.
def load_main_annotations(ratings_path, group: str, num_annot: int):

    texts = {}
    ratings = {i: {} for i in range(num_annot)}

    for rater in range(num_annot):
        fname = os.path.join(
            ratings_path,
            f"relevance_sample_{group}_{rater}_rated.csv"
        )
        with open(fname, "r", encoding="utf-8-sig", errors="ignore") as f:
            reader = csv.reader(f)
            for idx, line in enumerate(reader):
                if idx == 0 or not line:
                    continue
                doc_id_raw = line[0].strip()
                if not doc_id_raw:
                    continue
                doc_id = int(doc_id_raw)
                # Binarize: only the literal "1" counts as relevant; everything else
                # ("0", "x", "-1", blank, etc.) is treated as 0.
                cell = line[2].strip() if len(line) >= 3 else ""
                v = 1 if cell == "1" else 0
                # If a random_id appears more than once in the same file, OR the labels.
                ratings[rater][doc_id] = max(ratings[rater].get(doc_id, 0), v)
                if rater == 0:
                    texts[doc_id] = line[1].strip()

    # Use only doc_ids that appear in every rater's file
    common_ids = set(ratings[0].keys())
    for r in range(1, num_annot):
        common_ids &= set(ratings[r].keys())

    # Consider a post relevant if at least one annotator marked it relevant
    final_ratings = {}
    for doc_id in common_ids:
        if all(ratings[r][doc_id] == 0 for r in range(num_annot)):
            final_ratings[doc_id] = 0
        else:
            final_ratings[doc_id] = 1

    final_texts = [texts[did] for did in final_ratings.keys()]
    final_labels = list(final_ratings.values())

    print(f"Number of annotated docs used in training: {len(final_texts)}")
    print(f"Number of instances for each label: {Counter(final_labels)}")

    return final_texts, final_labels

# Load the additional Q/A retraining data.
# NOTE: Assumes a single file like: {group}_retraining_sample_rated.csv with text at column 2 and rating in the last column.
def load_retraining_annotations(reratings_path, group: str):

    retexts = {}
    reratings = {}

    fname = os.path.join(
        reratings_path,
        f"qa_r_retraining_input_{group}_n400_rated.csv"
    )

    with open(fname, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for idx, line in enumerate(reader):
            if idx == 0 or not line:
                continue
            doc_id = line[0]
            retexts[doc_id] = line[2].strip()
            if line[-1].strip() == "x":
                reratings[doc_id] = 0
            else:
                reratings[doc_id] = int(line[-1].strip())

    retext_list = list(retexts.values())
    relabel_list = list(reratings.values())

    print(f"Number of annotated docs used in retraining: {len(retext_list)}")
    print(f"Number of instances for each label in retraining: {Counter(relabel_list)}")

    return retext_list, relabel_list

# Create or load an 80/10/10 train/validation/test split, and persist it. Shared between original data and retraining data.
def prepare_splits(texts, labels, split_dir: str, group: str, description: str = ""):
    """
    
    """
    os.makedirs(split_dir, exist_ok=True)

    split_data = ["training", "validation", "test"]
    file_list = []
    for cat in split_data:
        file_list.append(os.path.join(split_dir, f"text_{group}_{cat}.csv"))
        file_list.append(os.path.join(split_dir, f"label_{group}_{cat}.txt"))

    missing_file = any(not os.path.exists(f) for f in file_list)

    # Guard against a STALE cache: if the ratings CSVs changed (relabeled or augmented) since the
    # split was written, the cached split silently ignores the new data. Detect this by comparing
    # the cached (text -> label) mapping against the current one; rebuild on any mismatch.
    stale = False
    if not missing_file:
        try:
            cached_texts = (split_dataset_from_file(file_list[0])
                            + split_dataset_from_file(file_list[2])
                            + split_dataset_from_file(file_list[4]))
            cached_labels = (split_dataset_from_file(file_list[1])
                             + split_dataset_from_file(file_list[3])
                             + split_dataset_from_file(file_list[5]))
            cached_map = {t: str(l) for t, l in zip(cached_texts, cached_labels)}
            current_map = {t: str(l) for t, l in zip(texts, labels)}
            if cached_map != current_map:
                stale = True
                print(f"[split] cached {description} split is STALE (data changed: "
                      f"{len(cached_map)} cached vs {len(current_map)} current) -> rebuilding")
        except Exception as e:
            stale = True
            print(f"[split] could not verify cached split ({e}) -> rebuilding")

    if missing_file or stale:
        print(f"Creating {description} training, validation and test sets (80/10/10 split)")

        train_texts, valid_texts_init, train_labels, valid_labels_init = dataset_split(
            texts, labels, proportion=0.8
        )
        valid_texts, test_texts, valid_labels, test_labels = dataset_split(
            valid_texts_init, valid_labels_init, proportion=0.5
        )

        split_dataset_to_file(file_list[0], train_texts)
        split_dataset_to_file(file_list[1], train_labels)
        split_dataset_to_file(file_list[2], valid_texts)
        split_dataset_to_file(file_list[3], valid_labels)
        split_dataset_to_file(file_list[4], test_texts)
        split_dataset_to_file(file_list[5], test_labels)
    else:
        print(f"Loading predetermined {description} training, validation and test sets (80/10/10 split)")
        train_texts = split_dataset_from_file(file_list[0])
        train_labels = split_dataset_from_file(file_list[1])
        valid_texts = split_dataset_from_file(file_list[2])
        valid_labels = split_dataset_from_file(file_list[3])
        test_texts = split_dataset_from_file(file_list[4])
        test_labels = split_dataset_from_file(file_list[5])

    return train_texts, train_labels, valid_texts, valid_labels, test_texts, test_labels

def summarize_split(name, texts, labels):
    print(f"Number of {name} documents: {len(texts)}")
    print(f"Number of instances for each label in {name} data: {Counter(labels)}")

def tensorize_labels(labels):
    return torch.from_numpy(np.array(labels)).type(torch.LongTensor)

class RelevanceDataset(torch.utils.data.Dataset):
    """
    Simple dataset wrapper used both for original training and retraining.
    """

    def __init__(self, encodings, labels_tensor):
        self.encodings = encodings
        self.labels = labels_tensor

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        # scalar label, not wrapped in a list
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        # keep in sync with encodings to avoid index errors
        first_key = next(iter(self.encodings.keys()))
        return len(self.encodings[first_key])
    
def make_training_args(
    epochs,
    train_batch_size,
    logging_steps=20,
    warmup_steps=20,
    description="",
):
    is_large = (model_name == "roberta-large")
    # roberta-large on a 6GB card can only fit train_batch_size=2. That effective batch is
    # too noisy for the large model to escape its initialized state (epoch-1 collapse to F1 0,
    # loss 0.699 = chance). Accumulate to effective batch 16 + longer warmup to stabilize.
    grad_accum = 8 if is_large else 1
    lr = 2e-5 if is_large else 1e-5   # base: 1e-5 best; large tolerates/needs higher LR with warmup + big eff-batch
    return TrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=grad_accum,   # effective batch = train_batch_size * grad_accum
        gradient_checkpointing=is_large,   # fit large model on 6GB
        learning_rate=lr,
        fp16=torch.cuda.is_available(),   # mixed precision on GPU: ~3x faster (instability is from LR, not precision)
        warmup_ratio=0.1 if is_large else 0.0,   # smooth cold-start for large model
        warmup_steps=0 if is_large else warmup_steps,
        weight_decay=0.01,
        logging_dir="./logs",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        logging_steps=logging_steps,
        save_strategy="epoch",
        eval_strategy="epoch",
    )

# Per-epoch validation metrics; early stopping monitors eval_f1 (Babak: stop when F1 drops twice).
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1 = f1_calculator([int(l) for l in labels], [int(p) for p in preds])
    return {"f1": f1, "precision": precision, "recall": recall}

# Create a Trainer subclass that uses class weights, optionally with additional penalty for predicting 1 when the true label is 0 or vice versa.
def make_weighted_trainer(
    model,
    train_dataset,
    eval_dataset,
    training_args,
    class_weights,
    custom_training=False,
    penalty_weight=1.0,
    penalize_confusion="1_to_0",  # "1_to_0", "0_to_1", or None
):

    weights_tensor = torch.tensor(
        [float(w) for w in class_weights],
        dtype=torch.float,
    )

    valid_options = ("1_to_0", "0_to_1", None)
    if penalize_confusion not in valid_options:
        raise ValueError(
            f"penalize_confusion must be one of {valid_options}, "
            f"got {penalize_confusion!r}"
        )

    class WeightedTrainer(Trainer):
        def compute_loss(
            self,
            model,
            inputs,
            return_outputs: bool = False,
            num_items_in_batch: int | None = None,  
        ):
            # Standard class-weighted CE loss
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")

            loss_fct = torch.nn.CrossEntropyLoss(
                weight=weights_tensor.to(logits.device)
            )
            ce_loss = loss_fct(
                logits.view(-1, model.config.num_labels),
                labels.view(-1),
            )

            if not custom_training or penalize_confusion is None: # If no custom training / penalty
                loss = ce_loss
            else:
                # confusion-based penalty
                preds = torch.argmax(logits, dim=-1)

                if penalize_confusion == "1_to_0":
                    # penalize false negatives
                    confusion_mask = (labels == 1) & (preds == 0)
                elif penalize_confusion == "0_to_1":
                    # penalize false positives
                    confusion_mask = (labels == 0) & (preds == 1)
                else:
                    confusion_mask = None

                if confusion_mask is not None:
                    penalty = penalty_weight * confusion_mask.sum().float()
                    loss = ce_loss + penalty
                else:
                    loss = ce_loss

            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

## Prediction and evaluation helpers

def get_prediction(
    text,
    threshold_class=threshold_class,
    threshold=threshold,
    thresholding=thresholding,
):
    """
    Run the trained model on a single string and return the predicted class index.
    """
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs[0].softmax(1)[0]  # (2,) for binary classification

    if thresholding:
        # If confident enough for the thresholded class, predict it
        if probs[threshold_class] > threshold:
            return threshold_class
        # Otherwise, disallow that class and pick the other
        masked_probs = probs.clone()
        masked_probs[threshold_class] = -1
        return masked_probs.argmax().item()
    else:
        return probs.argmax().item()

def evaluate_and_save(
    texts,
    labels,
    csv_path,
    txt_path,
    description="test",
):
    print(f"\nEvaluating on {description} set...")
    predictions = [get_prediction(text) for text in texts]

    # Save per-example results
    with open(csv_path, "w", encoding="utf-8", errors="ignore", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "true_label", "predicted_label"])
        for text, true_label, pred in zip(texts, labels, predictions):
            writer.writerow(
                [text, label_title[int(true_label)], label_title[int(pred)]]
            )

    precision, recall, f1 = f1_calculator(
        [int(l) for l in labels],
        [int(p) for p in predictions],
    )

    with open(txt_path, "a+", encoding="utf-8", errors="ignore") as f:
        f.write("***{}***\n".format(datetime.now()))

        run_params = collect_run_params()
        f.write("Run parameters:\n")
        for k, v in run_params.items():
            f.write(f"{k}: {v}\n")

        f.write("\nPerformance on the test set:\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1: {f1}\n")

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")

### Main flow

# 1. Original data: load, split, tokenize, and build datasets
texts, labels = load_main_annotations(ratings_path, group, num_annot)

# where we store the original train/val/test split files
split_data_path = os.path.join(MODELS_DIR, "train_relevance_data_split")

(
    train_texts,
    train_labels_raw,
    valid_texts,
    valid_labels_raw,
    test_texts,
    test_labels_raw,
) = prepare_splits(
    texts,
    labels,
    split_data_path,
    group,
    description="original",
)

# Labels read back from disk come in as strings; coerce to int so numpy/torch are happy.
train_labels_raw = [int(l) for l in train_labels_raw]
valid_labels_raw = [int(l) for l in valid_labels_raw]
test_labels_raw = [int(l) for l in test_labels_raw]

# print out stats for the splits
summarize_split("training", train_texts, train_labels_raw)
summarize_split("validation", valid_texts, valid_labels_raw)
summarize_split("test", test_texts, test_labels_raw)

# convert labels to tensors
train_labels = tensorize_labels(train_labels_raw)
valid_labels = tensorize_labels(valid_labels_raw)
test_labels = tensorize_labels(test_labels_raw)

# class weights for original training
orig_weights = list(
    compute_class_weight(
        "balanced",
        classes=np.asarray(np.unique(train_labels)),
        y=np.asarray(train_labels),
    )
)

print(f"Class weights to account for imbalanced training data: {orig_weights}")

# tokenize original splits
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)

# build datasets for original training
train_dataset = RelevanceDataset(train_encodings, train_labels)
valid_dataset = RelevanceDataset(valid_encodings, valid_labels)

# training args for original training
training_args = make_training_args(
    epochs=epochs,
    train_batch_size=train_batch_size,
    logging_steps=50,
    warmup_steps=60,
    description="original training",
)

# Train or load original model

if training:
    print(f"\nTraining a new classifier for relevance of posts to the {group} social group...")

    # fresh model from HF hub
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    ).to(device)

    trainer = make_weighted_trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        training_args=training_args,
        class_weights=orig_weights,
        custom_training=custom_training,
        penalty_weight=penalty_weight,
        penalize_confusion=penalize_confusion,
        )

    trainer.train()

    # save model + tokenizer
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

else:
    print(f"\nLoading a pretrained classifier for relevance of posts to the {group} social group...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

# global `model` is used by get_prediction
model.eval()

# retraining on additional data

if retraining:
    print("\n*** Retraining on additional Q/A data ***")

    # load retraining annotations
    retexts, reratings = load_retraining_annotations(reratings_path, group)

    # prepare/train/val/test splits for retraining
    split_redata_path = os.path.join(retrain_path, "retrain_relevance_data_split")

    (
        retrain_texts,
        retrain_labels_raw,
        revalid_texts,
        revalid_labels_raw,
        retest_texts,
        retest_labels_raw,
    ) = prepare_splits(
        retexts,
        reratings,
        split_redata_path,
        group,
        description="retraining",
    )

    # print stats for the splits
    summarize_split("retraining", retrain_texts, retrain_labels_raw)
    summarize_split("revalidation", revalid_texts, revalid_labels_raw)
    summarize_split("retest", retest_texts, retest_labels_raw)

    # prepare the labels for Torch
    retrain_labels = tensorize_labels(retrain_labels_raw)
    revalid_labels = tensorize_labels(revalid_labels_raw)
    retest_labels = tensorize_labels(retest_labels_raw)

    # class weights for retraining
    retrain_weights = list(
        compute_class_weight(
            "balanced",
            classes=np.asarray(np.unique(retrain_labels)),
            y=np.asarray(retrain_labels),
        )
    )
    print(f"Class weights to account for imbalanced retraining data: {retrain_weights}")

    # tokenize retraining splits
    retrain_encodings = tokenizer(
        retrain_texts, truncation=True, padding=True, max_length=max_length
    )
    revalid_encodings = tokenizer(
        revalid_texts, truncation=True, padding=True, max_length=max_length
    )
    retest_encodings = tokenizer(
        retest_texts, truncation=True, padding=True, max_length=max_length
    )

    # build datasets
    retrain_dataset = RelevanceDataset(retrain_encodings, retrain_labels)
    revalid_dataset = RelevanceDataset(revalid_encodings, revalid_labels)

    # retraining-specific TrainingArguments
    retraining_args = make_training_args(
        epochs=1,
        train_batch_size=retrain_batch_size,
        logging_steps=10,
        warmup_steps=10,
        description="retraining",
    )

    # fine-tune the already-loaded model
    retrainer = make_weighted_trainer(
    model=model,
    train_dataset=retrain_dataset,
    eval_dataset=revalid_dataset,
    training_args=retraining_args,
    class_weights=retrain_weights,
    custom_training=custom_training,
    penalty_weight=penalty_weight,
    penalize_confusion=penalize_confusion,
    )

    # perform the retraining
    retrainer.train()
    model.eval()

    # save retrained model
    os.makedirs(retrain_path, exist_ok=True)
    model.save_pretrained(retrain_path)
    tokenizer.save_pretrained(retrain_path)

    # valuate retrained model on RETRAINING test set
    retest_csv = os.path.join(
        retrain_path,
        f"retest_results_{group}_{model_name}_{retrain_trial}.csv",
    )
    retest_txt = os.path.join(
        retrain_path,
        f"retest_results_{group}_{model_name}_{retrain_trial}.txt",
    )

    # log the results and parameters
    evaluate_and_save(
        texts=retest_texts,
        labels=retest_labels_raw,
        csv_path=retest_csv,
        txt_path=retest_txt,
        description="retraining test set (held-out new data)",
    )

    # Evaluate retrained model on ORIGINAL test set
    origtest_csv = os.path.join(
        retrain_path,
        f"origtest_results_{group}_{model_name}_{retrain_trial}.csv",
    )
    origtest_txt = os.path.join(
        retrain_path,
        f"origtest_results_{group}_{model_name}_{retrain_trial}.txt",
    )

    evaluate_and_save(
        texts=test_texts,
        labels=test_labels_raw,
        csv_path=origtest_csv,
        txt_path=origtest_txt,
        description="original test set (after retraining)",
    )

# Final: evaluation on the original test set
#   - Path 1 (training=True, retraining=False): evaluate trained model
#   - Path 2 (training=False, retraining=False): evaluate loaded model
#   - Path 3 (retraining=True): already evaluated above, skip

if not retraining:
    test_csv = os.path.join(
        model_path,
        f"test_results_{group}_{model_name}_{trial}.csv",
    )
    test_txt = os.path.join(
        model_path,
        f"test_results_{group}_{model_name}_{trial}.txt",
    )

    evaluate_and_save(
        texts=test_texts,
        labels=test_labels_raw,
        csv_path=test_csv,
        txt_path=test_txt,
        description="original test",
    )
