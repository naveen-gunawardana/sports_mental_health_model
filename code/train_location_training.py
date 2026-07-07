### Imports

from __future__ import annotations

import json
import os
import pickle
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import load_npz, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from cli import get_args, MODELS_DIR
from utils import resolve_word_feature_src

# Logging
VERBOSITY = int(os.environ.get("VERBOSITY", "1"))

def log(msg: str, level: int = 1, stream=None):
    if level <= VERBOSITY:
        print(msg, file=stream)

### Argument Handling

args = get_args()
type_ = args.type

### Path Handling

# where the preprocessed feature sets are stored
if not args.input: 
    PREPROC_PATH = os.path.join(MODELS_DIR, "label_location", "preprocessed_streaming")
else:
    PREPROC_PATH = args.input

# where the trained models will be saved
if not args.output:
    MODEL_DIR = os.environ.get(
        "TRAIN_OUTPUT_DIR",
        os.path.join(MODELS_DIR, "label_location", "trained_lr"),
    )
else:
    MODEL_DIR = args.output
os.makedirs(MODEL_DIR, exist_ok=True)

### Preprocessing Hyperparameters
# NOTE: These must match train_location_preprocess.py for accurate model logging.

DEFAULT_WORD_FEATURE_SRC = os.environ.get("WORD_FEATURE_SRC", "all").strip().lower()
WORD_SELECTOR = os.environ.get("WORD_SELECTOR", "freq").strip().lower()
WORD_STAT = os.environ.get("WORD_STAT", "df").strip().lower()
WORD_MIN_DF = int(os.environ.get("WORD_MIN_DF", "2"))
WORD_MIN_TOTAL_COUNT = int(os.environ.get("WORD_MIN_TOTAL_COUNT", "2"))
WORD_CANDIDATE_POOL = int(os.environ.get("WORD_CANDIDATE_POOL", "250000"))
WORD_TOP_K = int(os.environ.get("WORD_TOP_K", "50000"))
WORD_SUBLINEAR_TF = os.environ.get("WORD_SUBLINEAR_TF", "1") == "1"
WORD_USE_IDF = os.environ.get("WORD_USE_IDF", "1") == "1"
WORD_SMOOTH_IDF = os.environ.get("WORD_SMOOTH_IDF", "1") == "1"
WORD_NORM = os.environ.get("WORD_NORM", "l2").strip().lower()
SUBREDDIT_TOP_K = int(os.environ.get("SUBREDDIT_TOP_K", "20000"))
SUBREDDIT_MIN_DF = int(os.environ.get("SUBREDDIT_MIN_DF", "2"))
STRUCT_SUB_MODE = os.environ.get("STRUCT_SUB_MODE", "log1p_l1").strip().lower()
STRUCT_HOUR_MODE = os.environ.get("STRUCT_HOUR_MODE", "l1").strip().lower()

### Model Training Hyperparameters

FEATURE_SET = os.environ.get("FEATURE_SET", "words").strip().lower()  # words | struct | combined
TASK = os.environ.get("TASK", "state").strip().lower()  # top (US/non-US) | state (US states) | region (for non-US; Europe,Africa,Asia_Oceania,Americas)
MODEL_SOLVER = os.environ.get("MODEL_SOLVER", "lbfgs").strip().lower() # logistic regression solving algorithm
MODEL_C = float(os.environ.get("MODEL_C", "3.0")) # regularization strength
MODEL_MAX_ITER = int(os.environ.get("MODEL_MAX_ITER", "200")) # max number of model iterations
MODEL_TOL = float(os.environ.get("MODEL_TOL", "1e-4")) # tolerance for probability aggregation misalignments
MODEL_N_JOBS = int(os.environ.get("MODEL_N_JOBS", "1"))
MODEL_CLASS_WEIGHT = os.environ.get("MODEL_CLASS_WEIGHT", "balanced").strip().lower() # weighting classes based on occurrence
MODEL_RANDOM_STATE = int(os.environ.get("MODEL_RANDOM_STATE", "1337")) # for reproducibility
SAVE_MODEL = os.environ.get("SAVE_MODEL", "1") == "1"
TRAIN_EVAL_LIMIT = int(os.environ.get("TRAIN_EVAL_LIMIT", "0")) # how many training examples are used for evaluation. For debugging/validation purposes. 0 uses the full dataset. 
DEBUG_TOP_N = int(os.environ.get("DEBUG_TOP_N", "15")) # how many top items get printed when inspecting label and prediction distributions
# show precision/recall/f1 for the top X contending labels based on the TASK variable
if TASK == "top":
    PRF_TOPK = int(os.environ.get("PRF_TOPK", "1"))
elif TASK == "region":
    PRF_TOPK = int(os.environ.get("PRF_TOPK", "2"))
else:
    PRF_TOPK = int(os.environ.get("PRF_TOPK", "5"))

# Unknown location labels
TOP_UNKNOWN = "UNKNOWN"
STATE_UNKNOWN = "UNKNOWN"
REGION_UNKNOWN = "UNKNOWN"

# confirm FEATURE_SET, TASK and MODEL_CLASS_WEIGHT are properly defined for training
if FEATURE_SET not in {"words", "struct", "combined"}:
    raise ValueError("FEATURE_SET must be one of: words, struct, combined")
if TASK not in {"top", "state", "region"}:
    raise ValueError("TASK must be one of: top, state, region")
if MODEL_CLASS_WEIGHT not in {"balanced", "none"}:
    raise ValueError("MODEL_CLASS_WEIGHT must be one of: balanced, none")

### Utilities

def _safe_div(num: float, den: float) -> float:
    return (num / den) if den else 0.0

def _as_object_array(values: Sequence[str] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=object)
    if arr.ndim != 1:
        raise RuntimeError(f"Expected a 1D label array, got shape={arr.shape}")
    return arr

def _counter_from_labels(values: Sequence[str] | np.ndarray) -> Counter:
    arr = _as_object_array(values)
    return Counter(arr.tolist())

def _subset_rows(X, idx: np.ndarray):
    return X[idx]

## Validate preprocessed data

def artifact_tag(word_feature_src: str) -> str:
    return f"src-{word_feature_src}"

def get_preprocess_artifact_paths(preprocess_dir: str, word_feature_src: str) -> Dict[str, str]:
    tag = artifact_tag(word_feature_src)
    base = Path(preprocess_dir).resolve()
    return {
        "users": str(base / f"users__{tag}.npy"),
        "X_words": str(base / f"X_words__{tag}.npz"),
        "X_words_masked": str(base / f"X_words_masked__{tag}.npz"),
        "X_struct": str(base / f"X_struct__{tag}.npz"),
        "labels_and_splits": str(base / f"labels_and_splits__{tag}.npz"),
        "metadata": str(base / f"metadata__{tag}.json"),
        "preprocessor": str(base / f"preprocessor__{tag}.pkl"),
        "tag": tag,
    }

def verify_preprocess_artifacts(preprocess_dir: str, feature_set: str, word_feature_src: str) -> Dict[str, str]:
    preprocess_path = Path(preprocess_dir).resolve()
    expected = get_preprocess_artifact_paths(str(preprocess_path), word_feature_src)

    print(f"[load] preprocess_dir={preprocess_path}")
    print(f"[load] tag={expected['tag']}")

    available_entries = {}
    available_names = []
    if preprocess_path.exists() and preprocess_path.is_dir():
        for p in preprocess_path.iterdir():
            available_entries[p.name] = p
            available_names.append(p.name)

    resolved = dict(expected)
    for key in ["users", "X_words", "X_words_masked", "X_struct", "labels_and_splits", "metadata", "preprocessor"]:
        candidate = Path(expected[key])
        if candidate.name in available_entries:
            resolved[key] = str(available_entries[candidate.name])
        print(f"[load] {key}: {resolved[key]} exists={candidate.name in available_entries}")

    required = ["users", "labels_and_splits"]
    if feature_set == "words":
        required.append("X_words")
    elif feature_set == "struct":
        required.append("X_struct")
    else:
        required.extend(["X_words", "X_struct"])

    missing = [name for name in required if Path(resolved[name]).name not in available_entries]
    if missing:
        raise FileNotFoundError(
            f"Missing required preprocessing artifacts in {preprocess_path} "
            f"for tag {resolved['tag']}: {', '.join(missing)}. Available files: {sorted(available_names)}"
        )

    return resolved

### Diagnostics / metrics

@dataclass
class SplitMetrics:
    n: int
    top1_acc: float
    top5_acc: float
    top10_acc: float
    mrr: float
    log_loss: float
    top1_precision_micro: float
    top1_recall_micro: float
    top1_f1_micro: float
    top1_precision_macro: float
    top1_recall_macro: float
    top1_f1_macro: float
    topk_precision_micro: float
    topk_recall_micro: float
    topk_f1_micro: float
    topk_precision_macro: float
    topk_recall_macro: float
    topk_f1_macro: float

def _prf_from_counts(tp: Dict[str, int], fp: Dict[str, int], fn: Dict[str, int]) -> Dict[str, float]:
    labels = set(tp) | set(fp) | set(fn)
    tp_sum = sum(tp.values())
    fp_sum = sum(fp.values())
    fn_sum = sum(fn.values())
    p_micro = _safe_div(tp_sum, tp_sum + fp_sum)
    r_micro = _safe_div(tp_sum, tp_sum + fn_sum)
    f1_micro = _safe_div(2 * p_micro * r_micro, p_micro + r_micro)

    p_list = []
    r_list = []
    f1_list = []
    for y in labels:
        t = tp.get(y, 0)
        f_p = fp.get(y, 0)
        f_n = fn.get(y, 0)
        p = _safe_div(t, t + f_p)
        r = _safe_div(t, t + f_n)
        f1 = _safe_div(2 * p * r, p + r)
        p_list.append(p)
        r_list.append(r)
        f1_list.append(f1)

    p_macro = sum(p_list) / len(p_list) if p_list else 0.0
    r_macro = sum(r_list) / len(r_list) if r_list else 0.0
    f1_macro = sum(f1_list) / len(f1_list) if f1_list else 0.0

    return {
        "precision_micro": p_micro,
        "recall_micro": r_micro,
        "f1_micro": f1_micro,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
    }


def summarize_split(name: str, labels: Sequence[str], topn: int = 15):
    c = _counter_from_labels(labels)
    log(f"\n{name} split:", 1)
    log(f"  users: {len(labels):,}", 1)
    log(f"  labels: {len(c):,}", 1)
    for lab, n in c.most_common(min(topn, len(c))):
        log(f"    {lab}: {n:,}", 1)


def print_label_count_diagnostics(train_labels: Sequence[str], topn: int = 15):
    c = _counter_from_labels(train_labels)
    counts = sorted(c.values())
    if not counts:
        log("[diag] no train labels found", 1)
        return

    def q(p: float) -> int:
        idx = int(p * (len(counts) - 1))
        return counts[idx]

    log("\n[diag] train label-count diagnostics:", 1)
    log(f"  total labels: {len(c):,}", 1)
    log(f"  min users/label: {counts[0]:,}", 1)
    log(f"  median users/label: {q(0.50):,}", 1)
    log(f"  p90 users/label: {q(0.90):,}", 1)
    log(f"  p95 users/label: {q(0.95):,}", 1)
    log(f"  p99 users/label: {q(0.99):,}", 1)
    log(f"  max users/label: {counts[-1]:,}", 1)
    log(f"  labels with 1 user: {sum(v == 1 for v in counts):,}", 1)
    log(f"  labels with <=2 users: {sum(v <= 2 for v in counts):,}", 1)
    log(f"  labels with <=5 users: {sum(v <= 5 for v in counts):,}", 1)
    log(f"  top {min(topn, len(c))} labels by train users:", 1)
    for lab, n in c.most_common(min(topn, len(c))):
        log(f"    {lab}: {n:,}", 1)


def print_majority_baseline(train_labels: Sequence[str], eval_labels: Sequence[str], split_name: str):
    train_arr = _as_object_array(train_labels)
    eval_arr = _as_object_array(eval_labels)

    if train_arr.size == 0 or eval_arr.size == 0:
        return

    c = Counter(train_arr.tolist())
    majority_label, majority_n = c.most_common(1)[0]
    acc = float(np.mean(eval_arr == majority_label))
    log(
        f"[baseline:{split_name}] majority_label={majority_label} train_count={majority_n:,} acc={acc:.4f}",
        1,
    )


def print_prediction_distribution(split_name: str, y_pred: Sequence[str], topn: int = 15):
    c = _counter_from_labels(y_pred)
    log(f"[{split_name}] top predicted labels:", 1)
    for lab, cnt in c.most_common(min(topn, len(c))):
        log(f"  {lab}: {cnt:,}", 1)


def _topk_labels(proba: np.ndarray, classes: Sequence[str], k: int) -> List[List[str]]:
    if proba.size == 0:
        return []
    k = min(k, proba.shape[1])
    idx = np.argpartition(-proba, kth=k - 1, axis=1)[:, :k]
    row_vals = np.take_along_axis(proba, idx, axis=1)
    order = np.argsort(-row_vals, axis=1)
    sorted_idx = np.take_along_axis(idx, order, axis=1)
    class_arr = np.asarray(classes, dtype=object)
    return [class_arr[row].tolist() for row in sorted_idx]



def validate_prediction_inputs(split_name: str, y_true: np.ndarray, proba: np.ndarray, classes: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = _as_object_array(y_true)
    proba = np.asarray(proba, dtype=np.float64)
    classes_arr = _as_object_array(classes)

    if y_true.size == 0:
        return y_true, proba, classes_arr

    if proba.ndim != 2:
        raise RuntimeError(f"[{split_name}] predict_proba output must be 2D, got shape={proba.shape}")
    if proba.shape[0] != y_true.size:
        raise RuntimeError(
            f"[{split_name}] prediction/label mismatch: proba rows={proba.shape[0]:,} labels={y_true.size:,}"
        )
    if proba.shape[1] != classes_arr.size:
        raise RuntimeError(
            f"[{split_name}] class/proba mismatch: proba cols={proba.shape[1]:,} classes={classes_arr.size:,}"
        )
    if classes_arr.size == 0:
        raise RuntimeError(f"[{split_name}] no model classes available")
    if len(set(classes_arr.tolist())) != classes_arr.size:
        raise RuntimeError(f"[{split_name}] duplicate class labels detected in model.classes_")
    if not np.isfinite(proba).all():
        raise RuntimeError(f"[{split_name}] predict_proba contains NaN or inf values")
    if np.any(proba < -1e-12) or np.any(proba > 1.0 + 1e-12):
        raise RuntimeError(f"[{split_name}] predict_proba contains values outside [0, 1]")
    row_sums = proba.sum(axis=1)
    if not np.all(np.isfinite(row_sums)):
        raise RuntimeError(f"[{split_name}] probability row sums contain NaN or inf values")
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise RuntimeError(
            f"[{split_name}] probability rows do not sum to 1 within tolerance; "
            f"min_sum={row_sums.min():.6f} max_sum={row_sums.max():.6f}"
        )

    unknown_labels = sorted(set(y_true.tolist()) - set(classes_arr.tolist()))
    if unknown_labels:
        preview = ", ".join(map(str, unknown_labels[:10]))
        suffix = " ..." if len(unknown_labels) > 10 else ""
        raise RuntimeError(
            f"[{split_name}] found labels absent from model.classes_: {preview}{suffix}"
        )

    return y_true, proba, classes_arr

def evaluate_predictions(split_name: str, y_true: np.ndarray, proba: np.ndarray, classes: Sequence[str]) -> SplitMetrics:
    y_true, proba, classes_arr = validate_prediction_inputs(split_name, y_true, proba, classes)
    n = int(len(y_true))
    if n == 0:
        return SplitMetrics(0, 0.0, 0.0, 0.0, 0.0, float("inf"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    top10_labels = _topk_labels(proba, classes_arr, 10)
    y_pred = np.asarray([labs[0] for labs in top10_labels], dtype=object)
    print_prediction_distribution(split_name, y_pred, topn=DEBUG_TOP_N)

    hit1 = 0
    hit5 = 0
    hit10 = 0
    rr_sum = 0.0

    tp1: Dict[str, int] = {}
    fp1: Dict[str, int] = {}
    fn1: Dict[str, int] = {}
    tpk: Dict[str, int] = {}
    fpk: Dict[str, int] = {}
    fnk: Dict[str, int] = {}

    for true_lab, pred1, predk in zip(y_true, y_pred, top10_labels):
        if pred1 == true_lab:
            hit1 += 1
            tp1[true_lab] = tp1.get(true_lab, 0) + 1
        else:
            fp1[pred1] = fp1.get(pred1, 0) + 1
            fn1[true_lab] = fn1.get(true_lab, 0) + 1

        if true_lab in predk[:5]:
            hit5 += 1
        if true_lab in predk[:10]:
            hit10 += 1

        predk_set = set(predk[:PRF_TOPK])
        if true_lab in predk_set:
            tpk[true_lab] = tpk.get(true_lab, 0) + 1
        else:
            fnk[true_lab] = fnk.get(true_lab, 0) + 1
        for lab in predk_set:
            if lab != true_lab:
                fpk[lab] = fpk.get(lab, 0) + 1

        try:
            rank = predk.index(true_lab) + 1
            rr_sum += 1.0 / rank
        except ValueError:
            pass

    prf1 = _prf_from_counts(tp1, fp1, fn1)
    prfk = _prf_from_counts(tpk, fpk, fnk)
    ll = float(log_loss(y_true, proba, labels=list(classes_arr)))

    return SplitMetrics(
        n=n,
        top1_acc=hit1 / n,
        top5_acc=hit5 / n,
        top10_acc=hit10 / n,
        mrr=rr_sum / n,
        log_loss=ll,
        top1_precision_micro=prf1["precision_micro"],
        top1_recall_micro=prf1["recall_micro"],
        top1_f1_micro=prf1["f1_micro"],
        top1_precision_macro=prf1["precision_macro"],
        top1_recall_macro=prf1["recall_macro"],
        top1_f1_macro=prf1["f1_macro"],
        topk_precision_micro=prfk["precision_micro"],
        topk_recall_micro=prfk["recall_micro"],
        topk_f1_micro=prfk["f1_micro"],
        topk_precision_macro=prfk["precision_macro"],
        topk_recall_macro=prfk["recall_macro"],
        topk_f1_macro=prfk["f1_macro"],
    )

### Data loading / selection

@dataclass
class TaskData:
    X_train: object
    y_train: np.ndarray
    X_val: object
    y_val: np.ndarray
    X_test: object
    y_test: np.ndarray
    users_train: np.ndarray
    users_val: np.ndarray
    users_test: np.ndarray
    task_name: str
    feature_set_name: str

@dataclass
class SavedModel:
    feature_set: str
    task: str
    classes_: List[str]
    model: LogisticRegression
    metadata: Dict[str, object]

@dataclass
class PreprocessIntrospection:
    selected_words: Optional[List[str]]
    struct_feature_names: Optional[List[str]]
    n_sub_features: Optional[int]
    n_hour_features: Optional[int]

def load_preprocess_introspection(
    preprocess_dir: str,
    artifact_paths: Optional[Dict[str, str]] = None,
) -> PreprocessIntrospection:
    artifact_paths = artifact_paths or {}
    preprocessor_path = artifact_paths.get("preprocessor", os.path.join(preprocess_dir, "preprocessor.pkl"))
    if not os.path.exists(preprocessor_path):
        log("[load] preprocessor artifact not found; feature-family breakdown logs will be limited", 1)
        return PreprocessIntrospection(None, None, None, None)

    try:
        with open(preprocessor_path, "rb") as f:
            pre = pickle.load(f)
    except Exception as e:
        log(f"[load] failed to read preprocessor artifact: {e}", 1)
        return PreprocessIntrospection(None, None, None, None)

    selected_words = pre.get("selected_words") if isinstance(pre, dict) else None
    struct_vectorizer = pre.get("struct_vectorizer") if isinstance(pre, dict) else None

    struct_feature_names = None
    n_sub_features = None
    n_hour_features = None
    if struct_vectorizer is not None:
        names = getattr(struct_vectorizer, "feature_names_", None)
        if names is None:
            try:
                names = list(struct_vectorizer.get_feature_names_out())
            except Exception:
                names = None
        if names is not None:
            struct_feature_names = list(names)
            n_sub_features = sum(1 for name in struct_feature_names if str(name).startswith("s:"))
            n_hour_features = sum(1 for name in struct_feature_names if str(name).startswith("h:"))

    return PreprocessIntrospection(selected_words, struct_feature_names, n_sub_features, n_hour_features)

def _density_stats(X) -> Tuple[float, float]:
    rows, cols = X.shape
    if rows <= 0 or cols <= 0:
        return 0.0, 0.0
    density = float(X.nnz) / float(rows * cols)
    avg_nnz_per_row = float(X.nnz) / float(rows)
    return density, avg_nnz_per_row


def log_matrix_density(name: str, X) -> None:
    density, avg_nnz_per_row = _density_stats(X)
    log(
        f"[density] {name}: shape={X.shape} nnz={X.nnz:,} density={density:.8f} avg_nnz_per_row={avg_nnz_per_row:.2f}",
        1,
    )

def log_preprocessed_feature_diagnostics(
    preprocess_dir: str,
    feature_set: str,
    artifact_paths: Optional[Dict[str, str]] = None,
) -> None:
    artifact_paths = artifact_paths or {}
    info = load_preprocess_introspection(preprocess_dir, artifact_paths)

    words_path = artifact_paths.get("X_words", os.path.join(preprocess_dir, "X_words.npz"))
    struct_path = artifact_paths.get("X_struct", os.path.join(preprocess_dir, "X_struct.npz"))

    if feature_set in {"words", "combined"} and os.path.exists(words_path):
        X_words = load_npz(words_path).tocsr()
        log_matrix_density("words", X_words)
        if info.selected_words is not None:
            log(f"[words] selected vocabulary size={len(info.selected_words):,}", 1)

    if feature_set in {"struct", "combined"} and os.path.exists(struct_path):
        X_struct = load_npz(struct_path).tocsr()
        log_matrix_density("struct(all)", X_struct)

        if info.struct_feature_names is not None:
            log(
                f"[struct] feature breakdown: subreddits={info.n_sub_features or 0:,} hours={info.n_hour_features or 0:,}",
                1,
            )
            sub_cols = [i for i, name in enumerate(info.struct_feature_names) if str(name).startswith("s:")]
            hour_cols = [i for i, name in enumerate(info.struct_feature_names) if str(name).startswith("h:")]

            if sub_cols:
                X_sub = X_struct[:, sub_cols]
                log_matrix_density("struct(subreddits)", X_sub)

            if hour_cols:
                X_hour = X_struct[:, hour_cols]
                log_matrix_density("struct(hours)", X_hour)
        else:
            log("[struct] preprocessor introspection unavailable; family-level breakdown skipped", 1)

def _load_labels_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        out = {k: data[k] for k in data.files}
    required = {"y_top", "y_state", "y_region", "mask_state", "mask_region", "train_idx", "val_idx", "test_idx"}
    missing = sorted(required - set(out))
    if missing:
        raise RuntimeError(f"labels_and_splits missing required arrays: {', '.join(missing)}")
    return out

def _load_feature_matrix(preprocess_dir: str, feature_set: str, artifact_paths: Dict[str, str]):
    words_path = artifact_paths["X_words"]
    struct_path = artifact_paths["X_struct"]
    if feature_set == "words":
        return load_npz(words_path).tocsr()
    if feature_set == "struct":
        return load_npz(struct_path).tocsr()
    if feature_set == "combined":
        return hstack([load_npz(words_path).tocsr(), load_npz(struct_path).tocsr()], format="csr")
    raise ValueError(feature_set)

def _load_masked_eval_feature_matrix(preprocess_dir: str, feature_set: str, artifact_paths: Dict[str, str]):
    words_masked_path = artifact_paths.get("X_words_masked")
    struct_path = artifact_paths["X_struct"]
    if not words_masked_path or not os.path.exists(words_masked_path):
        return None
    if feature_set == "words":
        return load_npz(words_masked_path).tocsr()
    if feature_set == "combined":
        return hstack([load_npz(words_masked_path).tocsr(), load_npz(struct_path).tocsr()], format="csr")
    return None

def _select_task_arrays(labels: Dict[str, np.ndarray], task: str) -> Tuple[np.ndarray, np.ndarray]:
    if task == "top":
        y = labels["y_top"].astype(object)
        mask = y != TOP_UNKNOWN
        return y, mask
    if task == "state":
        return labels["y_state"].astype(object), labels["mask_state"].astype(bool)
    if task == "region":
        return labels["y_region"].astype(object), labels["mask_region"].astype(bool)
    raise ValueError(task)

def load_task_data(preprocess_dir: str, feature_set: str, task: str, artifact_paths: Dict[str, str]) -> TaskData:
    
    users = np.load(artifact_paths["users"], allow_pickle=True)
    labels = _load_labels_npz(artifact_paths["labels_and_splits"])
    X = _load_feature_matrix(preprocess_dir, feature_set, artifact_paths)

    if X.shape[0] != len(users):
        raise RuntimeError(f"Row mismatch: X rows={X.shape[0]:,} users={len(users):,}")

    y_all, eligible_mask = _select_task_arrays(labels, task)

    train_idx = labels["train_idx"].astype(np.int64)
    val_idx = labels["val_idx"].astype(np.int64)
    test_idx = labels["test_idx"].astype(np.int64)

    train_keep = train_idx[eligible_mask[train_idx]]
    val_keep = val_idx[eligible_mask[val_idx]]
    test_keep = test_idx[eligible_mask[test_idx]]

    X_train = _subset_rows(X, train_keep)
    y_train = y_all[train_keep]
    X_val = _subset_rows(X, val_keep)
    y_val = y_all[val_keep]
    X_test = _subset_rows(X, test_keep)
    y_test = y_all[test_keep]

    if TRAIN_EVAL_LIMIT > 0 and X_train.shape[0] > TRAIN_EVAL_LIMIT:
        users_train_eval = users[train_keep][:TRAIN_EVAL_LIMIT]
    else:
        users_train_eval = users[train_keep]

    return TaskData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        users_train=users_train_eval,
        users_val=users[val_keep],
        users_test=users[test_keep],
        task_name=task,
        feature_set_name=feature_set,
    )

def load_masked_eval_splits(preprocess_dir: str, feature_set: str, task: str, artifact_paths: Dict[str, str]):
    X_masked = _load_masked_eval_feature_matrix(preprocess_dir, feature_set, artifact_paths)
    if X_masked is None:
        return None

    users = np.load(artifact_paths["users"], allow_pickle=True)
    labels = _load_labels_npz(artifact_paths["labels_and_splits"])
    if X_masked.shape[0] != len(users):
        raise RuntimeError(f"Row mismatch for masked eval: X rows={X_masked.shape[0]:,} users={len(users):,}")

    y_all, eligible_mask = _select_task_arrays(labels, task)
    val_idx = labels["val_idx"].astype(np.int64)
    test_idx = labels["test_idx"].astype(np.int64)

    val_keep = val_idx[eligible_mask[val_idx]]
    test_keep = test_idx[eligible_mask[test_idx]]

    return {
        "X_val": _subset_rows(X_masked, val_keep),
        "y_val": y_all[val_keep],
        "users_val": users[val_keep],
        "X_test": _subset_rows(X_masked, test_keep),
        "y_test": y_all[test_keep],
        "users_test": users[test_keep],
    }

### Training / saving

def build_model() -> LogisticRegression:
    class_weight = None if MODEL_CLASS_WEIGHT == "none" else MODEL_CLASS_WEIGHT
    return LogisticRegression(
        penalty="l2",
        C=MODEL_C,
        solver=MODEL_SOLVER,
        max_iter=MODEL_MAX_ITER,
        tol=MODEL_TOL,
        n_jobs=MODEL_N_JOBS,
        class_weight=class_weight,
        multi_class="multinomial",
        random_state=MODEL_RANDOM_STATE,
        verbose=0,
    )


def save_model(model: SavedModel, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"[saved] model -> {path}", 1)

# Main Execution

def main():
    
    # logging
    t0_all = time.time()
    log(f"[config] preprocess_dir={Path(PREPROC_PATH).resolve()}", 1)
    log(f"[config] output_dir={MODEL_DIR}", 1)
    log(f"[config] feature_set={FEATURE_SET} task={TASK}", 1)
    log(
        f"[config] solver={MODEL_SOLVER} C={MODEL_C} max_iter={MODEL_MAX_ITER} tol={MODEL_TOL} class_weight={MODEL_CLASS_WEIGHT}",
        1,
    )
    log(f"[config] word_feature_src={resolve_word_feature_src(type_)}", 1)
    
    # locate saved preprocessed artifacts
    word_feature_src = resolve_word_feature_src(type_)
    artifact_paths = verify_preprocess_artifacts(PREPROC_PATH, FEATURE_SET, word_feature_src)

    log(f"[load] feature set requested: {FEATURE_SET}", 1)
    log(f"[load] task requested: {TASK}", 1)
    log(f"[load] artifact tag: {artifact_tag(word_feature_src)}", 1)
    masked_eval_available = bool(artifact_paths.get("X_words_masked") and os.path.exists(artifact_paths["X_words_masked"])) and FEATURE_SET in {"words", "combined"}
    log(f"[load] masked word eval artifact present: {masked_eval_available}", 1)
    log_preprocessed_feature_diagnostics(PREPROC_PATH, FEATURE_SET, artifact_paths)

    # loading and overviewing task data
    data = load_task_data(PREPROC_PATH, FEATURE_SET, TASK, artifact_paths)
    masked_eval = load_masked_eval_splits(PREPROC_PATH, FEATURE_SET, TASK, artifact_paths) if masked_eval_available else None
    summarize_split("Train", data.y_train, topn=DEBUG_TOP_N)
    summarize_split("Valid", data.y_val, topn=DEBUG_TOP_N)
    summarize_split("Test", data.y_test, topn=DEBUG_TOP_N)
    print_label_count_diagnostics(data.y_train, topn=DEBUG_TOP_N)
    print_majority_baseline(data.y_train, data.y_train, "train")
    print_majority_baseline(data.y_train, data.y_val, "valid")
    print_majority_baseline(data.y_train, data.y_test, "test")

    log(
        f"[data] X_train={data.X_train.shape} X_val={data.X_val.shape} X_test={data.X_test.shape}",
        1,
    )

    # Model training
    model = build_model()
    t0 = time.time()
    model.fit(data.X_train, data.y_train)
    log(f"[train] fit elapsed {(time.time() - t0)/60:.2f} min", 1)
    log(f"[train] learned classes: {len(model.classes_):,}", 1)

    t0 = time.time()
    train_eval_X = data.X_train[: len(data.users_train)]
    train_eval_y = data.y_train[: len(data.users_train)]
    trm = evaluate_predictions("train", train_eval_y, model.predict_proba(train_eval_X), model.classes_)
    log(
        f"[train] n={trm.n:,} top1={trm.top1_acc:.4f} top5={trm.top5_acc:.4f} top10={trm.top10_acc:.4f} "
        f"mrr={trm.mrr:.4f} logloss={trm.log_loss:.4f} top1_f1_macro={trm.top1_f1_macro:.4f} "
        f"(elapsed {(time.time() - t0)/60:.2f} min)",
        1,
    )

    # valid set evaluation
    t0 = time.time()
    vm = evaluate_predictions("valid", data.y_val, model.predict_proba(data.X_val), model.classes_)
    log(
        f"[valid] n={vm.n:,} top1={vm.top1_acc:.4f} top5={vm.top5_acc:.4f} top10={vm.top10_acc:.4f} "
        f"mrr={vm.mrr:.4f} logloss={vm.log_loss:.4f} top1_f1_macro={vm.top1_f1_macro:.4f} "
        f"(elapsed {(time.time() - t0)/60:.2f} min)",
        1,
    )

    # masked validation evaluation
    if masked_eval is not None:
        t0 = time.time()
        vmm = evaluate_predictions("valid_masked", masked_eval["y_val"], model.predict_proba(masked_eval["X_val"]), model.classes_)
        log(
            f"[valid_masked] n={vmm.n:,} top1={vmm.top1_acc:.4f} top5={vmm.top5_acc:.4f} top10={vmm.top10_acc:.4f} "
            f"mrr={vmm.mrr:.4f} logloss={vmm.log_loss:.4f} top1_f1_macro={vmm.top1_f1_macro:.4f} "
            f"(elapsed {(time.time() - t0)/60:.2f} min)",
            1,
        )
    else:
        vmm = None
        log("[valid_masked] skipped: masked word artifact not available for this feature set", 1)

    # test evaluation
    t0 = time.time()
    tm = evaluate_predictions("test", data.y_test, model.predict_proba(data.X_test), model.classes_)
    log(
        f"\n[test] n={tm.n:,} hit@1={tm.top1_acc:.4f} hit@5={tm.top5_acc:.4f} hit@10={tm.top10_acc:.4f} "
        f"mrr={tm.mrr:.4f} logloss={tm.log_loss:.4f}\n"
        f"       top1 micro P/R/F1={tm.top1_precision_micro:.4f}/{tm.top1_recall_micro:.4f}/{tm.top1_f1_micro:.4f} "
        f"| macro P/R/F1={tm.top1_precision_macro:.4f}/{tm.top1_recall_macro:.4f}/{tm.top1_f1_macro:.4f}\n"
        f"       top{PRF_TOPK} micro P/R/F1={tm.topk_precision_micro:.4f}/{tm.topk_recall_micro:.4f}/{tm.topk_f1_micro:.4f} "
        f"| macro P/R/F1={tm.topk_precision_macro:.4f}/{tm.topk_recall_macro:.4f}/{tm.topk_f1_macro:.4f}\n"
        f"       elapsed {(time.time() - t0)/60:.2f} min",
        1,
    )

    # masked test evaluation
    if masked_eval is not None:
        t0 = time.time()
        tmm = evaluate_predictions("test_masked", masked_eval["y_test"], model.predict_proba(masked_eval["X_test"]), model.classes_)
        log(
            f"\n[test_masked] n={tmm.n:,} hit@1={tmm.top1_acc:.4f} hit@5={tmm.top5_acc:.4f} hit@10={tmm.top10_acc:.4f} "
            f"mrr={tmm.mrr:.4f} logloss={tmm.log_loss:.4f}\n"
            f"       top1 micro P/R/F1={tmm.top1_precision_micro:.4f}/{tmm.top1_recall_micro:.4f}/{tmm.top1_f1_micro:.4f} "
            f"| macro P/R/F1={tmm.top1_precision_macro:.4f}/{tmm.top1_recall_macro:.4f}/{tmm.top1_f1_macro:.4f}\n"
            f"       top{PRF_TOPK} micro P/R/F1={tmm.topk_precision_micro:.4f}/{tmm.topk_recall_micro:.4f}/{tmm.topk_f1_micro:.4f} "
            f"| macro P/R/F1={tmm.topk_precision_macro:.4f}/{tmm.topk_recall_macro:.4f}/{tmm.topk_f1_macro:.4f}\n"
            f"       elapsed {(time.time() - t0)/60:.2f} min",
            1,
        )
    else:
        tmm = None
        log("[test_masked] skipped: masked word artifact not available for this feature set", 1)

    # save metadata
    metadata = {
        "feature_set": FEATURE_SET,
        "task": TASK,
        "solver": MODEL_SOLVER,
        "C": MODEL_C,
        "max_iter": MODEL_MAX_ITER,
        "tol": MODEL_TOL,
        "n_jobs": MODEL_N_JOBS,
        "class_weight": MODEL_CLASS_WEIGHT,
        "random_state": MODEL_RANDOM_STATE,
        "train_metrics": asdict(trm),
        "valid_metrics": asdict(vm),
        "valid_masked_metrics": asdict(vmm) if vmm is not None else None,
        "test_metrics": asdict(tm),
        "test_masked_metrics": asdict(tmm) if tmm is not None else None,
        "n_train": int(data.X_train.shape[0]),
        "n_val": int(data.X_val.shape[0]),
        "n_test": int(data.X_test.shape[0]),
        "n_features": int(data.X_train.shape[1]),
        "n_classes": int(len(model.classes_)),
    }

    # save model
    tag = artifact_paths["tag"]
    model_stub = f"lr__{FEATURE_SET}__{TASK}__{tag}"
    with open(os.path.join(MODEL_DIR, f"{model_stub}__metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    if SAVE_MODEL:
        model_path = os.path.join(MODEL_DIR, f"{model_stub}.pkl")
        save_model(
            SavedModel(
                feature_set=FEATURE_SET,
                task=TASK,
                classes_=list(model.classes_),
                model=model,
                metadata=metadata,
            ),
            model_path,
        )

    log(f"\n[done] total elapsed {(time.time() - t0_all)/60:.2f} min", 1)

### Main Execution

if __name__ == "__main__":
    main()
