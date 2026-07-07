### Imports

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics import log_loss
from cli import get_args, MODELS_DIR
from utils import log_report

# Logging
VERBOSITY = int(os.environ.get("VERBOSITY", "1"))

def log(msg: str, level: int = 1, stream=None):
    if level <= VERBOSITY:
        print(msg, file=stream)

### Argument Handling

args = get_args()
type_ = args.type

### Paths Handling

# input path processing
if not args.input and not args.input_2:
    log_report(
        "No custom paths provided for preprocessed model features and pretrained models. Setting up default paths..."
    )
    PREPROC_DIR = os.path.join(MODELS_DIR, "label_location", "preprocessed_streaming")
    MODEL_DIR = os.path.join(MODELS_DIR, "label_location", "trained_lr")

# NOTE: for custom pathing, 'input' argument should point to the preprocessed feature sets folder, and 'input_2' to pretrained lr models folder.
elif args.input and args.input_2:
    PREPROC_DIR = args.input
    MODEL_DIR = args.input_2
else:
    raise ValueError(
        "Provide either both --input and --input_2, or neither. "
        "Supplying only one is not supported."
    )

# parse the output path
if not args.output:
    output_path = Path(MODEL_DIR)
else:
    output_path = Path(args.output)

os.makedirs(output_path, exist_ok=True)

### Location Model Mixture Hyperparameters

# task and features
WORD_FEATURE_SRC = os.environ.get("WORD_FEATURE_SRC", "all").strip().lower() # comments | submissions | all
TASK = os.environ.get("TASK", "state").strip().lower()  # top | state | region
WEIGHT_STRUCT = float(os.environ.get("WEIGHT_STRUCT", "0.20")) # matters if OPTIMIZE_WEIGHT_STRUCT == 0
WEIGHT_WORDS = float(os.environ.get("WEIGHT_WORDS", str(1.0 - WEIGHT_STRUCT))) # MATTERS if OPTIMIZE_WEIGHT_STRUCT == 0

# optimization of model weights
OPTIMIZE_WEIGHT_STRUCT = os.environ.get("OPTIMIZE_WEIGHT_STRUCT", "1") == "1"
WEIGHT_STRUCT_GRID = os.environ.get(
    "WEIGHT_STRUCT_GRID",
    "0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50",
).strip()
OPTIMIZATION_TARGET = os.environ.get("OPTIMIZATION_TARGET", "valid_masked_top1").strip().lower()

# reporting/saving
SAVE_METRICS = os.environ.get("SAVE_METRICS", "1") == "1"
DEBUG_TOP_N = int(os.environ.get("DEBUG_TOP_N", "15"))
PRF_TOPK = int(os.environ.get("PRF_TOPK", "5"))

# smoothing
TEMPERATURE = float(os.environ.get("TEMPERATURE", "1.25")) # to reduce overconfidence in the chosen label
ENABLE_GEO_SMOOTHING = os.environ.get("ENABLE_GEO_SMOOTHING", "0") == "1"
DISTANCE_RADII_KM: Tuple[int, ...] = (100, 300, 500, 1000)
GEO_SMOOTHING_SIGMA_KM = float(os.environ.get("GEO_SMOOTHING_SIGMA_KM", "400.0")) # Geosmoothing SD in kilometers

# hyperparameter checks
if TASK not in {"top", "state", "region"}:
    raise ValueError("TASK must be one of: top, state, region")
if WEIGHT_STRUCT < 0.0 or WEIGHT_WORDS < 0.0:
    raise ValueError("WEIGHT_STRUCT and WEIGHT_WORDS must be non-negative")
if (WEIGHT_STRUCT + WEIGHT_WORDS) <= 0.0:
    raise ValueError("WEIGHT_STRUCT + WEIGHT_WORDS must be > 0")
if GEO_SMOOTHING_SIGMA_KM <= 0.0:
    raise ValueError("GEO_SMOOTHING_SIGMA_KM must be > 0")
if TEMPERATURE <= 0.0:
    raise ValueError("TEMPERATURE must be > 0")

### Labels Mapping

TOP_UNKNOWN = "UNKNOWN"
STATE_UNKNOWN = "UNKNOWN"
REGION_UNKNOWN = "UNKNOWN"

# from US Census; to calculate the distance from the predicted label to the real label
STATE_CENTROIDS: Dict[str, Tuple[float, float]] = {
    "AL": (32.806671, -86.791130),
    "AK": (61.370716, -152.404419),
    "AZ": (33.729759, -111.431221),
    "AR": (34.969704, -92.373123),
    "CA": (36.116203, -119.681564),
    "CO": (39.059811, -105.311104),
    "CT": (41.597782, -72.755371),
    "DC": (38.9072, -77.0369),
    "DE": (39.318523, -75.507141),
    "FL": (27.766279, -81.686783),
    "GA": (33.040619, -83.643074),
    "HI": (21.094318, -157.498337),
    "IA": (42.011539, -93.210526),
    "ID": (44.240459, -114.478828),
    "IL": (40.349457, -88.986137),
    "IN": (39.849426, -86.258278),
    "KS": (38.5266, -96.726486),
    "KY": (37.66814, -84.670067),
    "LA": (31.169546, -91.867805),
    "MA": (42.230171, -71.530106),
    "MD": (39.063946, -76.802101),
    "ME": (44.693947, -69.381927),
    "MI": (43.326618, -84.536095),
    "MN": (45.694454, -93.900192),
    "MO": (38.456085, -92.288368),
    "MS": (32.741646, -89.678696),
    "MT": (46.921925, -110.454353),
    "NC": (35.630066, -79.806419),
    "ND": (47.528912, -99.784012),
    "NE": (41.12537, -98.268082),
    "NH": (43.452492, -71.563896),
    "NJ": (40.298904, -74.521011),
    "NM": (34.840515, -106.248482),
    "NV": (38.313515, -117.055374),
    "NY": (42.165726, -74.948051),
    "OH": (40.388783, -82.764915),
    "OK": (35.565342, -96.928917),
    "OR": (44.572021, -122.070938),
    "PA": (40.590752, -77.209755),
    "RI": (41.680893, -71.51178),
    "SC": (33.856892, -80.945007),
    "SD": (44.299782, -99.438828),
    "TN": (35.747845, -86.692345),
    "TX": (31.054487, -97.563461),
    "UT": (40.150032, -111.862434),
    "VA": (37.769337, -78.169968),
    "VT": (44.045876, -72.710686),
    "WA": (47.400902, -121.490494),
    "WI": (44.268543, -89.616508),
    "WV": (38.491226, -80.954453),
    "WY": (42.755966, -107.30249),
}

# from US census; to calculate region-level accuracy
STATE_TO_REGION: Dict[str, str] = {
    "CT": "NORTHEAST", "ME": "NORTHEAST", "MA": "NORTHEAST", "NH": "NORTHEAST",
    "RI": "NORTHEAST", "VT": "NORTHEAST", "NJ": "NORTHEAST", "NY": "NORTHEAST", "PA": "NORTHEAST",
    "IL": "MIDWEST", "IN": "MIDWEST", "MI": "MIDWEST", "OH": "MIDWEST", "WI": "MIDWEST",
    "IA": "MIDWEST", "KS": "MIDWEST", "MN": "MIDWEST", "MO": "MIDWEST", "NE": "MIDWEST",
    "ND": "MIDWEST", "SD": "MIDWEST",
    "DE": "SOUTH", "DC": "SOUTH", "FL": "SOUTH", "GA": "SOUTH", "MD": "SOUTH",
    "NC": "SOUTH", "SC": "SOUTH", "VA": "SOUTH", "WV": "SOUTH",
    "AL": "SOUTH", "KY": "SOUTH", "MS": "SOUTH", "TN": "SOUTH",
    "AR": "SOUTH", "LA": "SOUTH", "OK": "SOUTH", "TX": "SOUTH",
    "AZ": "WEST", "CO": "WEST", "ID": "WEST", "MT": "WEST", "NV": "WEST",
    "NM": "WEST", "UT": "WEST", "WY": "WEST",
    "AK": "WEST", "CA": "WEST", "HI": "WEST", "OR": "WEST", "WA": "WEST",
}

### Data Classes

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
    mean_distance_km: Optional[float] = None
    median_distance_km: Optional[float] = None
    top5_min_mean_distance_km: Optional[float] = None
    top5_min_median_distance_km: Optional[float] = None
    top10_min_mean_distance_km: Optional[float] = None
    top10_min_median_distance_km: Optional[float] = None
    within_100km_acc: Optional[float] = None
    within_300km_acc: Optional[float] = None
    within_500km_acc: Optional[float] = None
    within_1000km_acc: Optional[float] = None
    region_top1_acc: Optional[float] = None
    region_top5_acc: Optional[float] = None


@dataclass
class TaskData:
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    users_train: np.ndarray
    users_val: np.ndarray
    users_test: np.ndarray
    task_name: str


@dataclass
class SavedModel:
    feature_set: str
    task: str
    classes_: List[str]
    model: object
    metadata: Dict[str, object]

### Utilities 

def resolve_word_feature_src(type_arg: str) -> str:
    src = (type_arg or WORD_FEATURE_SRC).strip().lower()
    if src not in {"comments", "submissions", "all"}:
        raise ValueError("type / WORD_FEATURE_SRC must be one of: comments, submissions, all")
    return src


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


def get_model_paths(model_dir: str, task: str, tag: str) -> Dict[str, str]:
    base = Path(model_dir).resolve()
    return {
        "words_model": str(base / f"lr__words__{task}__{tag}.pkl"),
        "words_metrics": str(base / f"lr__words__{task}__{tag}__metrics.json"),
        "struct_model": str(base / f"lr__struct__{task}__{tag}.pkl"),
        "struct_metrics": str(base / f"lr__struct__{task}__{tag}__metrics.json"),
    }


def _as_object_array(values: Sequence[str] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=object)
    if arr.ndim != 1:
        raise RuntimeError(f"Expected a 1D label array, got shape={arr.shape}")
    return arr


def _counter_from_labels(values: Sequence[str] | np.ndarray) -> Counter:
    arr = _as_object_array(values)
    return Counter(arr.tolist())


def _safe_div(num: float, den: float) -> float:
    return (num / den) if den else 0.0

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


def print_prediction_distribution(split_name: str, y_pred: Sequence[str], topn: int = 15):
    c = _counter_from_labels(y_pred)
    log(f"[{split_name}] top predicted labels:", 1)
    for lab, cnt in c.most_common(min(topn, len(c))):
        log(f"  {lab}: {cnt:,}", 1)


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

def _is_state_task_with_geo(y_true: np.ndarray, y_pred: np.ndarray) -> bool:
    if TASK != "state":
        return False
    true_set = set(map(str, np.unique(y_true)))
    pred_set = set(map(str, np.unique(y_pred)))
    return bool(true_set) and true_set.issubset(set(STATE_CENTROIDS)) and pred_set.issubset(set(STATE_CENTROIDS))


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2.0) ** 2
    return float(2.0 * r * np.arcsin(np.sqrt(a)))


def _state_distance_km(state_a: str, state_b: str) -> float:
    lat1, lon1 = STATE_CENTROIDS[state_a]
    lat2, lon2 = STATE_CENTROIDS[state_b]
    return _haversine_km(lat1, lon1, lat2, lon2)

### Geographical Metrics Calculation

def compute_geographic_metrics(y_true: np.ndarray, y_pred: np.ndarray, top10_labels: List[List[str]]) -> Dict[str, Optional[float]]:
    if not _is_state_task_with_geo(y_true, y_pred):
        return {
            "mean_distance_km": None,
            "median_distance_km": None,
            "top5_min_mean_distance_km": None,
            "top5_min_median_distance_km": None,
            "top10_min_mean_distance_km": None,
            "top10_min_median_distance_km": None,
            "within_100km_acc": None,
            "within_300km_acc": None,
            "within_500km_acc": None,
            "within_1000km_acc": None,
            "region_top1_acc": None,
            "region_top5_acc": None,
        }

    top1_distances: List[float] = []
    top5_min_distances: List[float] = []
    top10_min_distances: List[float] = []
    within_counts = {radius: 0 for radius in DISTANCE_RADII_KM}
    region_top1_hits = 0
    region_top5_hits = 0

    for true_lab, pred1, predk in zip(map(str, y_true.tolist()), map(str, y_pred.tolist()), top10_labels):
        d1 = _state_distance_km(true_lab, pred1)
        top1_distances.append(d1)
        for radius in DISTANCE_RADII_KM:
            if d1 <= radius:
                within_counts[radius] += 1

        predk_str = [str(x) for x in predk]
        top5_min_distances.append(min(_state_distance_km(true_lab, lab) for lab in predk_str[:5]))
        top10_min_distances.append(min(_state_distance_km(true_lab, lab) for lab in predk_str[:10]))

        true_region = STATE_TO_REGION.get(true_lab)
        if true_region is not None:
            if STATE_TO_REGION.get(pred1) == true_region:
                region_top1_hits += 1
            if any(STATE_TO_REGION.get(lab) == true_region for lab in predk_str[:5]):
                region_top5_hits += 1

    n = max(len(top1_distances), 1)
    return {
        "mean_distance_km": float(np.mean(top1_distances)),
        "median_distance_km": float(np.median(top1_distances)),
        "top5_min_mean_distance_km": float(np.mean(top5_min_distances)),
        "top5_min_median_distance_km": float(np.median(top5_min_distances)),
        "top10_min_mean_distance_km": float(np.mean(top10_min_distances)),
        "top10_min_median_distance_km": float(np.median(top10_min_distances)),
        "within_100km_acc": float(within_counts[100] / n),
        "within_300km_acc": float(within_counts[300] / n),
        "within_500km_acc": float(within_counts[500] / n),
        "within_1000km_acc": float(within_counts[1000] / n),
        "region_top1_acc": float(region_top1_hits / n),
        "region_top5_acc": float(region_top5_hits / n),
    }


def print_geographic_diagnostics(split_name: str, geo: Dict[str, Optional[float]]) -> None:
    if geo.get("mean_distance_km") is None:
        return
    log(
        f"[{split_name}] geo distance: "
        f"mean={geo['mean_distance_km']:.1f} km median={geo['median_distance_km']:.1f} km",
        1,
    )
    log(
        f"[{split_name}] geo top-k min distance: "
        f"top5 mean/median={geo['top5_min_mean_distance_km']:.1f}/{geo['top5_min_median_distance_km']:.1f} km "
        f"| top10 mean/median={geo['top10_min_mean_distance_km']:.1f}/{geo['top10_min_median_distance_km']:.1f} km",
        1,
    )
    log(
        f"[{split_name}] geo within-radius acc: "
        f"<=100km={geo['within_100km_acc']:.4f} <=300km={geo['within_300km_acc']:.4f} "
        f"<=500km={geo['within_500km_acc']:.4f} <=1000km={geo['within_1000km_acc']:.4f}",
        1,
    )
    log(
        f"[{split_name}] geo regional correctness: "
        f"top1={geo['region_top1_acc']:.4f} top5_any={geo['region_top5_acc']:.4f}",
        1,
    )

### Smoothing Functions

def build_state_similarity_matrix(classes: Sequence[str], sigma_km: float) -> Optional[np.ndarray]:
    class_list = [str(c) for c in classes]
    if TASK != "state":
        return None
    if not class_list or any(c not in STATE_CENTROIDS for c in class_list):
        return None
    n = len(class_list)
    sim = np.zeros((n, n), dtype=np.float64)
    for i, ci in enumerate(class_list):
        for j, cj in enumerate(class_list):
            d = _state_distance_km(ci, cj)
            sim[i, j] = np.exp(-(d * d) / (2.0 * sigma_km * sigma_km))
    row_sums = sim.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(row_sums)) or np.any(row_sums <= 0.0):
        raise RuntimeError("Invalid state similarity matrix row sums")
    sim = sim / row_sums
    return sim


def maybe_apply_geo_smoothing(proba: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    if not ENABLE_GEO_SMOOTHING:
        return proba
    sim = build_state_similarity_matrix(classes, GEO_SMOOTHING_SIGMA_KM)
    if sim is None:
        return proba
    smoothed = np.asarray(proba, dtype=np.float64) @ sim
    row_sums = smoothed.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(row_sums)) or np.any(row_sums <= 0.0):
        raise RuntimeError("Encountered invalid row sums after geo smoothing")
    smoothed = smoothed / row_sums
    return smoothed



def apply_temperature_to_probabilities(proba: np.ndarray, temperature: float) -> np.ndarray:
    proba = np.asarray(proba, dtype=np.float64)
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")
    if proba.ndim != 2:
        raise RuntimeError(f"Expected 2D probabilities for temperature scaling, got shape={proba.shape}")
    if proba.size == 0:
        return proba.copy()

    clipped = np.clip(proba, 1e-12, 1.0)
    tempered = np.power(clipped, 1.0 / temperature)
    row_sums = tempered.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(row_sums)) or np.any(row_sums <= 0.0):
        raise RuntimeError("Encountered invalid row sums after temperature adjustment")
    return tempered / row_sums

### Evaluation Functions

def validate_prediction_inputs(split_name: str, y_true: np.ndarray, proba: np.ndarray, classes: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = _as_object_array(y_true)
    proba = np.asarray(proba, dtype=np.float64)
    classes_arr = _as_object_array(classes)

    if y_true.size == 0:
        return y_true, proba, classes_arr
    if proba.ndim != 2:
        raise RuntimeError(f"[{split_name}] predict_proba output must be 2D, got shape={proba.shape}")
    if proba.shape[0] != y_true.size:
        raise RuntimeError(f"[{split_name}] prediction/label mismatch: proba rows={proba.shape[0]:,} labels={y_true.size:,}")
    if proba.shape[1] != classes_arr.size:
        raise RuntimeError(f"[{split_name}] class/proba mismatch: proba cols={proba.shape[1]:,} classes={classes_arr.size:,}")
    if classes_arr.size == 0:
        raise RuntimeError(f"[{split_name}] no model classes available")
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
        raise RuntimeError(f"[{split_name}] found labels absent from ensemble classes: {preview}{suffix}")
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
    geo = compute_geographic_metrics(y_true, y_pred, top10_labels)
    print_geographic_diagnostics(split_name, geo)

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
        mean_distance_km=geo["mean_distance_km"],
        median_distance_km=geo["median_distance_km"],
        top5_min_mean_distance_km=geo["top5_min_mean_distance_km"],
        top5_min_median_distance_km=geo["top5_min_median_distance_km"],
        top10_min_mean_distance_km=geo["top10_min_mean_distance_km"],
        top10_min_median_distance_km=geo["top10_min_median_distance_km"],
        within_100km_acc=geo["within_100km_acc"],
        within_300km_acc=geo["within_300km_acc"],
        within_500km_acc=geo["within_500km_acc"],
        within_1000km_acc=geo["within_1000km_acc"],
        region_top1_acc=geo["region_top1_acc"],
        region_top5_acc=geo["region_top5_acc"],
    )

### Labels / splits / data

def _load_labels_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        out = {k: data[k] for k in data.files}
    required = {"y_top", "y_state", "y_region", "mask_state", "mask_region", "train_idx", "val_idx", "test_idx"}
    missing = sorted(required - set(out))
    if missing:
        raise RuntimeError(f"labels_and_splits missing required arrays: {', '.join(missing)}")
    return out


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


def load_task_data(task: str, artifact_paths: Dict[str, str]) -> TaskData:
    users = np.load(artifact_paths["users"], allow_pickle=True)
    labels = _load_labels_npz(artifact_paths["labels_and_splits"])
    y_all, eligible_mask = _select_task_arrays(labels, task)

    train_idx = labels["train_idx"].astype(np.int64)
    val_idx = labels["val_idx"].astype(np.int64)
    test_idx = labels["test_idx"].astype(np.int64)

    train_keep = train_idx[eligible_mask[train_idx]]
    val_keep = val_idx[eligible_mask[val_idx]]
    test_keep = test_idx[eligible_mask[test_idx]]

    return TaskData(
        y_train=y_all[train_keep],
        y_val=y_all[val_keep],
        y_test=y_all[test_keep],
        users_train=users[train_keep],
        users_val=users[val_keep],
        users_test=users[test_keep],
        task_name=task,
    )


def load_masked_eval_labels(task: str, artifact_paths: Dict[str, str]):
    if not os.path.exists(artifact_paths["X_words_masked"]):
        return None
    users = np.load(artifact_paths["users"], allow_pickle=True)
    labels = _load_labels_npz(artifact_paths["labels_and_splits"])
    y_all, eligible_mask = _select_task_arrays(labels, task)

    val_idx = labels["val_idx"].astype(np.int64)
    test_idx = labels["test_idx"].astype(np.int64)
    val_keep = val_idx[eligible_mask[val_idx]]
    test_keep = test_idx[eligible_mask[test_idx]]
    return {
        "y_val": y_all[val_keep],
        "users_val": users[val_keep],
        "y_test": y_all[test_keep],
        "users_test": users[test_keep],
    }


### Model loading / ensemble

def load_saved_model(path: str) -> SavedModel:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if hasattr(obj, "model") and hasattr(obj, "classes_"):
        return obj
    if isinstance(obj, dict) and "model" in obj and "classes_" in obj:
        return SavedModel(
            feature_set=obj.get("feature_set", "unknown"),
            task=obj.get("task", "unknown"),
            classes_=list(obj["classes_"]),
            model=obj["model"],
            metadata=obj.get("metadata", {}),
        )
    raise RuntimeError(f"Unsupported saved model format in {path}")


def _normalize_weights(words_weight: float, struct_weight: float) -> Tuple[float, float]:
    total = words_weight + struct_weight
    return words_weight / total, struct_weight / total


def align_and_blend_probabilities(
    proba_words: np.ndarray,
    classes_words: Sequence[str],
    proba_struct: np.ndarray,
    classes_struct: Sequence[str],
    words_weight: float,
    struct_weight: float,
) -> Tuple[np.ndarray, np.ndarray]:
    words_weight, struct_weight = _normalize_weights(words_weight, struct_weight)
    classes_union = sorted(set(map(str, classes_words)) | set(map(str, classes_struct)))
    class_to_idx = {c: i for i, c in enumerate(classes_union)}

    out_words = np.zeros((proba_words.shape[0], len(classes_union)), dtype=np.float64)
    out_struct = np.zeros((proba_struct.shape[0], len(classes_union)), dtype=np.float64)

    for j, c in enumerate(map(str, classes_words)):
        out_words[:, class_to_idx[c]] = proba_words[:, j]
    for j, c in enumerate(map(str, classes_struct)):
        out_struct[:, class_to_idx[c]] = proba_struct[:, j]

    blended = words_weight * out_words + struct_weight * out_struct
    row_sums = blended.sum(axis=1, keepdims=True)
    bad = ~np.isfinite(row_sums[:, 0]) | (row_sums[:, 0] <= 0.0)
    if np.any(bad):
        raise RuntimeError("Encountered non-finite or non-positive ensemble row sums")
    blended = blended / row_sums
    classes_arr = np.asarray(classes_union, dtype=object)
    blended = maybe_apply_geo_smoothing(blended, classes_arr)
    return blended, classes_arr


def load_feature_matrix(path: str):
    return load_npz(path).tocsr()


def parse_weight_grid(spec: str) -> List[float]:
    values: List[float] = []
    for part in (spec or "").split(","):
        token = part.strip()
        if not token:
            continue
        try:
            val = float(token)
        except Exception as e:
            raise ValueError(f"Invalid WEIGHT_STRUCT_GRID entry: {token}") from e
        if val < 0.0:
            raise ValueError("WEIGHT_STRUCT_GRID entries must be non-negative")
        values.append(val)
    if not values:
        raise ValueError("WEIGHT_STRUCT_GRID must contain at least one non-negative value")
    deduped = sorted(set(round(v, 10) for v in values))
    return deduped


def metric_value(metrics: SplitMetrics, target: str) -> float:
    if target.endswith("_top1"):
        return metrics.top1_acc
    if target.endswith("_macro_f1"):
        return metrics.top1_f1_macro
    if target.endswith("_logloss"):
        return -metrics.log_loss
    raise ValueError(f"Unsupported OPTIMIZATION_TARGET: {target}")


def choose_best_weight(
    candidate_results: List[Dict[str, object]],
    target: str,
) -> Dict[str, object]:
    if not candidate_results:
        raise RuntimeError("No candidate ensemble results were generated")

    def sort_key(rec: Dict[str, object]):
        valid = rec["valid_metrics"]
        valid_masked = rec.get("valid_masked_metrics")
        if target.startswith("valid_masked_"):
            if valid_masked is None:
                primary = float("-inf")
            else:
                primary = metric_value(valid_masked, target)
        elif target.startswith("valid_"):
            primary = metric_value(valid, target)
        else:
            raise ValueError(f"Unsupported OPTIMIZATION_TARGET: {target}")
        secondary = valid_masked.top1_acc if valid_masked is not None else valid.top1_acc
        tertiary = valid.top1_acc
        quaternary = -valid.log_loss
        return (primary, secondary, tertiary, quaternary, -abs(float(rec["w_struct"]) - 0.20))

    return max(candidate_results, key=sort_key)


### Main Execution

def main(argv: Optional[Sequence[str]] = None):

    # resolve paths and locate needed processed feature and trained model files

    t0_all = time.time()
    word_feature_src = resolve_word_feature_src(type_)
    artifact_paths = get_preprocess_artifact_paths(PREPROC_DIR, word_feature_src)
    tag = artifact_paths["tag"]
    model_paths = get_model_paths(MODEL_DIR, TASK, tag)

    log(f"[config] output_dir={Path(output_path).resolve()}", 1)
    log(f"[config] task={TASK} tag={tag}", 1)
    log(f"[config] words_weight (matters if not optimizing weights)={WEIGHT_WORDS:.4f} struct_weight={WEIGHT_STRUCT:.4f}", 1)
    log(f"[config] temperature={TEMPERATURE:.4f}", 1)
    log(f"[config] geo_smoothing_enabled={ENABLE_GEO_SMOOTHING} sigma_km={GEO_SMOOTHING_SIGMA_KM:.1f}", 1)

    log(f"[load] preprocess_dir={Path(PREPROC_DIR).resolve()}", 1)
    for key, path in artifact_paths.items():
        if key == "tag":
            continue
        log(f"[load] {key}: {path} exists={os.path.exists(path)}", 1)

    for key, path in model_paths.items():
        log(f"[load] {key}: {path} exists={os.path.exists(path)}", 1)
        if key.endswith("_model") and not os.path.exists(path):
            raise FileNotFoundError(f"Required model not found: {path}")

    words_saved = load_saved_model(model_paths["words_model"])
    struct_saved = load_saved_model(model_paths["struct_model"])
    if str(words_saved.task) != TASK or str(struct_saved.task) != TASK:
        raise RuntimeError("Loaded models do not match requested TASK")

    data = load_task_data(TASK, artifact_paths)
    masked_labels = load_masked_eval_labels(TASK, artifact_paths)

    # perform train/valid/test 80/10/10 split and print diagnostics

    summarize_split("Train", data.y_train, topn=DEBUG_TOP_N)
    summarize_split("Valid", data.y_val, topn=DEBUG_TOP_N)
    summarize_split("Test", data.y_test, topn=DEBUG_TOP_N)
    print_label_count_diagnostics(data.y_train, topn=DEBUG_TOP_N)
    print_majority_baseline(data.y_train, data.y_train, "train")
    print_majority_baseline(data.y_train, data.y_val, "valid")
    print_majority_baseline(data.y_train, data.y_test, "test")

    # load features

    X_words = load_feature_matrix(artifact_paths["X_words"])
    X_struct = load_feature_matrix(artifact_paths["X_struct"])
    log(f"[data] X_words={X_words.shape} X_struct={X_struct.shape}", 1)

    # load labels and masks

    labels = _load_labels_npz(artifact_paths["labels_and_splits"])
    y_all, eligible_mask = _select_task_arrays(labels, TASK)
    train_idx = labels["train_idx"].astype(np.int64)
    val_idx = labels["val_idx"].astype(np.int64)
    test_idx = labels["test_idx"].astype(np.int64)
    train_keep = train_idx[eligible_mask[train_idx]]
    val_keep = val_idx[eligible_mask[val_idx]]
    test_keep = test_idx[eligible_mask[test_idx]]

    X_words_train = X_words[train_keep]
    X_words_val = X_words[val_keep]
    X_words_test = X_words[test_keep]
    X_struct_train = X_struct[train_keep]
    X_struct_val = X_struct[val_keep]
    X_struct_test = X_struct[test_keep]

    X_words_test_masked = None
    if masked_labels is not None and os.path.exists(artifact_paths["X_words_masked"]):
        X_words_masked = load_feature_matrix(artifact_paths["X_words_masked"])
        X_words_val_masked = X_words_masked[val_keep]
        X_words_test_masked = X_words_masked[test_keep]
    else:
        X_words_val_masked = None

    # calculate label probabilities from the mixture logistic regression models

    t0_predict = time.time()
    proba_words_train = words_saved.model.predict_proba(X_words_train)
    proba_struct_train = struct_saved.model.predict_proba(X_struct_train)
    proba_words_val = words_saved.model.predict_proba(X_words_val)
    proba_struct_val = struct_saved.model.predict_proba(X_struct_val)
    proba_words_test = words_saved.model.predict_proba(X_words_test)
    proba_struct_test = struct_saved.model.predict_proba(X_struct_test)
    if X_words_val_masked is not None:
        proba_words_val_masked = words_saved.model.predict_proba(X_words_val_masked)
        proba_words_test_masked = words_saved.model.predict_proba(X_words_test_masked)
    else:
        proba_words_val_masked = None
        proba_words_test_masked = None
    log(f"[predict] base model probability generation elapsed {(time.time() - t0_predict)/60:.2f} min", 1)

    # model weight optimization grid search

    if OPTIMIZE_WEIGHT_STRUCT:
        candidate_weights = parse_weight_grid(WEIGHT_STRUCT_GRID)
    else:
        candidate_weights = [WEIGHT_STRUCT]
    log(f"[config] optimize_weight_struct={OPTIMIZE_WEIGHT_STRUCT} target={OPTIMIZATION_TARGET}", 1)
    log(f"[config] candidate_struct_weights={', '.join(f'{w:.3f}' for w in candidate_weights)}", 1)

    candidate_results: List[Dict[str, object]] = []
    t0_opt = time.time()
    for w_struct in candidate_weights:
        w_words = max(0.0, 1.0 - w_struct)
        proba_val, classes = align_and_blend_probabilities(
            proba_words_val, words_saved.classes_, proba_struct_val, struct_saved.classes_, w_words, w_struct
        )
        vm_candidate = evaluate_predictions(f"valid[w_struct={w_struct:.3f}]", data.y_val, proba_val, classes)
        if proba_words_val_masked is not None:
            proba_val_masked, _ = align_and_blend_probabilities(
                proba_words_val_masked, words_saved.classes_, proba_struct_val, struct_saved.classes_, w_words, w_struct
            )
            vmm_candidate = evaluate_predictions(
                f"valid_masked[w_struct={w_struct:.3f}]",
                masked_labels["y_val"],
                proba_val_masked,
                classes,
            )
        else:
            vmm_candidate = None
        candidate_results.append(
            {
                "w_struct": float(w_struct),
                "w_words": float(w_words),
                "valid_metrics": vm_candidate,
                "valid_masked_metrics": vmm_candidate,
            }
        )

    best = choose_best_weight(candidate_results, OPTIMIZATION_TARGET)
    best_w_struct = float(best["w_struct"])
    best_w_words = float(best["w_words"])
    vm = best["valid_metrics"]
    vmm = best.get("valid_masked_metrics")
    log(f"[opt] weight search elapsed {(time.time() - t0_opt)/60:.2f} min", 1)
    log(f"[opt] selected weights: words={best_w_words:.4f} struct={best_w_struct:.4f}", 1)
    log(
        f"[opt] selected valid metrics: top1={vm.top1_acc:.4f} macro_f1={vm.top1_f1_macro:.4f} logloss={vm.log_loss:.4f}",
        1,
    )
    if vmm is not None:
        log(
            f"[opt] selected valid_masked metrics: top1={vmm.top1_acc:.4f} macro_f1={vmm.top1_f1_macro:.4f} logloss={vmm.log_loss:.4f}",
            1,
        )

    # probability blending and prediction evaluation in training/validation/test sets

    t0 = time.time()
    proba_train, classes = align_and_blend_probabilities(
        proba_words_train, words_saved.classes_, proba_struct_train, struct_saved.classes_, best_w_words, best_w_struct
    )
    trm = evaluate_predictions("train", data.y_train, proba_train, classes)
    log(
        f"[train] n={trm.n:,} top1={trm.top1_acc:.4f} top5={trm.top5_acc:.4f} top10={trm.top10_acc:.4f} "
        f"mrr={trm.mrr:.4f} logloss={trm.log_loss:.4f} top1_f1_macro={trm.top1_f1_macro:.4f} "
        f"(elapsed {(time.time() - t0)/60:.2f} min)",
        1,
    )

    log(
        f"[valid] n={vm.n:,} top1={vm.top1_acc:.4f} top5={vm.top5_acc:.4f} top10={vm.top10_acc:.4f} "
        f"mrr={vm.mrr:.4f} logloss={vm.log_loss:.4f} top1_f1_macro={vm.top1_f1_macro:.4f}",
        1,
    )

    if vmm is not None:
        log(
            f"[valid_masked] n={vmm.n:,} top1={vmm.top1_acc:.4f} top5={vmm.top5_acc:.4f} top10={vmm.top10_acc:.4f} "
            f"mrr={vmm.mrr:.4f} logloss={vmm.log_loss:.4f} top1_f1_macro={vmm.top1_f1_macro:.4f}",
            1,
        )
    else:
        log("[valid_masked] skipped: masked word artifact not available", 1)

    t0 = time.time()
    proba_test, classes = align_and_blend_probabilities(
        proba_words_test, words_saved.classes_, proba_struct_test, struct_saved.classes_, best_w_words, best_w_struct
    )
    tm = evaluate_predictions("test", data.y_test, proba_test, classes)
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

    # masked model evaluation

    if masked_labels is not None and X_words_test_masked is not None:
        t0 = time.time()
        proba_words_test_masked = words_saved.model.predict_proba(X_words_test_masked)
        proba_test_masked, classes = align_and_blend_probabilities(
            proba_words_test_masked, words_saved.classes_, proba_struct_test, struct_saved.classes_, best_w_words, best_w_struct
        )
        tmm = evaluate_predictions("test_masked", masked_labels["y_test"], proba_test_masked, classes)
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
        log("[test_masked] skipped: masked word artifact not available", 1)


    t0 = time.time()
    proba_test_tempered = apply_temperature_to_probabilities(proba_test, TEMPERATURE)
    tm_tempered = evaluate_predictions(f"test_temp_T={TEMPERATURE:.3f}", data.y_test, proba_test_tempered, classes)
    log(
        f"\n[test_temp_T={TEMPERATURE:.3f}] n={tm_tempered.n:,} hit@1={tm_tempered.top1_acc:.4f} hit@5={tm_tempered.top5_acc:.4f} hit@10={tm_tempered.top10_acc:.4f} "
        f"mrr={tm_tempered.mrr:.4f} logloss={tm_tempered.log_loss:.4f}\n"
        f"       top1 micro P/R/F1={tm_tempered.top1_precision_micro:.4f}/{tm_tempered.top1_recall_micro:.4f}/{tm_tempered.top1_f1_micro:.4f} "
        f"| macro P/R/F1={tm_tempered.top1_precision_macro:.4f}/{tm_tempered.top1_recall_macro:.4f}/{tm_tempered.top1_f1_macro:.4f}\n"
        f"       top{PRF_TOPK} micro P/R/F1={tm_tempered.topk_precision_micro:.4f}/{tm_tempered.topk_recall_micro:.4f}/{tm_tempered.topk_f1_micro:.4f} "
        f"| macro P/R/F1={tm_tempered.topk_precision_macro:.4f}/{tm_tempered.topk_recall_macro:.4f}/{tm_tempered.topk_f1_macro:.4f}\n"
        f"       elapsed {(time.time() - t0)/60:.2f} min",
        1,
    )

    if tmm is not None:
        t0 = time.time()
        proba_test_masked_tempered = apply_temperature_to_probabilities(proba_test_masked, TEMPERATURE)
        tmm_tempered = evaluate_predictions(f"test_masked_temp_T={TEMPERATURE:.3f}", masked_labels["y_test"], proba_test_masked_tempered, classes)
        log(
            f"\n[test_masked_temp_T={TEMPERATURE:.3f}] n={tmm_tempered.n:,} hit@1={tmm_tempered.top1_acc:.4f} hit@5={tmm_tempered.top5_acc:.4f} hit@10={tmm_tempered.top10_acc:.4f} "
            f"mrr={tmm_tempered.mrr:.4f} logloss={tmm_tempered.log_loss:.4f}\n"
            f"       top1 micro P/R/F1={tmm_tempered.top1_precision_micro:.4f}/{tmm_tempered.top1_recall_micro:.4f}/{tmm_tempered.top1_f1_micro:.4f} "
            f"| macro P/R/F1={tmm_tempered.top1_precision_macro:.4f}/{tmm_tempered.top1_recall_macro:.4f}/{tmm_tempered.top1_f1_macro:.4f}\n"
            f"       top{PRF_TOPK} micro P/R/F1={tmm_tempered.topk_precision_micro:.4f}/{tmm_tempered.topk_recall_micro:.4f}/{tmm_tempered.topk_f1_micro:.4f} "
            f"| macro P/R/F1={tmm_tempered.topk_precision_macro:.4f}/{tmm_tempered.topk_recall_macro:.4f}/{tmm_tempered.topk_f1_macro:.4f}\n"
            f"       elapsed {(time.time() - t0)/60:.2f} min",
            1,
        )
    else:
        tmm_tempered = None

    # metadata logging

    metadata = {
        "task": TASK,
        "tag": tag,
        "weights": {
            "words": best_w_words,
            "struct": best_w_struct,
        },
        "temperature": TEMPERATURE,
        "geo_smoothing": {
            "enabled": ENABLE_GEO_SMOOTHING,
            "sigma_km": GEO_SMOOTHING_SIGMA_KM,
        },
        "weight_search": {
            "enabled": OPTIMIZE_WEIGHT_STRUCT,
            "target": OPTIMIZATION_TARGET,
            "candidate_struct_weights": candidate_weights,
            "selected_struct_weight": best_w_struct,
            "selected_words_weight": best_w_words,
            "results": [
                {
                    "w_struct": float(rec["w_struct"]),
                    "w_words": float(rec["w_words"]),
                    "valid_metrics": asdict(rec["valid_metrics"]),
                    "valid_masked_metrics": asdict(rec["valid_masked_metrics"]) if rec.get("valid_masked_metrics") is not None else None,
                }
                for rec in candidate_results
            ],
        },
        "words_model_path": model_paths["words_model"],
        "struct_model_path": model_paths["struct_model"],
        "train_metrics": asdict(trm),
        "valid_metrics": asdict(vm),
        "valid_masked_metrics": asdict(vmm) if vmm is not None else None,
        "test_metrics": asdict(tm),
        "test_tempered_metrics": asdict(tm_tempered),
        "test_masked_metrics": asdict(tmm) if tmm is not None else None,
        "test_masked_tempered_metrics": asdict(tmm_tempered) if tmm_tempered is not None else None,
        "n_train": int(len(data.y_train)),
        "n_val": int(len(data.y_val)),
        "n_test": int(len(data.y_test)),
        "n_classes": int(len(classes)),
    }

    # save metrics to output path and print final updates

    if SAVE_METRICS:
        stub = f"lr__weighted_words_struct__{TASK}__{tag}__wstruct-{best_w_struct:.3f}".replace(".", "p")
        metrics_path = os.path.join(output_path, f"{stub}__metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
        log(f"[saved] metrics -> {metrics_path}", 1)

    log(f"\n[done] total elapsed {(time.time() - t0_all)/60:.2f} min", 1)


if __name__ == "__main__":
    main()
