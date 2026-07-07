from __future__ import annotations

import csv
import json
import math
import os
import pickle
import re
import sqlite3
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

csv.field_size_limit(2**31 - 1)  # match the other resources; some curated rows have very large text fields

from cli import get_args, MODELS_DIR, DATA_DIR
from utils import (
    _sqlite_retry_on_locked,
    build_author_feature_map_from_raw_zst_with_seen,
    iter_author_feature_map_streaming,
    init_author_file_counts_cache,
    init_author_file_counts_caches,
    cache_get_author_file_counts,
    cache_put_author_file_counts,
    cache_get_author_file_counts_sharded,
    cache_put_author_file_counts_sharded,
    location_label_db_path,
    cache_get_locations,
    cache_put_locations,
    check_reqd_files,
    find_raw_month_files,
    get_last_source_row,
    init_location_cache,
    init_location_detail_cache,
    log_report,
    parse_range,
)


### Argument Handling

TOP_CONF_THRESHOLD = 0.60
REG_CONF_MARGIN = 0.10
STA_CONF_MARGIN = 0.05
UNKNOWN_LABEL = "UNK"

MIN_SAMPLES_FOR_INFERENCE = 10
MIN_SAMPLES_FOR_CACHE = 25
# After every FLUSH_EVERY_N_RAW raw .zst files complete, infer locations for
# now-saturated authors and write their rows incrementally to the OUTPUT csv.
# Set to 1 (flush after EVERY completed raw file) for the most frequent output
# the streaming structure allows: each flush re-walks the curated csv from the
# last written source_row, so flushing per inference-batch instead would mean
# hundreds of full-csv rescans per month -- file granularity is the right unit.
# This shrinks the worst-case lost-output window on a wall-time kill to a single
# raw file's scan time. (Distinct from CACHE_FLUSH_ROWS, which drives how often
# the scan/label CACHES are persisted and is batchsize-driven -- see below.)
FLUSH_EVERY_N_RAW = 1
# When an author has at least this many items in the curated (group-filtered)
# input for the current month, the local seed features dominate the inference
# enough that any cross-group cached label is least trustworthy. Skip the cache
# lookup for these authors and re-infer; the cache write step still updates the
# cached row if the freshly-inferred prob beats the cached one.
LOCAL_SEEN_BIAS_THRESHOLD = 40

regional_weights = {"words": 0.7, "struct": 0.3}
top_weights = {"words": 0.55, "struct": 0.45}
state_weights = {"words": 0.5, "struct": 0.5}

RAW_START_YM = (2007, 1)
RAW_END_YM = (2023, 12)
DEFAULT_BATCH_SIZE = 256
PROGRESS_HEARTBEAT_SECONDS = 30 * 60

# Source of the raw zst feature scan, independent of the curated input type.
# >90% of the location model's training data comes from comments, so author
# features should always be drawn from the comment archive even when labeling
# a submissions-side curated file. Override via the LABEL_LOCATION_RAW_TYPE env
# var (e.g. "submissions") only if you have a specific reason to deviate.
RAW_TYPE = os.environ.get("LABEL_LOCATION_RAW_TYPE", "comments").strip() or "comments"

args = get_args()
type_ = args.type
years = parse_range(args.years)
if isinstance(years, int):
    years = [years]
group = args.group
# CLI knob overrides. None means "no explicit value; fall back to the
# per-year-band policy in YEAR_BAND_SCAN_KNOBS". Resolved per-month inside
# label_location_month() via _resolve_scan_knobs_for_year().
_cli_max_items_per_author = getattr(args, "maxitems", None)
_cli_max_files_to_scan = getattr(args, "maxfiles", None)
_cli_max_radius = getattr(args, "maxradius", None)
# Year-band policy: stringent through 2019 (the historical defaults), then
# halved per-author sampling and tighter month window for 2020-2023 where
# raw-zst activity per month is 3-4x higher and the old defaults push the
# scan past the 4-day SLURM wall on the largest months. Each tuple is
# (max_items_per_author, max_files_to_scan, max_radius). Bands are inclusive
# on both ends; the first matching band wins.
YEAR_BAND_SCAN_KNOBS: List[Tuple[int, int, int, int, int]] = [
    # (year_lo, year_hi, max_items_per_author, max_files_to_scan, max_radius)
    (2007, 2019, 50, 60, 30),
    (2020, 2023, 25, 60, 20),
]


def _resolve_scan_knobs_for_year(year: int) -> Tuple[int, int, int]:
    """Return (max_items_per_author, max_files_to_scan, max_radius) for year.

    Picks the matching band from YEAR_BAND_SCAN_KNOBS, then lets any
    explicit CLI override (--maxitems / --maxfiles / --maxradius) win.
    """
    band_items, band_files, band_radius = 50, 60, 30  # ultimate fallback
    for lo, hi, items, files, radius in YEAR_BAND_SCAN_KNOBS:
        if lo <= year <= hi:
            band_items, band_files, band_radius = items, files, radius
            break
    return (
        _cli_max_items_per_author if _cli_max_items_per_author is not None else band_items,
        _cli_max_files_to_scan if _cli_max_files_to_scan is not None else band_files,
        _cli_max_radius if _cli_max_radius is not None else band_radius,
    )


batch_size = max(1, int(getattr(args, "batchsize", DEFAULT_BATCH_SIZE) or DEFAULT_BATCH_SIZE))
# Persist accumulated caches once this many pending (author, file) rows buffer
# up, instead of only at end-of-month. Tied to batch_size so one knob scales
# both the inference micro-batch and the write cadence (approach a): a larger
# batch_size -> larger buffer -> fewer, bigger writes. This bounds the peak
# memory held in pending_file_cache_rows AND banks scan progress mid-month, so a
# 4-day wall-time kill no longer discards the whole month's work (the old design
# wrote once at end-of-month, so timed-out months cached nothing). Year-sharded
# file_counts DBs keep these mid-month writes cheap enough to be safe under
# concurrency. Tune via batch_size; floor keeps small batches from over-writing.
CACHE_FLUSH_ROWS = max(100_000, batch_size * 100)
# Parallel file scan: enabled only for SLURM array tasks (single-process); falls back to 1 locally
# and in multi-process mode to avoid compounding memory with ProcessPoolExecutor workers.
_slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
n_scan_workers = _slurm_cpus if (_slurm_cpus > 0 and getattr(args, "array", None) is not None) else 1


### Path Handling

MODEL_DIR = MODELS_DIR / "label_location" / "trained_lr"
PREPROC_DIR = MODELS_DIR / "label_location" / "preprocessed_streaming"
PREPROC_TAG = "src-all"
PREPROCESSOR_PATH = PREPROC_DIR / f"preprocessor__{PREPROC_TAG}.pkl"
PREPROC_METADATA_PATH = PREPROC_DIR / f"metadata__{PREPROC_TAG}.json"

TOP_WORDS_MODEL = str(MODEL_DIR / f"lr__words__top__{PREPROC_TAG}.pkl")
REG_WORDS_MODEL = str(MODEL_DIR / f"lr__words__region__{PREPROC_TAG}.pkl")
STA_WORDS_MODEL = str(MODEL_DIR / f"lr__words__state__{PREPROC_TAG}.pkl")
TOP_STRUCT_MODEL = str(MODEL_DIR / f"lr__struct__top__{PREPROC_TAG}.pkl")
REG_STRUCT_MODEL = str(MODEL_DIR / f"lr__struct__region__{PREPROC_TAG}.pkl")
STA_STRUCT_MODEL = str(MODEL_DIR / f"lr__struct__state__{PREPROC_TAG}.pkl")

RAW_DIR = DATA_DIR / "data_reddit_raw" / RAW_TYPE

if args.input:
    input_path = Path(args.input)
else:
    input_path = DATA_DIR / "data_reddit_curated" / group / type_ / "labeled_emotion"

file_list = check_reqd_files(years, input_path, type_)
file_list = sorted(file_list, key=lambda p: Path(p).name)

if args.output:
    output_path = Path(args.output)
else:
    output_path = DATA_DIR / "data_reddit_curated" / group / type_ / "labeled_location"
output_path.mkdir(parents=True, exist_ok=True)

# Group-global location cache, keyed by the raw feature source rather than the
# curated input type. Since author features come from RAW_TYPE (comments by
# default), submissions and comments runs share the same cache and all six
# social groups within a raw type share one cache.
CACHE_DIR = DATA_DIR / "data_reddit_curated" / "data_reddit_location"
# Label tables (author_location + author_location_detail) live in ONE DB, keyed
# by author (year-independent) to preserve cross-year/-group dedup. The big,
# regenerable author_file_counts table is sharded into one DB per raw_file year
# (see utils); reads merge across the spiral's years, writes route by file year.
LABEL_DB_PATH = str(CACHE_DIR / f"author_location_label_{RAW_TYPE}.sqlite")
init_location_cache(LABEL_DB_PATH)
# Year-sharded file_counts DBs are pre-created by cli.py for the requested years
# and lazily created on write for any spiral year outside that set.
report_file_path = os.path.join(output_path, "report_label_location.csv")


### Local helpers aligned with train_location_preprocess.py

_token_re = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> List[str]:
    return _token_re.findall((text or "").lower())


def parse_time_to_hour(time_str: str) -> Optional[int]:
    if not time_str:
        return None
    s = str(time_str).strip()
    if s.isdigit():
        try:
            return time.localtime(int(s)).tm_hour
        except Exception:
            return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            import datetime as _dt
            return _dt.datetime.strptime(s, fmt).hour
        except Exception:
            pass
    try:
        import datetime as _dt
        return _dt.datetime.fromisoformat(s).hour
    except Exception:
        return None


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
            key = f"w:{tok}"
            counts[key] = counts.get(key, 0) + 1
    else:
        for tok in tokenize(text):
            if tok not in word_vocab:
                continue
            key = f"w:{tok}"
            counts[key] = counts.get(key, 0) + 1

    subreddit = (subreddit or "").strip()
    if subreddit and (subreddit_vocab is None or subreddit in subreddit_vocab):
        key = f"s:{subreddit}"
        counts[key] = counts.get(key, 0) + 1

    hr = parse_time_to_hour(time_value)
    if hr is not None:
        key = f"h:{hr:02d}"
        counts[key] = counts.get(key, 0) + 1


def _normalize_struct_counts(raw: Dict[str, int], mode: str, weight: float, prefix: str) -> Dict[str, float]:
    if not raw or weight <= 0.0:
        return {}
    out: Dict[str, float] = {}
    total = 0.0
    for k, v in raw.items():
        fv = float(v)
        if fv <= 0:
            continue
        if mode == "log1p_l1":
            tv = math.log1p(fv)
        elif mode in {"l1", "tfidf"}:
            tv = fv
        elif mode == "binary_l1":
            tv = 1.0
        else:
            raise ValueError(f"unsupported struct mode: {mode}")
        out[f"{prefix}{k}"] = tv
        total += tv
    if mode in {"log1p_l1", "l1", "binary_l1"} and total > 0.0:
        scale = weight / total
        out = {k: v * scale for k, v in out.items()}
    elif mode == "tfidf" and weight != 1.0:
        out = {k: v * weight for k, v in out.items()}
    return out


def _extract_year_month_from_name(path: str) -> Optional[Tuple[int, int]]:
    m = re.search(r"(\d{4})-(\d{2})", path)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _ym_to_index(year: int, month: int) -> int:
    return year * 12 + (month - 1)


def month_spiral(year: int, month: int, max_files_to_scan: int = 60, max_radius: int = 30) -> List[Tuple[int, str]]:
    center = _ym_to_index(year, month)
    min_idx = _ym_to_index(*RAW_START_YM)
    max_idx = _ym_to_index(*RAW_END_YM)

    offsets = [0]
    for r in range(1, max_radius + 1):
        offsets.append(-r)
        offsets.append(r)

    out: List[Tuple[int, str]] = []
    seen: set[Tuple[int, str]] = set()
    for off in offsets:
        idx = center + off
        if idx < min_idx or idx > max_idx:
            continue
        y = idx // 12
        mo = idx % 12 + 1
        pair = (y, f"{mo:02d}")
        if pair in seen:
            continue
        seen.add(pair)
        out.append(pair)
        if len(out) >= max_files_to_scan:
            break
    return out


### Cache detail helpers


init_location_detail_cache(LABEL_DB_PATH)


def cache_get_location_details(db_path: str, authors: Sequence[str]) -> Dict[str, Dict[str, object]]:
    if not authors:
        return {}
    def _do() -> Dict[str, Dict[str, object]]:
        conn = sqlite3.connect(db_path, timeout=60)
        try:
            cur = conn.cursor()
            out: Dict[str, Dict[str, object]] = {}
            author_list = list(authors)
            for i in range(0, len(author_list), 900):
                chunk = author_list[i:i + 900]
                qmarks = ",".join(["?"] * len(chunk))
                cur.execute(
                    f"""
                    SELECT author, location, location_prob, contender_location, contender_location_prob,
                           top_location, top_location_prob, top_contender_location, top_contender_location_prob,
                           tier, seen_count
                    FROM author_location_detail
                    WHERE author IN ({qmarks})
                    """,
                    chunk,
                )
                for row in cur.fetchall():
                    out[row[0]] = {
                        "location": row[1],
                        "location_prob": row[2],
                        "contender_location": row[3],
                        "contender_location_prob": row[4],
                        "top_location": row[5],
                        "top_location_prob": row[6],
                        "top_contender_location": row[7],
                        "top_contender_location_prob": row[8],
                        "tier": row[9],
                        "seen_count": row[10],
                    }
            return out
        finally:
            conn.close()
    return _sqlite_retry_on_locked(_do)


def cache_put_location_details(db_path: str, details_by_author: Dict[str, Dict[str, object]]) -> None:
    """Upsert per-author detail rows with confidence-aware overwrite. An existing
    cached row is atomically replaced only when the new location_prob is
    strictly greater than the cached one (or the cached one is NULL); a new
    prob of None never overwrites a labeled row.

    Wrapped in _sqlite_retry_on_locked so that bursts of concurrent writers
    on the NFS-hosted DB don't surface SQLITE_BUSY as a fatal error; the
    write is retried with exponential backoff."""
    if not details_by_author:
        return

    def _do() -> None:
        conn = sqlite3.connect(db_path, timeout=60)
        try:
            cur = conn.cursor()
            now = int(time.time())
            rows = []
            for author, d in details_by_author.items():
                if not author or not d.get("location"):
                    continue
                rows.append(
                    (
                        author,
                        d.get("location"),
                        d.get("location_prob"),
                        d.get("contender_location"),
                        d.get("contender_location_prob"),
                        d.get("top_location"),
                        d.get("top_location_prob"),
                        d.get("top_contender_location"),
                        d.get("top_contender_location_prob"),
                        d.get("tier"),
                        d.get("seen_count"),
                        now,
                    )
                )
            if not rows:
                return
            cur.executemany(
                """
                INSERT INTO author_location_detail(
                    author, location, location_prob, contender_location, contender_location_prob,
                    top_location, top_location_prob, top_contender_location, top_contender_location_prob,
                    tier, seen_count, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(author) DO UPDATE SET
                    location = excluded.location,
                    location_prob = excluded.location_prob,
                    contender_location = excluded.contender_location,
                    contender_location_prob = excluded.contender_location_prob,
                    top_location = excluded.top_location,
                    top_location_prob = excluded.top_location_prob,
                    top_contender_location = excluded.top_contender_location,
                    top_contender_location_prob = excluded.top_contender_location_prob,
                    tier = excluded.tier,
                    seen_count = excluded.seen_count,
                    updated_at = excluded.updated_at
                WHERE excluded.location_prob IS NOT NULL
                  AND (author_location_detail.location_prob IS NULL
                       OR excluded.location_prob > author_location_detail.location_prob)
                """,
                rows,
            )
            conn.commit()
        finally:
            conn.close()

    _sqlite_retry_on_locked(_do)


### Model loading + inference


@dataclass
class SavedModel:
    """Mirror of the SavedModel dataclass from the training script, needed for pickle deserialization."""
    feature_set: str
    task: str
    classes_: list
    model: object
    metadata: dict


class _RemappingUnpickler(pickle.Unpickler):
    """Remap SavedModel to the local class regardless of which __main__ module it was saved from."""
    def find_class(self, module, name):
        if name == "SavedModel":
            return SavedModel
        return super().find_class(module, name)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return _RemappingUnpickler(f).load()


def _unwrap_saved_model(obj) -> Tuple[object, List[str], Dict[str, object]]:
    if hasattr(obj, "model") and hasattr(obj, "classes_"):
        classes = list(getattr(obj, "classes_"))
        metadata = getattr(obj, "metadata", {}) or {}
        return getattr(obj, "model"), classes, metadata
    if hasattr(obj, "predict_proba") and hasattr(obj, "classes_"):
        classes = list(getattr(obj, "classes_"))
        return obj, classes, {}
    raise TypeError(f"Unsupported model payload loaded from pickle: {type(obj)!r}")


class LRBatchScorer:
    def __init__(
        self,
        model_payload,
        feature_set: str,
        selected_words: Sequence[str],
        word_tfidf,
        struct_vectorizer,
        struct_tfidf,
        struct_sub_mode: str,
        struct_hour_mode: str,
    ) -> None:
        self.model, self.classes_, self.metadata = _unwrap_saved_model(model_payload)
        self.feature_set = feature_set
        self.selected_words = list(selected_words)
        self.word_tfidf = word_tfidf
        self.struct_vectorizer = struct_vectorizer
        self.struct_tfidf = struct_tfidf
        self.struct_sub_mode = struct_sub_mode
        self.struct_hour_mode = struct_hour_mode
        self.word_index = {str(w): i for i, w in enumerate(self.selected_words)}

        self.struct_feature_names: List[str] = []
        if struct_vectorizer is not None:
            names = getattr(struct_vectorizer, "feature_names_", None)
            if names is None:
                try:
                    names = list(struct_vectorizer.get_feature_names_out())
                except Exception:
                    names = []
            self.struct_feature_names = list(names or [])
        self.struct_sub_cols = [i for i, name in enumerate(self.struct_feature_names) if str(name).startswith("s:")]

    def _build_word_matrix(self, batch_counts: List[Dict[str, int]]) -> sparse.csr_matrix:
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []
        for rix, counts in enumerate(batch_counts):
            for feat, value in counts.items():
                if not feat.startswith("w:"):
                    continue
                col = self.word_index.get(feat[2:])
                if col is None:
                    continue
                fv = float(value)
                if fv <= 0:
                    continue
                rows.append(rix)
                cols.append(col)
                data.append(fv)
        X_counts = sparse.csr_matrix(
            (np.asarray(data, dtype=np.float32), (rows, cols)),
            shape=(len(batch_counts), len(self.selected_words)),
            dtype=np.float32,
        )
        X_counts.sum_duplicates()
        X_counts.sort_indices()
        return self.word_tfidf.transform(X_counts).astype(np.float32).tocsr()

    def _build_struct_matrix(self, batch_counts: List[Dict[str, int]]) -> sparse.csr_matrix:
        dict_rows: List[Dict[str, float]] = []
        for counts in batch_counts:
            raw_subs = {k[2:]: int(v) for k, v in counts.items() if k.startswith("s:") and int(v) > 0}
            raw_hours = {k[2:]: int(v) for k, v in counts.items() if k.startswith("h:") and int(v) > 0}
            feats: Dict[str, float] = {}
            feats.update(_normalize_struct_counts(raw_subs, self.struct_sub_mode, 1.0, "s:"))
            feats.update(_normalize_struct_counts(raw_hours, self.struct_hour_mode, 1.0, "h:"))
            dict_rows.append(feats)
        X = self.struct_vectorizer.transform(dict_rows).astype(np.float32).tocsr()
        if self.struct_sub_mode == "tfidf" and self.struct_tfidf is not None and self.struct_sub_cols:
            X_sub = self.struct_tfidf.transform(X[:, self.struct_sub_cols])
            X = X.tolil(copy=True)
            X[:, self.struct_sub_cols] = X_sub
            X = X.tocsr().astype(np.float32)
        return X

    def transform(self, batch_counts: List[Dict[str, int]]) -> sparse.csr_matrix:
        if self.feature_set == "words":
            return self._build_word_matrix(batch_counts)
        if self.feature_set == "struct":
            return self._build_struct_matrix(batch_counts)
        raise ValueError(f"Unsupported feature_set for inference scorer: {self.feature_set}")

    def predict_proba(self, batch_counts: List[Dict[str, int]]) -> np.ndarray:
        X = self.transform(batch_counts)
        if X.shape[0] == 0:
            return np.zeros((0, len(self.classes_)), dtype=np.float64)
        return np.asarray(self.model.predict_proba(X), dtype=np.float64)

    def predict_topk(self, batch_counts: List[Dict[str, int]], topk: int = 1) -> List[List[Tuple[str, float]]]:
        proba = self.predict_proba(batch_counts)
        if proba.shape[0] == 0:
            return []
        k = min(max(1, topk), proba.shape[1])
        idx = np.argpartition(-proba, kth=k - 1, axis=1)[:, :k]
        row_vals = np.take_along_axis(proba, idx, axis=1)
        order = np.argsort(-row_vals, axis=1)
        sorted_idx = np.take_along_axis(idx, order, axis=1)
        out: List[List[Tuple[str, float]]] = []
        for row, probs in zip(sorted_idx, np.take_along_axis(proba, sorted_idx, axis=1)):
            out.append([(self.classes_[int(i)], float(p)) for i, p in zip(row, probs)])
        return out


def predict_batch(batch_counts: List[Dict[str, int]], model=None, topk: int = 1, scorer: Optional[LRBatchScorer] = None) -> List[List[Tuple[str, float]]]:
    if scorer is not None:
        return scorer.predict_topk(batch_counts, topk=max(topk, 1))
    if model is None:
        raise ValueError("predict_batch requires either a model/scorer or an explicit scorer")
    raise ValueError("Direct model-only prediction is not supported in this resource; pass scorer instead")


_WORKER_BUNDLE = None


def _verify_required_artifacts() -> None:
    required = [
        PREPROCESSOR_PATH,
        PREPROC_METADATA_PATH,
        Path(TOP_WORDS_MODEL),
        Path(REG_WORDS_MODEL),
        Path(STA_WORDS_MODEL),
        Path(TOP_STRUCT_MODEL),
        Path(REG_STRUCT_MODEL),
        Path(STA_STRUCT_MODEL),
    ]
    missing = [str(p) for p in required if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing required preprocessing/model artifacts: " + ", ".join(missing))


def get_worker_bundle():
    global _WORKER_BUNDLE
    if _WORKER_BUNDLE is not None:
        return _WORKER_BUNDLE

    _verify_required_artifacts()

    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)
    with open(PREPROC_METADATA_PATH, "r", encoding="utf-8") as f:
        preproc_meta = json.load(f)

    selected_words = preprocessor.get("selected_words")
    word_tfidf = preprocessor.get("word_tfidf")
    struct_vectorizer = preprocessor.get("struct_vectorizer")
    struct_tfidf = preprocessor.get("struct_tfidf")
    if selected_words is None or word_tfidf is None or struct_vectorizer is None:
        raise RuntimeError("Preprocessor artifact is missing selected_words, word_tfidf, or struct_vectorizer")

    struct_sub_mode = str(preproc_meta.get("struct_sub_mode", "log1p_l1"))
    struct_hour_mode = str(preproc_meta.get("struct_hour_mode", "l1"))

    _WORKER_BUNDLE = {
        "top_words": LRBatchScorer(load_pickle(TOP_WORDS_MODEL), "words", selected_words, word_tfidf, struct_vectorizer, struct_tfidf, struct_sub_mode, struct_hour_mode),
        "top_struct": LRBatchScorer(load_pickle(TOP_STRUCT_MODEL), "struct", selected_words, word_tfidf, struct_vectorizer, struct_tfidf, struct_sub_mode, struct_hour_mode),
        "reg_words": LRBatchScorer(load_pickle(REG_WORDS_MODEL), "words", selected_words, word_tfidf, struct_vectorizer, struct_tfidf, struct_sub_mode, struct_hour_mode),
        "reg_struct": LRBatchScorer(load_pickle(REG_STRUCT_MODEL), "struct", selected_words, word_tfidf, struct_vectorizer, struct_tfidf, struct_sub_mode, struct_hour_mode),
        "sta_words": LRBatchScorer(load_pickle(STA_WORDS_MODEL), "words", selected_words, word_tfidf, struct_vectorizer, struct_tfidf, struct_sub_mode, struct_hour_mode),
        "sta_struct": LRBatchScorer(load_pickle(STA_STRUCT_MODEL), "struct", selected_words, word_tfidf, struct_vectorizer, struct_tfidf, struct_sub_mode, struct_hour_mode),
    }
    return _WORKER_BUNDLE


### Probability aggregation / decision logic


def _rankings_to_prob_map(rankings: List[Tuple[str, float]]) -> Dict[str, float]:
    return {label: float(prob) for label, prob in rankings if label}


def blend_rankings(rankings_by_name: Dict[str, List[Tuple[str, float]]], weights: Dict[str, float], topk: int = 2) -> List[Tuple[str, float]]:
    totals: Dict[str, float] = {}
    weight_sum = 0.0
    for name, rankings in rankings_by_name.items():
        weight = float(weights.get(name, 0.0))
        if weight <= 0.0:
            continue
        prob_map = _rankings_to_prob_map(rankings)
        if not prob_map:
            continue
        weight_sum += weight
        for label, prob in prob_map.items():
            totals[label] = totals.get(label, 0.0) + (weight * prob)
    if weight_sum <= 0.0:
        return []
    blended = [(label, score / weight_sum) for label, score in totals.items()]
    blended.sort(key=lambda x: (-x[1], x[0]))
    return blended[:max(1, topk)]


def _margin(scores: List[Tuple[str, float]]) -> float:
    if len(scores) < 2:
        return float("inf")
    return float(scores[0][1]) - float(scores[1][1])


def _fmt_prob(value: Optional[float]) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.6f}"
    except Exception:
        return ""


def unknown_result(top_scores: Optional[List[Tuple[str, float]]] = None, seen_count: int = 0, reason: str = "unknown") -> Dict[str, object]:
    top1_label = ""
    top1_prob: Optional[float] = None
    top2_label = ""
    top2_prob: Optional[float] = None
    if top_scores:
        top1_label = top_scores[0][0]
        top1_prob = float(top_scores[0][1])
        if len(top_scores) >= 2:
            top2_label = top_scores[1][0]
            top2_prob = float(top_scores[1][1])
    # When the final label is UNK, the contender exposes the rejected best guess
    # (top-1) so the field consistently means "best candidate not chosen as the
    # label" across UNK and labeled rows. The top_contender_* fields keep the
    # top-level classifier's raw runner-up.
    return {
        "location": UNKNOWN_LABEL,
        "location_prob": None,
        "contender_location": top1_label,
        "contender_location_prob": top1_prob,
        "top_location": top1_label,
        "top_location_prob": top1_prob,
        "top_contender_location": top2_label,
        "top_contender_location_prob": top2_prob,
        "tier": reason,
        "seen_count": seen_count,
    }


def top_level_fallback(top_scores: List[Tuple[str, float]], seen_count: int, reason: str) -> Dict[str, object]:
    top1 = top_scores[0] if top_scores else (UNKNOWN_LABEL, None)
    top2 = top_scores[1] if len(top_scores) >= 2 else ("", None)
    return {
        "location": top1[0] if top1[0] else UNKNOWN_LABEL,
        "location_prob": float(top1[1]) if top1[1] is not None else None,
        "contender_location": top2[0] if top2[0] else "",
        "contender_location_prob": float(top2[1]) if top2[1] is not None else None,
        "top_location": top1[0] if top1[0] else "",
        "top_location_prob": float(top1[1]) if top1[1] is not None else None,
        "top_contender_location": top2[0] if top2[0] else "",
        "top_contender_location_prob": float(top2[1]) if top2[1] is not None else None,
        "tier": reason,
        "seen_count": seen_count,
    }


def final_tier_result(label_scores: List[Tuple[str, float]], top_scores: List[Tuple[str, float]], tier: str, seen_count: int) -> Dict[str, object]:
    top1 = label_scores[0] if label_scores else (UNKNOWN_LABEL, None)
    top2 = label_scores[1] if len(label_scores) >= 2 else ("", None)
    top_top1 = top_scores[0] if top_scores else ("", None)
    top_top2 = top_scores[1] if len(top_scores) >= 2 else ("", None)
    return {
        "location": top1[0] if top1[0] else UNKNOWN_LABEL,
        "location_prob": float(top1[1]) if top1[1] is not None else None,
        "contender_location": top2[0] if top2[0] else "",
        "contender_location_prob": float(top2[1]) if top2[1] is not None else None,
        "top_location": top_top1[0] if top_top1[0] else "",
        "top_location_prob": float(top_top1[1]) if top_top1[1] is not None else None,
        "top_contender_location": top_top2[0] if top_top2[0] else "",
        "top_contender_location_prob": float(top_top2[1]) if top_top2[1] is not None else None,
        "tier": tier,
        "seen_count": seen_count,
    }


def infer_locations_for_batch(batch_counts: List[Dict[str, int]], batch_seen: List[int], bundle) -> List[Dict[str, object]]:
    top_words = predict_batch(batch_counts, scorer=bundle["top_words"], topk=2)
    top_struct = predict_batch(batch_counts, scorer=bundle["top_struct"], topk=2)
    reg_words = predict_batch(batch_counts, scorer=bundle["reg_words"], topk=2)
    reg_struct = predict_batch(batch_counts, scorer=bundle["reg_struct"], topk=2)
    sta_words = predict_batch(batch_counts, scorer=bundle["sta_words"], topk=2)
    sta_struct = predict_batch(batch_counts, scorer=bundle["sta_struct"], topk=2)

    out: List[Dict[str, object]] = []
    for idx, seen in enumerate(batch_seen):
        top_scores = blend_rankings(
            {"words": top_words[idx], "struct": top_struct[idx]},
            top_weights,
            topk=2,
        )
        if seen < MIN_SAMPLES_FOR_INFERENCE:
            out.append(unknown_result(top_scores=top_scores, seen_count=seen, reason="low_samples"))
            continue
        if not top_scores or not top_scores[0][0]:
            out.append(unknown_result(top_scores=top_scores, seen_count=seen, reason="no_top_prediction"))
            continue
        if float(top_scores[0][1]) < TOP_CONF_THRESHOLD:
            out.append(unknown_result(top_scores=top_scores, seen_count=seen, reason="low_top_conf"))
            continue

        top_label = top_scores[0][0]
        if top_label == "NON_US":
            region_scores = blend_rankings(
                {"words": reg_words[idx], "struct": reg_struct[idx]},
                regional_weights,
                topk=2,
            )
            if region_scores and _margin(region_scores) >= REG_CONF_MARGIN:
                out.append(final_tier_result(region_scores, top_scores, tier="region", seen_count=seen))
            else:
                out.append(top_level_fallback(top_scores, seen_count=seen, reason="top_non_us_fallback"))
        elif top_label == "US":
            state_scores = blend_rankings(
                {"words": sta_words[idx], "struct": sta_struct[idx]},
                state_weights,
                topk=2,
            )
            if state_scores and _margin(state_scores) >= STA_CONF_MARGIN:
                out.append(final_tier_result(state_scores, top_scores, tier="state", seen_count=seen))
            else:
                out.append(top_level_fallback(top_scores, seen_count=seen, reason="top_us_fallback"))
        else:
            out.append(top_level_fallback(top_scores, seen_count=seen, reason="top_only"))
    return out


### Location labeling pipeline


def _find_header_index(header: Sequence[str], name: str, fallback: int) -> int:
    try:
        return header.index(name)
    except ValueError:
        return fallback


def _scan_month_authors(
    curated_csv_path: str,
    last_processed: int,
) -> Tuple[List[str], Dict[str, int], int, int, set, Dict[str, int]]:
    """Pass 1 of the streaming pipeline.

    Scans the curated CSV header + rows without holding row contents in memory.
    Returns:
      header (the input CSV header row),
      idx_map (column indices: author/text/time/subreddit/source_row; -1 if absent),
      total_rows (parsed non-empty rows),
      remaining_row_count (rows that still need to be written: source_row >
        last_processed, or all rows on a fresh run / file without source_row),
      target_row_authors (authors appearing in remaining rows; excludes
        "[deleted]" and empty author),
      local_seen (per-author total curated activity in this month; counts
        every parsed row with a valid author, not just remaining rows).

    Raises StopIteration if the CSV is empty (no header row).
    """
    rows_total = 0
    remaining_row_count = 0
    target_row_authors: set = set()
    local_seen: Dict[str, int] = {}

    with open(curated_csv_path, "r", encoding="utf-8-sig", errors="ignore") as f:
        reader = csv.reader((line.replace("\x00", "") for line in f))
        header = next(reader)  # propagates StopIteration on empty file

        author_idx = _find_header_index(header, "author", 3)
        text_idx = _find_header_index(header, "text", 2)
        time_idx = _find_header_index(header, "time", 4)
        subreddit_idx = _find_header_index(header, "subreddit", 5)
        src_idx = header.index("source_row") if "source_row" in header else -1
        id_idx = header.index("id") if "id" in header else -1
        idx_map = {
            "author": author_idx,
            "text": text_idx,
            "time": time_idx,
            "subreddit": subreddit_idx,
            "source_row": src_idx,
            "id": id_idx,
        }

        filter_by_src = src_idx >= 0 and last_processed >= 0

        for r in reader:
            if not r:
                continue
            rows_total += 1

            # Track every parsed row's author in local_seen so the bias
            # threshold check and inference seen_count match the original
            # behavior, even for rows that have already been written
            # (source_row <= last_processed).
            author = r[author_idx].strip() if len(r) > author_idx else ""
            if author and author != "[deleted]":
                local_seen[author] = local_seen.get(author, 0) + 1

            if filter_by_src:
                if len(r) <= src_idx:
                    continue
                try:
                    if int(r[src_idx].strip()) <= last_processed:
                        continue
                except ValueError:
                    continue

            remaining_row_count += 1
            if author and author != "[deleted]":
                target_row_authors.add(author)

    return header, idx_map, rows_total, remaining_row_count, target_row_authors, local_seen


def _seed_features_for_authors(
    curated_csv_path: str,
    idx_map: Dict[str, int],
    target_authors_set: set,
    word_vocab: Optional[set] = None,
    subreddit_vocab: Optional[set] = None,
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, set]]:
    """Pass 2 of the streaming pipeline.

    Re-reads the curated CSV and accumulates per-author feature counts only
    for authors in target_authors_set. Authors not in the set are skipped,
    so local_counts only holds entries for authors that still need
    inference -- typically ~75% of the curated authors are pre-cached and
    don't need seed features.

    Returns (local_counts, curated_seen_ids). curated_seen_ids[author] is the
    set of comment/submission IDs the author contributed via the curated CSV;
    Pass 3's raw scanner uses it to deduplicate posts that appear in BOTH the
    curated CSV (already counted as local) AND the same month's raw .zst (would
    otherwise be counted again as raw), correcting a per-post double-count
    that's been latent in the original pipeline. Dedup is applied only when
    the raw scanner reads the target month's .zst file -- other spiral months'
    files have no overlap with this month's curated content.
    """
    local_counts: Dict[str, Dict[str, int]] = {}
    curated_seen_ids: Dict[str, set] = {}

    author_idx = idx_map["author"]
    text_idx = idx_map["text"]
    time_idx = idx_map["time"]
    subreddit_idx = idx_map["subreddit"]
    id_idx = idx_map.get("id", -1)

    with open(curated_csv_path, "r", encoding="utf-8-sig", errors="ignore") as f:
        reader = csv.reader((line.replace("\x00", "") for line in f))
        try:
            next(reader)  # skip header
        except StopIteration:
            return local_counts, curated_seen_ids

        for r in reader:
            if not r or len(r) <= author_idx:
                continue
            author = r[author_idx].strip()
            if author not in target_authors_set:
                continue
            counts = local_counts.setdefault(author, {})
            add_features_for_row(
                counts,
                text=r[text_idx] if len(r) > text_idx else "",
                subreddit=r[subreddit_idx] if len(r) > subreddit_idx else "",
                time_value=r[time_idx] if len(r) > time_idx else "",
                word_vocab=word_vocab,
                subreddit_vocab=subreddit_vocab,
            )
            if id_idx >= 0 and len(r) > id_idx:
                post_id = r[id_idx].strip()
                if post_id:
                    curated_seen_ids.setdefault(author, set()).add(post_id)

    return local_counts, curated_seen_ids


def _merge_feature_maps(
    local_counts: Dict[str, Dict[str, int]],
    local_seen: Dict[str, int],
    raw_counts: Dict[str, Dict[str, int]],
    raw_seen: Dict[str, int],
    target_authors: Sequence[str],
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    # Merge in place into raw_counts (it dominates memory since each target
    # author has up to max_items_per_author raw items vs typically much less
    # curated activity). Empty {} is created for authors absent from raw_counts
    # so the returned dict has one entry per target author. Items popped from
    # local_counts as we go to release its memory before the inference loop.
    target_set = set(target_authors)
    merged_seen: Dict[str, int] = {}
    for author in list(local_counts.keys()):
        if author not in target_set:
            local_counts.pop(author, None)
            continue
        src = local_counts.pop(author)
        dst = raw_counts.get(author)
        if dst is None:
            raw_counts[author] = src
        else:
            for k, v in src.items():
                dst[k] = dst.get(k, 0) + int(v)
    for author in target_authors:
        if author not in raw_counts:
            raw_counts[author] = {}
        merged_seen[author] = int(local_seen.get(author, 0)) + int(raw_seen.get(author, 0))
    return raw_counts, merged_seen


def _details_from_cache_or_label(author: str, cached_details: Dict[str, Dict[str, object]], cached_locations: Dict[str, str]) -> Dict[str, object]:
    if author in cached_details:
        return cached_details[author]
    loc = cached_locations.get(author, "")
    return {
        "location": loc or UNKNOWN_LABEL,
        "location_prob": None,
        "contender_location": "",
        "contender_location_prob": None,
        "top_location": loc or "",
        "top_location_prob": None,
        "top_contender_location": "",
        "top_contender_location_prob": None,
        "tier": "cache_legacy",
        "seen_count": None,
    }


def _stream_write_output(
    curated_csv_path: str,
    out_file: Path,
    last_processed: int,
    idx_map: Dict[str, int],
    detail_by_author: Dict[str, Dict[str, object]],
    write_mode: str = "w",
) -> None:
    """Pass 3 of the streaming pipeline.

    Re-reads the curated CSV row by row and writes each row that still needs
    writing to out_file with the four location columns appended. Never holds
    the row set in memory, so the dominant peak-RAM term (the curated
    month's rows: 5-8 GB for the largest months) is freed.

    Crash recovery: writes are streamed in input order. If the writer dies
    mid-row, the next run's get_last_source_row truncates the partial trailing
    row before re-opening in append mode.
    """
    author_idx = idx_map["author"]
    src_idx = idx_map.get("source_row", -1)
    filter_by_src = src_idx >= 0 and last_processed >= 0

    with open(curated_csv_path, "r", encoding="utf-8-sig", errors="ignore") as fi, \
            open(out_file, write_mode, encoding="utf-8", newline="", errors="ignore") as fo:
        reader = csv.reader((line.replace("\x00", "") for line in fi))
        writer = csv.writer(fo)

        try:
            header = next(reader)
        except StopIteration:
            return

        if write_mode == "w":
            writer.writerow(
                list(header)
                + ["location", "location_prob", "contender_location", "contender_location_prob"]
            )

        for r in reader:
            if not r:
                continue
            if filter_by_src:
                if len(r) <= src_idx:
                    continue
                try:
                    if int(r[src_idx].strip()) <= last_processed:
                        continue
                except ValueError:
                    continue

            author = r[author_idx].strip() if len(r) > author_idx else ""
            detail = detail_by_author.get(author, {"location": UNKNOWN_LABEL})
            writer.writerow(
                list(r)
                + [
                    detail.get("location", UNKNOWN_LABEL),
                    _fmt_prob(detail.get("location_prob")),
                    detail.get("contender_location", "") or "",
                    _fmt_prob(detail.get("contender_location_prob")),
                ]
            )


def _flush_completed_prefix(
    curated_csv_path: str,
    out_file: Path,
    last_processed: int,
    idx_map: Dict[str, int],
    detail_by_author: Dict[str, Dict[str, object]],
    write_mode: str,
) -> Tuple[int, int]:
    """Walk the curated CSV in source_row order and write the longest leading
    prefix of rows beyond `last_processed` whose author label is known.

    A row is "writable" when:
      - its author is empty / "[deleted]" (defaults to UNK), or
      - its author appears in detail_by_author.

    The walk STOPS at the first writable-gating failure (a real author whose
    detail is not yet known). This preserves monotonic source_row ordering in
    the output so get_last_source_row can correctly resume an interrupted run.

    Returns (new_last_processed, rows_written). When rows_written == 0 the file
    is not modified and `write_mode` should still apply to the next call.
    """
    author_idx = idx_map["author"]
    src_idx = idx_map.get("source_row", -1)
    filter_by_src = src_idx >= 0 and last_processed >= 0
    new_last = last_processed
    rows_written = 0

    # We open the output in the requested mode only if/when we have at least
    # one row to write -- avoids creating an empty header-only file when the
    # first flush has nothing to do (e.g. the very first row's author has not
    # been inferred yet).
    fi = open(curated_csv_path, "r", encoding="utf-8-sig", errors="ignore")
    try:
        reader = csv.reader((line.replace("\x00", "") for line in fi))
        try:
            header = next(reader)
        except StopIteration:
            return last_processed, 0

        fo = None
        writer = None
        try:
            for r in reader:
                if not r:
                    continue

                row_src: Optional[int] = None
                if src_idx >= 0 and len(r) > src_idx:
                    try:
                        row_src = int(r[src_idx].strip())
                    except ValueError:
                        row_src = None

                if filter_by_src:
                    if row_src is None or row_src <= last_processed:
                        continue

                author = r[author_idx].strip() if len(r) > author_idx else ""
                if author and author not in detail_by_author:
                    # First unresolved author -> stop; everything after must
                    # wait until that author's detail is known so we preserve
                    # source_row monotonicity in the output.
                    break

                # Lazy-open the output writer on first writable row.
                if fo is None:
                    fo = open(out_file, write_mode, encoding="utf-8", newline="", errors="ignore")
                    writer = csv.writer(fo)
                    if write_mode == "w":
                        writer.writerow(
                            list(header)
                            + ["location", "location_prob", "contender_location", "contender_location_prob"]
                        )

                detail = detail_by_author.get(author, {"location": UNKNOWN_LABEL})
                writer.writerow(
                    list(r)
                    + [
                        detail.get("location", UNKNOWN_LABEL),
                        _fmt_prob(detail.get("location_prob")),
                        detail.get("contender_location", "") or "",
                        _fmt_prob(detail.get("contender_location_prob")),
                    ]
                )
                rows_written += 1
                if row_src is not None and row_src > new_last:
                    new_last = row_src
        finally:
            if fo is not None:
                fo.close()
    finally:
        fi.close()

    return new_last, rows_written


def label_location_month(curated_csv_path: str) -> Tuple[str, int, int, int]:
    curated_csv_path = str(curated_csv_path)
    stem = Path(curated_csv_path).stem
    log_report(report_file_path, f"Started labeling user location for {stem} for the {group} social group")

    ym = _extract_year_month_from_name(curated_csv_path)
    if ym is None:
        log_report(report_file_path, f"[warn] could not parse year-month from {curated_csv_path}; skipping")
        return (stem, 0, 0, 0)
    year, month_int = ym

    # Resolve per-year-band scan knobs (CLI overrides win if set).
    max_items_per_author, max_files_to_scan, max_radius = _resolve_scan_knobs_for_year(year)

    start = time.time()

    out_file = output_path / f"{stem}.csv"
    last_processed = get_last_source_row(
        out_file,
        report_file_path=report_file_path,
        file_for_log=curated_csv_path,
    )
    write_mode = "a" if last_processed >= 0 else "w"

    # Load model bundle up front so we can pass its word vocab into both the
    # curated seed pass and the raw-zst scan. OOV word-filtering at scan time
    # prevents the per-author count dicts from accumulating w:tok features
    # that would be dropped by word_index lookup anyway (Reddit's token Zipf
    # tail vs the 50k selected_words), which is the dominant peak-RAM term
    # and what was driving the 47-58 GiB cgroup OOM kills on the 32G
    # label_location array tasks.
    #
    # Subreddit OOV is NOT filtered here even though most subreddits are
    # long-tail OOV: _normalize_struct_counts divides s: counts by their full
    # total before struct_vectorizer.transform drops OOV columns, so removing
    # OOV subreddits at scan time would shift the normalization total relative
    # to what the saved struct model was trained against. Hour features are
    # fully in-vocab (24/24) so no filter is needed.
    bundle = get_worker_bundle()
    word_vocab = set(bundle["top_words"].selected_words)

    # Pass 1: scan the curated CSV to determine which rows still need writing
    # (source_row > last_processed) and accumulate per-author totals
    # (local_seen). Rows themselves are not retained -- pass 2 and pass 3
    # re-read the file. Trades two extra CSV passes (a few seconds each on
    # OS-cached input) for ~5-8 GB of peak RAM on the largest months.
    try:
        header, idx_map, rows_total, remaining_row_count, target_row_authors, local_seen = \
            _scan_month_authors(curated_csv_path, last_processed)
    except StopIteration:
        return (stem, 0, 0, 0)

    n_total = len(local_seen)

    if remaining_row_count == 0:
        log_report(report_file_path, f"[skip-complete] user location labeling for {stem}: all rows already processed (last_source_row={last_processed})")
        return (stem, rows_total, n_total, 0)

    if not target_row_authors:
        # No author-bearing rows remain. Still need to write any "[deleted]"
        # / empty-author rows through the output as UNK so resume points
        # match input.
        _stream_write_output(curated_csv_path, out_file, last_processed, idx_map, {}, write_mode=write_mode)
        elapsed = (time.time() - start) / 60
        log_report(report_file_path, f"[done-no-authors] user location labeling for {stem}: rows={remaining_row_count:,} minutes={elapsed:.2f}")
        return (stem, remaining_row_count, n_total, 0)

    # Cache lookups are restricted to authors that still need a row written.
    cached_locations = cache_get_locations(LABEL_DB_PATH, target_row_authors)
    cached_details = cache_get_location_details(LABEL_DB_PATH, list(target_row_authors))

    detail_by_author: Dict[str, Dict[str, object]] = {}
    n_skipped_bias = 0
    for author in target_row_authors:
        # For authors whose curated-input footprint is large, the local seed
        # features dominate inference enough that any cross-group cached label
        # is least trustworthy. Skip the lookup so they get re-inferred from
        # this group's full feature mix; the cache write step still updates
        # the cached row if the new prob beats the existing one.
        if int(local_seen.get(author, 0)) >= LOCAL_SEEN_BIAS_THRESHOLD:
            if author in cached_locations or author in cached_details:
                n_skipped_bias += 1
            continue
        if author in cached_locations or author in cached_details:
            detail_by_author[author] = _details_from_cache_or_label(author, cached_details, cached_locations)

    remaining_authors = sorted(a for a in target_row_authors if a not in detail_by_author)
    n_cached = len(detail_by_author)

    if not remaining_authors:
        _stream_write_output(curated_csv_path, out_file, last_processed, idx_map, detail_by_author, write_mode=write_mode)
        elapsed = (time.time() - start) / 60
        log_report(report_file_path, f"[done-cache] user location labeling for {stem}: rows={remaining_row_count:,} authors={len(target_row_authors):,} cached={n_cached:,} minutes={elapsed:.2f}")
        return (stem, remaining_row_count, len(target_row_authors), 0)

    scan_months = month_spiral(year, month_int, max_files_to_scan=max_files_to_scan, max_radius=max_radius)
    raw_files: List[str] = []
    months_with_files = 0
    for y, mstr in scan_months:
        files = find_raw_month_files(RAW_DIR, RAW_TYPE, y, mstr)
        if files:
            raw_files.extend(files)
            months_with_files += 1

    if not raw_files:
        for author in remaining_authors:
            detail_by_author[author] = unknown_result(seen_count=local_seen.get(author, 0), reason="no_raw_files")
        _stream_write_output(curated_csv_path, out_file, last_processed, idx_map, detail_by_author, write_mode=write_mode)
        elapsed = (time.time() - start) / 60
        log_report(report_file_path, f"[warn] user location labeling for {stem}: no raw files in scan window; wrote UNKNOWN for {len(remaining_authors):,}. minutes={elapsed:.2f}")
        return (stem, remaining_row_count, n_total, len(remaining_authors))

    # Identify the target month's raw .zst basename(s). These must ALWAYS be
    # rescanned (cache never short-circuits them) because per-post dedup needs
    # the scanner to read this run's curated_seen_ids alongside the raw posts,
    # and the cached counts -- being group-agnostic -- don't carry that info.
    target_month_str = f"{month_int:02d}"
    target_month_files = find_raw_month_files(RAW_DIR, RAW_TYPE, year, target_month_str)
    target_month_basenames = {os.path.basename(p) for p in target_month_files}

    # Persistent scan-state cache lookup: subtract previously-scanned files
    # from each author's per-spiral workload. Authors whose cached scanned_files
    # already cover this spiral's basenames are "fully cached" -- no raw-zst
    # scan at all this month, EXCEPT for the target month's file which must
    # always be opened so we can run the dedup pass against the current
    # curated_seen_ids.
    remaining_authors_set = set(remaining_authors)
    spiral_basenames = {os.path.basename(rf) for rf in raw_files}
    # Exclude target month basenames from the cached aggregation so we don't
    # double-count features that a prior run's spiral happened to scan from
    # this month's raw .zst. Those rows still live in the cache (other-target
    # months' spirals can use them); they just don't contribute to *this*
    # run's cached_counts because we will scan the target month fresh below.
    # The spiral can straddle a few years; read+merge only those year-sharded
    # file_counts DBs. Each raw_file lives in exactly one year DB, so the merge
    # never double-counts.
    spiral_years = sorted({str(y) for (y, _m) in scan_months})
    scan_state_existing = cache_get_author_file_counts_sharded(
        str(CACHE_DIR),
        RAW_TYPE,
        spiral_years,
        remaining_authors_set,
        exclude_basenames=target_month_basenames,
    )
    cached_counts_by_author: Dict[str, Dict[str, int]] = {}
    cached_seen_by_author: Dict[str, int] = {}
    cached_scanned_by_author: Dict[str, set] = {}
    per_file_targets: Dict[str, set] = {}
    fully_cached_authors: set = set()
    for author in remaining_authors_set:
        state = scan_state_existing.get(author)
        # The target month's files are always added to per_file_targets so
        # they get rescanned for dedup; cache hits for those files are
        # ignored. For other spiral files, normal cache-hit subtraction.
        if state is not None:
            scanned_files, counts, seen = state
            cached_counts_by_author[author] = counts
            cached_seen_by_author[author] = seen
            cached_scanned_by_author[author] = scanned_files
            new_files = (spiral_basenames - scanned_files) | target_month_basenames
            if not new_files:
                fully_cached_authors.add(author)
                continue
            for fb in new_files:
                per_file_targets.setdefault(fb, set()).add(author)
        else:
            for fb in spiral_basenames:
                per_file_targets.setdefault(fb, set()).add(author)
    files_to_scan = [rf for rf in raw_files if os.path.basename(rf) in per_file_targets]

    log_report(
        report_file_path,
        f"[start] {stem}: authors={n_total:,} cached={n_cached:,} cache_skipped_bias={n_skipped_bias:,} "
        f"need_raw={len(remaining_authors):,} scan_state_hits={len(scan_state_existing):,} fully_cached_for_raw={len(fully_cached_authors):,} "
        f"raw_type={RAW_TYPE} scan_months={len(scan_months)} months_with_files={months_with_files} "
        f"raw_files_in_spiral={len(raw_files)} raw_files_to_scan={len(files_to_scan)} "
        f"samples_per_author={max_items_per_author} batch_size={batch_size} max_files_to_scan={max_files_to_scan} max_radius={max_radius} "
        f"local_seen_bias_threshold={LOCAL_SEEN_BIAS_THRESHOLD} flush_every_n_raw={FLUSH_EVERY_N_RAW}",
    )

    # Pass 2: seed curated features ONLY for the authors that still need
    # inference (typically ~25-75% of target_row_authors after cache+bias
    # filtering). Drops local_counts entries for cache-hit authors entirely,
    # saving ~25-50% of the local_counts peak vs the old "seed for everyone
    # then filter" pattern.
    # (remaining_authors_set already computed above for the scan-state lookup.)
    # Also collects curated_seen_ids (Dict[author, set[post_id]]) so Pass 3 can
    # subtract from raw scan any posts that already contributed via local_counts.
    local_counts, curated_seen_ids = _seed_features_for_authors(
        curated_csv_path,
        idx_map,
        remaining_authors_set,
        word_vocab=word_vocab,
    )

    # Initial incremental flush: write the longest curated prefix whose authors
    # we already know from the cache (or are empty/[deleted]). Cuts down the
    # downstream final-write workload and, on resume, lets crash-after-cache-hits
    # tasks skip these rows immediately.
    if detail_by_author:
        new_last, rows_init = _flush_completed_prefix(
            curated_csv_path, out_file, last_processed, idx_map, detail_by_author, write_mode
        )
        if rows_init > 0:
            last_processed = new_last
            write_mode = "a"
            log_report(
                report_file_path,
                f"[stream-flush] {stem}: trigger=cache-init rows_written={rows_init:,} last_processed={last_processed}",
            )

    # Pass 3 (streaming): scan raw .zst files, and after every FLUSH_EVERY_N_RAW
    # files completed, infer locations for newly-saturated authors and flush
    # their rows. A crash during the multi-day raw scan now leaves a usable
    # partial output that the next submission can resume from.
    raw_scan_start = time.time()
    raw_counts_state: Dict[str, Dict[str, int]] = {}
    raw_seen_state: Dict[str, int] = {a: 0 for a in remaining_authors_set}
    files_since_flush = 0
    files_done = 0
    inferred_authors: set = set()  # authors already moved into detail_by_author during streaming
    to_cache_running: Dict[str, Dict[str, object]] = {}
    n_cache_confident = 0
    n_cache_skipped_lowconf = 0
    n_cache_skipped_lowsamples = 0
    n_stream_flushes = 0

    # Emit a [scan-progress] line roughly every 10% of the spiral so a long
    # silent raw-scan phase is observable without flooding the log. Capped at
    # ~10 lines per month regardless of year: with the default spirals that is
    # every ~6 files (early years, 60-file spiral) / ~4 files (later years,
    # ~41-file spiral). max(1, ...) keeps it sane for tiny spirals.
    scan_log_every = max(1, len(files_to_scan) // 10)
    next_scan_log = scan_log_every

    def _infer_and_flush(saturated_chunk: List[str], trigger: str) -> Tuple[int, int]:
        """Infer for a list of authors, cache the confident ones, flush rows.
        Returns (rows_written, n_inferred). Mutates closure state."""
        nonlocal last_processed, write_mode, n_cache_confident, n_cache_skipped_lowconf, n_cache_skipped_lowsamples, n_stream_flushes
        if not saturated_chunk:
            return 0, 0
        n_inferred = 0
        for i in range(0, len(saturated_chunk), batch_size):
            chunk = saturated_chunk[i:i + batch_size]
            chunk_counts: List[Dict[str, int]] = []
            chunk_seen: List[int] = []
            for a in chunk:
                merged: Dict[str, int] = {}
                lc = local_counts.get(a)
                if lc:
                    merged.update(lc)
                # Fold cached per-author counts from prior months' scans into
                # the inference input -- the persistent scan-state cache stores
                # raw-zst feature contributions across spirals so we get the
                # full historical context here even though this month only
                # scanned the spiral-delta files.
                cc = cached_counts_by_author.get(a)
                if cc:
                    for k, v in cc.items():
                        merged[k] = merged.get(k, 0) + v
                rc = raw_counts_state.get(a)
                if rc:
                    for k, v in rc.items():
                        merged[k] = merged.get(k, 0) + v
                # Subtract the target-month overlap so posts already counted
                # via local_counts (curated) aren't counted again via raw.
                ov = raw_overlap_counts_state.get(a)
                if ov:
                    for k, v in ov.items():
                        new_val = merged.get(k, 0) - v
                        if new_val > 0:
                            merged[k] = new_val
                        elif k in merged:
                            del merged[k]
                chunk_counts.append(merged)
                chunk_seen.append(
                    int(local_seen.get(a, 0))
                    + int(cached_seen_by_author.get(a, 0))
                    + int(raw_seen_state.get(a, 0))
                    - int(raw_overlap_seen_state.get(a, 0))
                )
            batch_results = infer_locations_for_batch(chunk_counts, chunk_seen, bundle)
            for author, detail in zip(chunk, batch_results):
                detail_by_author[author] = detail
                inferred_authors.add(author)
                n_inferred += 1
                seen = int(detail.get("seen_count") or 0)
                if seen < MIN_SAMPLES_FOR_CACHE:
                    n_cache_skipped_lowsamples += 1
                    continue
                if detail.get("location") == UNKNOWN_LABEL:
                    n_cache_skipped_lowconf += 1
                    continue
                to_cache_running[author] = detail
                n_cache_confident += 1
        # NOTE: we accumulate confident-author details in to_cache_running and
        # write them ONCE at the end of label_location_month. Writing here
        # (every stream-flush, ~10x per task per month) caused NFS-mediated
        # SQLite lock contention under %25 concurrency. Per-month write keeps
        # the DB consistent at the cost of losing this task's cache
        # contribution if the task crashes after a stream flush but before
        # end-of-month.

        # Stream the longest now-resolvable prefix to disk.
        new_last, rows_w = _flush_completed_prefix(
            curated_csv_path, out_file, last_processed, idx_map, detail_by_author, write_mode
        )
        if rows_w > 0:
            last_processed = new_last
            write_mode = "a"
        n_stream_flushes += 1
        log_report(
            report_file_path,
            f"[stream-flush] {stem}: trigger={trigger} inferred={n_inferred:,} rows_written={rows_w:,} "
            f"last_processed={last_processed} files_done={files_done}/{len(files_to_scan)}",
        )
        return rows_w, n_inferred

    # Drive the generator with files_to_scan + per_file_targets so we open
    # only the spiral files that have at least one author needing them. The
    # target month's file is always in per_file_targets so dedup can run; the
    # cache stores undeduped counts so cross-group reuse works.
    raw_counts_state: Dict[str, Dict[str, int]] = {}
    raw_seen_state: Dict[str, int] = {a: 0 for a in remaining_authors_set}
    raw_overlap_counts_state: Dict[str, Dict[str, int]] = {}
    raw_overlap_seen_state: Dict[str, int] = {}
    pending_file_cache_rows: List[Tuple[str, str, Dict[str, int], int]] = []
    n_file_cache_rows_written = 0

    def _persist_caches() -> None:
        """Write accumulated confident label details (single label DB) and
        per-(author, file) scan rows (year-sharded file_counts DBs), then clear
        the buffers. Called periodically mid-month to bound memory and bank
        progress against a wall-time kill, and once more at end-of-month."""
        nonlocal n_file_cache_rows_written
        if to_cache_running:
            cache_put_locations(LABEL_DB_PATH, to_cache_running)
            cache_put_location_details(LABEL_DB_PATH, to_cache_running)
            to_cache_running.clear()
        if pending_file_cache_rows:
            n_file_cache_rows_written += cache_put_author_file_counts_sharded(
                str(CACHE_DIR), RAW_TYPE, pending_file_cache_rows
            )
            pending_file_cache_rows.clear()
    # Per-author remaining quota: total sampling cap for this run is
    # max_items_per_author MINUS whatever the persistent cache already
    # contributed for this author. Restores the original semantics that the
    # location LR was trained on (one author -> at most max_items_per_author
    # samples across the entire scan, cumulative across spiral files and
    # cumulative across all prior runs).
    remaining_quota_per_author = {
        a: max(0, max_items_per_author - int(cached_seen_by_author.get(a, 0)))
        for a in remaining_authors_set
    }
    for yld in iter_author_feature_map_streaming(
        raw_files=files_to_scan,
        target_authors=per_file_targets if per_file_targets else remaining_authors_set,
        type_=RAW_TYPE,
        max_items_per_author=max_items_per_author,
        n_scan_workers=n_scan_workers,
        word_vocab=word_vocab,
        curated_seen_ids=curated_seen_ids,
        target_month_basenames=target_month_basenames,
        remaining_quota_per_author=remaining_quota_per_author,
    ):
        per_file_deltas = yld.per_file_deltas
        raw_counts_state = yld.cumulative_counts
        raw_seen_state = yld.cumulative_seen
        raw_overlap_counts_state = yld.cumulative_overlap_counts
        raw_overlap_seen_state = yld.cumulative_overlap_seen
        files_remaining = yld.files_remaining
        files_done += len(yld.files_just_done)
        files_since_flush += len(yld.files_just_done)

        # Progress heartbeat for the otherwise-silent raw scan (every ~10% of
        # the spiral). Reports files scanned, authors collected from raw so far,
        # the unflushed cache backlog, and elapsed scan minutes -- enough to
        # estimate per-month completion against the wall-time limit.
        if files_to_scan and files_done >= next_scan_log:
            pct = 100.0 * files_done / len(files_to_scan)
            log_report(
                report_file_path,
                f"[scan-progress] {stem}: {files_done}/{len(files_to_scan)} files ({pct:.0f}%) "
                f"raw_for={len(raw_counts_state):,} authors "
                f"pending_cache_rows={len(pending_file_cache_rows):,} "
                f"elapsed={(time.time() - raw_scan_start) / 60:.1f}m",
            )
            while next_scan_log <= files_done:
                next_scan_log += scan_log_every

        # Persist per-(author, file) rows for everything actually found.
        # "scanned-but-empty" rows (seen=0) are deliberately NOT cached:
        # next month's overlapping spiral may re-open files where this
        # author wasn't present, but the file gets opened anyway for any
        # other target author needing it -- the marginal cost is a fast
        # per-line author-set lookup. Skipping seen=0 rows reduces the
        # cache to ~3x smaller without changing scan correctness.
        for basename, file_payload in per_file_deltas.items():
            file_counts, file_seen, _file_overlap_counts, _file_overlap_seen = file_payload
            for author, counts in file_counts.items():
                seen = int(file_seen.get(author, 0))
                if seen <= 0:
                    continue
                pending_file_cache_rows.append((
                    author,
                    basename,
                    counts,
                    seen,
                ))
        # Periodically persist the accumulated per-file rows (and any confident
        # label details) once a buffer grows past CACHE_FLUSH_ROWS. This bounds
        # peak memory and banks scan progress mid-month, so a wall-time kill no
        # longer discards the whole month. Writes go to the year-sharded
        # file_counts DBs (contention spread across years) plus the single label
        # DB; DELETE-journal POSIX locks keep concurrent NFS writers safe.
        if len(pending_file_cache_rows) >= CACHE_FLUSH_ROWS or len(to_cache_running) >= CACHE_FLUSH_ROWS:
            _persist_caches()

        if files_since_flush < FLUSH_EVERY_N_RAW and files_remaining > 0:
            continue

        # Pick saturated authors not yet inferred. "Saturated" includes both
        # this-month newly-saturated AND fully_cached_authors whose cached
        # seen already exceeds the cap. Effective seen = cached + this-month.
        saturated = sorted(
            a for a in remaining_authors_set
            if a not in inferred_authors
            and (
                int(cached_seen_by_author.get(a, 0)) + int(raw_seen_state.get(a, 0))
                >= max_items_per_author
            )
        )
        if saturated:
            _infer_and_flush(saturated, trigger=f"saturated@{files_done}files")
        files_since_flush = 0

    log_report(
        report_file_path,
        f"[scan] {stem}: collected_raw_for={len(raw_counts_state):,} authors in {(time.time() - raw_scan_start)/60:.2f} minutes "
        f"pending_file_cache_rows={len(pending_file_cache_rows):,}",
    )

    # Final inference for any remaining_authors not yet processed (the
    # non-saturated tail, whose seen_count never reached max_items_per_author).
    # The merge here folds local_counts (this-month curated seed) + raw_counts_state
    # (this-month raw-zst scan delta) + cached_counts_by_author (prior-month scans
    # from the persistent scan-state cache), then SUBTRACTS the per-post
    # target-month overlap so curated and raw don't double-count the same post.
    author_to_counts, author_seen = _merge_feature_maps(local_counts, local_seen, raw_counts_state, raw_seen_state, remaining_authors)
    for author, ccnt in cached_counts_by_author.items():
        if author not in author_to_counts:
            author_to_counts[author] = {}
        dst = author_to_counts[author]
        for k, v in ccnt.items():
            dst[k] = dst.get(k, 0) + int(v)
    for author in remaining_authors:
        author_seen[author] = int(author_seen.get(author, 0)) + int(cached_seen_by_author.get(author, 0))
    # Dedup: subtract the target-month curated/raw overlap from each author's
    # combined counts and seen. Net effect: each post counted exactly once
    # across (curated, raw) sources for the target month.
    for author, ov in raw_overlap_counts_state.items():
        if not ov:
            continue
        dst = author_to_counts.get(author)
        if dst is None:
            continue
        for k, v in ov.items():
            new_val = dst.get(k, 0) - int(v)
            if new_val > 0:
                dst[k] = new_val
            elif k in dst:
                del dst[k]
    for author, s in raw_overlap_seen_state.items():
        if s and author in author_seen:
            author_seen[author] = max(0, int(author_seen[author]) - int(s))

    leftover_authors = [a for a in remaining_authors if a not in inferred_authors]
    for i in range(0, len(leftover_authors), batch_size):
        chunk = leftover_authors[i:i + batch_size]
        batch_counts = [author_to_counts.get(a, {}) for a in chunk]
        batch_seen = [int(author_seen.get(a, 0)) for a in chunk]
        batch_results = infer_locations_for_batch(batch_counts, batch_seen, bundle)
        for author, detail in zip(chunk, batch_results):
            detail_by_author[author] = detail
            inferred_authors.add(author)
            seen = int(detail.get("seen_count") or 0)
            if seen < MIN_SAMPLES_FOR_CACHE:
                n_cache_skipped_lowsamples += 1
                continue
            if detail.get("location") == UNKNOWN_LABEL:
                n_cache_skipped_lowconf += 1
                continue
            to_cache_running[author] = detail
            n_cache_confident += 1

    # Final persist of any remaining confident labels + per-file scan rows.
    # Mid-month _persist_caches calls have already banked most of the work to
    # the single label DB and the year-sharded file_counts DBs.
    _persist_caches()

    n_dedup_authors = sum(1 for v in raw_overlap_seen_state.values() if v > 0)
    n_dedup_items = sum(int(v) for v in raw_overlap_seen_state.values())
    log_report(
        report_file_path,
        f"[cache] {stem}: newly_labeled={len(remaining_authors):,} cached_confident={n_cache_confident:,} "
        f"skipped_lowconf_or_unknown={n_cache_skipped_lowconf:,} skipped_lowsamples={n_cache_skipped_lowsamples:,} "
        f"stream_flushes={n_stream_flushes} file_cache_rows_written={n_file_cache_rows_written:,} "
        f"dedup_authors={n_dedup_authors:,} dedup_items={n_dedup_items:,} "
        f"top_conf>={TOP_CONF_THRESHOLD} reg_margin>={REG_CONF_MARGIN} state_margin>={STA_CONF_MARGIN} min_samples_cache={MIN_SAMPLES_FOR_CACHE}",
    )

    # Final write: anything past last_processed gets written here. By this point
    # every author is in detail_by_author, so _stream_write_output will resolve
    # every row.
    _stream_write_output(curated_csv_path, out_file, last_processed, idx_map, detail_by_author, write_mode=write_mode)

    elapsed = (time.time() - start) / 60
    covered = sum(1 for a in remaining_authors if author_seen.get(a, 0) > 0)
    log_report(
        report_file_path,
        f"[done] user location labeling for {stem}: rows={remaining_row_count:,} authors={n_total:,} cached={n_cached:,} scanned_raw={len(remaining_authors):,} "
        f"covered={covered:,} minutes={elapsed:.2f}",
    )
    return (stem, remaining_row_count, n_total, len(remaining_authors))


def label_location_parallel() -> None:
    array_idx = getattr(args, "array", None)
    if array_idx is not None:
        try:
            slot = int(array_idx)
        except (ValueError, TypeError):
            log_report(report_file_path, f"[error] --array '{array_idx}' is not an integer; aborting")
            raise SystemExit(2)
        # Optional spread schedule: --array-order remaps the SLURM array slot to
        # a file_list index, so the months running concurrently under %cap are
        # chronologically far apart. Disjoint spirals route their cache writes to
        # different year-sharded file_counts DBs (less SQLite lock contention)
        # and avoid two concurrent tasks decompressing the same overlapping raw
        # .zst files. Accepts a path to a file of indices or an inline
        # comma/whitespace-separated list; absent -> slot indexes file_list
        # directly (original behavior).
        order_spec = getattr(args, "array_order", None)
        if order_spec:
            raw = order_spec
            if os.path.exists(order_spec):
                with open(order_spec) as fh:
                    raw = fh.read()
            try:
                order = [int(tok) for tok in re.split(r"[,\s]+", raw.strip()) if tok]
            except ValueError:
                log_report(report_file_path, f"[error] --array-order '{order_spec}' is not a list of ints; aborting")
                raise SystemExit(2)
            if not (0 <= slot < len(order)):
                log_report(
                    report_file_path,
                    f"[error] --array slot {slot} out of range for --array-order (len={len(order)}); aborting",
                )
                raise SystemExit(2)
            idx = order[slot]
        else:
            idx = slot
        if not (0 <= idx < len(file_list)):
            log_report(
                report_file_path,
                f"[error] resolved index {idx} (slot {slot}) out of range (file_list size={len(file_list)}); aborting",
            )
            raise SystemExit(2)
        try:
            label_location_month(file_list[idx])
        except Exception as e:
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            log_report(
                report_file_path,
                f"[error] task {idx} ({Path(file_list[idx]).name}) failed: {e}\n{tb}",
            )
            raise
        return

    _slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
    max_workers = _slurm_cpus if _slurm_cpus > 0 else min(2, os.cpu_count() or 1)
    log_report(report_file_path, f"Using {max_workers} processes for parallel month processing.")

    total_rows = 0
    total_authors = 0
    total_raw = 0
    pending = {}
    started = time.time()
    last_heartbeat = started

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for f in file_list:
            fut = ex.submit(label_location_month, f)
            pending[fut] = f

        while pending:
            done, not_done = wait(list(pending.keys()), timeout=60, return_when=FIRST_COMPLETED)
            if not done:
                now = time.time()
                if now - last_heartbeat >= PROGRESS_HEARTBEAT_SECONDS:
                    completed = len(file_list) - len(pending)
                    elapsed_min = (now - started) / 60.0
                    log_report(
                        report_file_path,
                        f"[progress] completed_months={completed:,}/{len(file_list):,} pending={len(pending):,} elapsed_minutes={elapsed_min:.2f}",
                    )
                    last_heartbeat = now
                continue

            for fut in done:
                src = pending.pop(fut)
                try:
                    _, rows, n_auth, n_raw = fut.result()
                    total_rows += rows
                    total_authors += n_auth
                    total_raw += n_raw
                except Exception as e:
                    tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                    log_report(report_file_path, f"[error] month failed for {src}: {e}\n{tb}")

    log_report(report_file_path, f"[summary] total rows written: {total_rows:,} total authors: {total_authors:,} raw-scanned authors: {total_raw:,}")


if __name__ == "__main__":
    overall = time.time()
    try:
        label_location_parallel()
    except Exception as e:
        log_report(report_file_path, f"Fatal error during location labeling: {e}")
        raise
    finally:
        mins = (time.time() - overall) / 60
        log_report(report_file_path, f"Location labeling finished in {mins:.2f} minutes.")
