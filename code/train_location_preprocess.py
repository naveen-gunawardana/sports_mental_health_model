### Imports

from __future__ import annotations

import csv
csv.field_size_limit(2**31 - 1) # Increase the field size limit to handle larger fields
import json
import math
import os
import pickle
import re
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from cli import get_args, DATA_DIR, MODELS_DIR
from scipy import sparse
from scipy.sparse import save_npz
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from utils import prepare_splits, resolve_word_feature_src

### Argument Handling

# Extract and transform CLI arguments 
args = get_args()
type_ = args.type

# Logging
VERBOSITY = int(os.environ.get("VERBOSITY", "1"))
def log(msg: str, level: int = 1, stream=None):
    if level <= VERBOSITY:
        print(msg, file=stream)

### Path Handling

# find the feature set input files
if not args.input:
    input_path = DATA_DIR / "data_reddit_location"
else:
    input_path = args.input
SUBS_JSONL = os.path.join(input_path, "subreddit_counts.jsonl")
HOURS_JSONL = os.path.join(input_path, "hour_counts.jsonl")
VOCAB_FILE_COMMENTS = os.path.join(input_path, "vocab_counts_comments.jsonl")
VOCAB_FILE_SUBMISSIONS = os.path.join(input_path, "vocab_counts_submissions.jsonl")

# where the raw input files are

# Output path handling
if not args.output:
    PREPROC_PATH = os.environ.get(
        "PREPROC_PATH",
        os.path.join(MODELS_DIR, "label_location", "preprocessed_streaming"),
    )
else:
    PREPROC_PATH = args.output
os.makedirs(PREPROC_PATH, exist_ok=True)


SPLIT_DIR = os.environ.get("SPLIT_DIR", os.path.join(MODELS_DIR, "train_location_data_split")) # where the train/valid/test data split is stored for reproducibility
SAVE_PREPROCESSOR = os.environ.get("SAVE_PREPROCESSOR", "1") == "1"

### Preprocessing Hyperparameters

# Relative weight of feature classes
STRUCT_WORD_WEIGHT = float(os.environ.get("STRUCT_WORD_WEIGHT", "1.0"))
STRUCT_SUB_WEIGHT = float(os.environ.get("STRUCT_SUB_WEIGHT", "1.0"))
STRUCT_HOUR_WEIGHT = float(os.environ.get("STRUCT_HOUR_WEIGHT", "1.0"))

# whether performance on a masked dataset removing direct references to the user's location in their texts is compared with the full sample. On by default
MASK_TIER1_WORDS = os.environ.get("MASK_TIER1_WORDS", "1") == "1" 

## Word Features

# which "type" dataset the word features come from. Can be set via CLI.
# NOTE: Only relevant for word features. subreddit and timestamp featurs automatically set to "all"
WORD_FEATURE_SRC = resolve_word_feature_src(type_)

# Word feature selection
# NOTE: freq: tf-idf; mi: 
WORD_SELECTOR = os.environ.get("WORD_SELECTOR", "freq").strip().lower()  # freq | mi
if WORD_SELECTOR not in {"freq", "mi"}:
    raise ValueError("WORD_SELECTOR must be one of: freq, mi")

# word statistic to use for word feature processing
# df: document frequency; count: term frequency; defaults to df
WORD_STAT = os.environ.get("WORD_STAT", "df").strip().lower()  # df | count
if WORD_STAT not in {"df", "count"}:
    raise ValueError("WORD_STAT must be one of: df, count")

# word feature min and max frequency/count parameters
WORD_MIN_DF = int(os.environ.get("WORD_MIN_DF", "2"))
WORD_MIN_TOTAL_COUNT = int(os.environ.get("WORD_MIN_TOTAL_COUNT", "2"))
WORD_CANDIDATE_POOL = int(os.environ.get("WORD_CANDIDATE_POOL", "250000"))
WORD_TOP_K = int(os.environ.get("WORD_TOP_K", "50000"))

# word feature processing knobs
WORD_USE_IDF = os.environ.get("WORD_USE_IDF", "1") == "1" # tf-idf transformation. On by default.
WORD_SUBLINEAR_TF = os.environ.get("WORD_SUBLINEAR_TF", "1") == "1" # sublinear version of tf-idf. On by default
WORD_SMOOTH_IDF = os.environ.get("WORD_SMOOTH_IDF", "1") == "1" # tf-idf smoothing. On by default.
WORD_NORM = os.environ.get("WORD_NORM", "l2").strip().lower()  # normalization: l1 | l2 | none. l2 by default.
if WORD_NORM == "none":
    WORD_NORM_OPT = None
elif WORD_NORM in {"l1", "l2"}:
    WORD_NORM_OPT = WORD_NORM
else:
    raise ValueError("WORD_NORM must be one of: l1, l2, none")

## Structured features

# NOTE: includes SUB=subreddit frequency, HOUR=timestamp bin frequency

# feature mode
STRUCT_SUB_MODE = os.environ.get("STRUCT_SUB_MODE", "log1p_l1").strip().lower()  # log1p_l1 | l1 | binary_l1 | tfidf
STRUCT_HOUR_MODE = os.environ.get("STRUCT_HOUR_MODE", "l1").strip().lower()      # log1p_l1 | l1 | binary_l1
for _mode_name, _mode in [("STRUCT_SUB_MODE", STRUCT_SUB_MODE), ("STRUCT_HOUR_MODE", STRUCT_HOUR_MODE)]:
    if _mode not in {"log1p_l1", "l1", "binary_l1", "tfidf"}:
        raise ValueError(f"{_mode_name} must be one of: log1p_l1, l1, binary_l1, tfidf")

# feature count/frequency min/max
SUBREDDIT_TOP_K = int(os.environ.get("SUBREDDIT_TOP_K", "20000"))
SUBREDDIT_MIN_DF = int(os.environ.get("SUBREDDIT_MIN_DF", "2"))

### Data Mapping

TOP_US = "US"
TOP_NON_US = "NON_US"
TOP_UNKNOWN = "UNKNOWN"
STATE_UNKNOWN = "UNKNOWN"
REGION_UNKNOWN = "UNKNOWN"

US_STATE_TO_CODE = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "district of columbia": "DC",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID", "illinois": "IL",
    "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA",
    "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
    "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC", "south dakota": "SD",
    "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA",
    "washington": "WA", "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
}

# Maps countries to one of five regions: AMERICAS, EUROPE, ASIA_OCEANIA, AFRICA and REGION_UNKNOWN
# NOTE: Antarctica maps to REGION_UNKNOWN, as it is often used disingenuously in natural discourse.
REGION_BY_COUNTRY = {
    # Americas
    "antigua and barbuda": "AMERICAS", "argentina": "AMERICAS", "bahamas": "AMERICAS",
    "barbados": "AMERICAS", "belize": "AMERICAS", "bermuda": "AMERICAS", "bolivia": "AMERICAS",
    "brazil": "AMERICAS", "canada": "AMERICAS", "cayman islands": "AMERICAS", "chile": "AMERICAS",
    "colombia": "AMERICAS", "costa rica": "AMERICAS", "cuba": "AMERICAS", "dominica": "AMERICAS",
    "dominican republic": "AMERICAS", "ecuador": "AMERICAS", "el salvador": "AMERICAS", "grenada": "AMERICAS",
    "guadeloupe": "AMERICAS", "guatemala": "AMERICAS", "guyana": "AMERICAS", "haiti": "AMERICAS",
    "honduras": "AMERICAS", "jamaica": "AMERICAS", "martinique": "AMERICAS", "mexico": "AMERICAS",
    "netherlands antilles": "AMERICAS", "nicaragua": "AMERICAS", "panama": "AMERICAS", "paraguay": "AMERICAS",
    "peru": "AMERICAS", "puerto rico": "AMERICAS", "saint kitts and nevis": "AMERICAS",
    "saint lucia": "AMERICAS", "saint vincent and the grenadines": "AMERICAS", "suriname": "AMERICAS",
    "trinidad and tobago": "AMERICAS", "uruguay": "AMERICAS", "venezuela": "AMERICAS",
    "virgin islands": "AMERICAS",
    # Europe
    "albania": "EUROPE", "andorra": "EUROPE", "armenia": "EUROPE", "austria": "EUROPE",
    "azerbaijan": "EUROPE", "belarus": "EUROPE", "belgium": "EUROPE", "bosnia and herzegovina": "EUROPE",
    "bulgaria": "EUROPE", "croatia": "EUROPE", "cyprus": "EUROPE", "czech republic": "EUROPE",
    "czechia": "EUROPE", "denmark": "EUROPE", "estonia": "EUROPE", "faroe islands": "EUROPE",
    "finland": "EUROPE", "france": "EUROPE", "georgia": "EUROPE", "germany": "EUROPE",
    "gibraltar": "EUROPE", "greece": "EUROPE", "greenland": "EUROPE", "guernsey": "EUROPE",
    "hungary": "EUROPE", "iceland": "EUROPE", "ireland": "EUROPE", "isle of man": "EUROPE",
    "italy": "EUROPE", "jersey": "EUROPE", "kosovo": "EUROPE", "latvia": "EUROPE",
    "liechtenstein": "EUROPE", "lithuania": "EUROPE", "luxembourg": "EUROPE", "macedonia": "EUROPE",
    "malta": "EUROPE", "moldova": "EUROPE", "monaco": "EUROPE", "montenegro": "EUROPE",
    "netherlands": "EUROPE", "north macedonia": "EUROPE", "norway": "EUROPE", "poland": "EUROPE",
    "portugal": "EUROPE", "romania": "EUROPE", "russia": "EUROPE", "san marino": "EUROPE",
    "serbia": "EUROPE", "slovakia": "EUROPE", "slovenia": "EUROPE", "spain": "EUROPE",
    "svalbard and jan mayen": "EUROPE", "sweden": "EUROPE", "switzerland": "EUROPE", "turkmenistan": "EUROPE",
    "uk": "EUROPE", "u.k.": "EUROPE", "ukraine": "EUROPE", "united kingdom": "EUROPE",
    "england": "EUROPE", "scotland": "EUROPE", "wales": "EUROPE",
    # Africa
    "algeria": "AFRICA", "angola": "AFRICA", "benin": "AFRICA", "botswana": "AFRICA",
    "burkina faso": "AFRICA", "burundi": "AFRICA", "cameroon": "AFRICA", "cape verde": "AFRICA",
    "central african republic": "AFRICA", "chad": "AFRICA", "comoros": "AFRICA", "congo": "AFRICA",
    "djibouti": "AFRICA", "egypt": "AFRICA", "equatorial guinea": "AFRICA", "eritrea": "AFRICA",
    "ethiopia": "AFRICA", "gabon": "AFRICA", "gambia": "AFRICA", "ghana": "AFRICA",
    "guinea": "AFRICA", "guinea-bissau": "AFRICA", "ivory coast": "AFRICA", "cote d'ivoire": "AFRICA",
    "kenya": "AFRICA", "lesotho": "AFRICA", "liberia": "AFRICA", "libya": "AFRICA",
    "madagascar": "AFRICA", "malawi": "AFRICA", "mali": "AFRICA", "mauritania": "AFRICA",
    "mauritius": "AFRICA", "mayotte": "AFRICA", "morocco": "AFRICA", "mozambique": "AFRICA",
    "namibia": "AFRICA", "niger": "AFRICA", "nigeria": "AFRICA", "reunion": "AFRICA",
    "rwanda": "AFRICA", "senegal": "AFRICA", "seychelles": "AFRICA", "sierra leone": "AFRICA",
    "somalia": "AFRICA", "south africa": "AFRICA", "sudan": "AFRICA", "swaziland": "AFRICA",
    "eswatini": "AFRICA", "tanzania": "AFRICA", "togo": "AFRICA", "tunisia": "AFRICA",
    "uganda": "AFRICA", "western sahara": "AFRICA", "zambia": "AFRICA", "zimbabwe": "AFRICA",
    # Asia / Oceania
    "afghanistan": "ASIA_OCEANIA", "australia": "ASIA_OCEANIA", "bahrain": "ASIA_OCEANIA",
    "bangladesh": "ASIA_OCEANIA", "bhutan": "ASIA_OCEANIA", "brunei": "ASIA_OCEANIA", "cambodia": "ASIA_OCEANIA",
    "china": "ASIA_OCEANIA", "christmas island": "ASIA_OCEANIA", "fiji": "ASIA_OCEANIA",
    "french polynesia": "ASIA_OCEANIA", "guam": "ASIA_OCEANIA", "hong kong": "ASIA_OCEANIA",
    "india": "ASIA_OCEANIA", "indonesia": "ASIA_OCEANIA", "iran": "ASIA_OCEANIA", "iraq": "ASIA_OCEANIA",
    "israel": "ASIA_OCEANIA", "japan": "ASIA_OCEANIA", "jordan": "ASIA_OCEANIA", "kazakhstan": "ASIA_OCEANIA",
    "kiribati": "ASIA_OCEANIA", "korea": "ASIA_OCEANIA", "korea, republic of": "ASIA_OCEANIA",
    "south korea": "ASIA_OCEANIA", "north korea": "ASIA_OCEANIA", "kuwait": "ASIA_OCEANIA",
    "kyrgyzstan": "ASIA_OCEANIA", "laos": "ASIA_OCEANIA", "lebanon": "ASIA_OCEANIA", "macau": "ASIA_OCEANIA",
    "malaysia": "ASIA_OCEANIA", "maldives": "ASIA_OCEANIA", "marshall islands": "ASIA_OCEANIA",
    "micronesia": "ASIA_OCEANIA", "mongolia": "ASIA_OCEANIA", "myanmar": "ASIA_OCEANIA", "nauru": "ASIA_OCEANIA",
    "nepal": "ASIA_OCEANIA", "new caledonia": "ASIA_OCEANIA", "new zealand": "ASIA_OCEANIA",
    "norfolk island": "ASIA_OCEANIA", "northern mariana islands": "ASIA_OCEANIA", "oman": "ASIA_OCEANIA",
    "pakistan": "ASIA_OCEANIA", "palau": "ASIA_OCEANIA", "papua new guinea": "ASIA_OCEANIA",
    "philippines": "ASIA_OCEANIA", "qatar": "ASIA_OCEANIA", "saudi arabia": "ASIA_OCEANIA",
    "singapore": "ASIA_OCEANIA", "solomon islands": "ASIA_OCEANIA", "sri lanka": "ASIA_OCEANIA",
    "syria": "ASIA_OCEANIA", "taiwan": "ASIA_OCEANIA", "tajikistan": "ASIA_OCEANIA", "thailand": "ASIA_OCEANIA",
    "timor-leste": "ASIA_OCEANIA", "tokelau": "ASIA_OCEANIA", "tonga": "ASIA_OCEANIA", "turkey": "ASIA_OCEANIA",
    "tuvalu": "ASIA_OCEANIA", "united arab emirates": "ASIA_OCEANIA", "uzbekistan": "ASIA_OCEANIA",
    "vanuatu": "ASIA_OCEANIA", "vietnam": "ASIA_OCEANIA", "yemen": "ASIA_OCEANIA",
}

## Aliases for the labels, used in masking

US_COUNTRY_ALIASES = {
    "usa", "u.s.a", "u.s.a.", "us", "u.s", "u.s.",
    "united states", "united states of america", "america",
}

COUNTRY_ALIAS_MAP: Dict[str, Set[str]] = {
    "united states": set(US_COUNTRY_ALIASES),
    "united states of america": set(US_COUNTRY_ALIASES),
    "us": set(US_COUNTRY_ALIASES),
    "usa": set(US_COUNTRY_ALIASES),
    "united kingdom": {"uk", "u.k", "u.k.", "britain", "great britain", "gb", "g.b.", "england"},
    "uk": {"united kingdom", "u.k", "u.k.", "britain", "great britain", "gb", "g.b.", "england"},
    "uae": {"united arab emirates", "u.a.e", "u.a.e.", "emirates"},
    "united arab emirates": {"uae", "u.a.e", "u.a.e.", "emirates"},
    "south korea": {"republic of korea", "korea, republic of", "rok"},
    "north korea": {"dprk", "democratic people's republic of korea"},
    "czechia": {"czech republic"},
    "czech republic": {"czechia"},
    "russia": {"russian federation"},
}

STATE_ALIAS_MAP: Dict[str, Set[str]] = {
    "district of columbia": {"dc", "d.c.", "washington dc", "washington d.c."},
}

### Classes and Helper Functions

@dataclass
class UserLabel:
    user_id: str
    top_label: str
    state_label: str
    region_label: str
    country_raw: str
    state_raw: str


@dataclass
class WordSelectionMetadata:
    selector: str
    stat: str
    min_df: int
    min_total_count: int
    candidate_pool: int
    top_k: int
    selected_size: int


@dataclass
class PreprocessMetadata:
    word_feature_src: str
    word_selector: str
    word_stat: str
    word_min_df: int
    word_min_total_count: int
    word_candidate_pool: int
    word_top_k: int
    subreddit_top_k: int
    struct_sub_mode: str
    struct_hour_mode: str
    n_users_total: int
    n_users_feature_aligned: int
    n_train: int
    n_val: int
    n_test: int
    n_word_features: int
    n_struct_features: int
    n_state_classes: int
    n_region_classes: int
    has_masked_words: bool
    n_word_features_masked: int

# Optional fast JSON
try:
    import orjson as _fastjson  # type: ignore

    def _json_loads(b: bytes):
        return _fastjson.loads(b)
except Exception:
    def _json_loads(b: bytes):
        return json.loads(b)



def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _term_tokens(term: str) -> List[str]:
    norm = normalize_text(term)
    if not norm:
        return []
    parts = re.split(r"[^a-z0-9]+", norm)
    return [p for p in parts if p]

def get_user_mask_terms(label: UserLabel) -> Set[str]:
    terms: Set[str] = set()

    def add_term(term: str):
        norm = normalize_text(term)
        if norm:
            terms.add(norm)
        for tok in _term_tokens(term):
            terms.add(tok)

    def add_aliases(term: str, alias_map: Dict[str, Set[str]]):
        norm = normalize_text(term)
        if not norm:
            return
        add_term(norm)
        for alias in alias_map.get(norm, set()):
            add_term(alias)

    if label.top_label == TOP_US:
        for term in US_COUNTRY_ALIASES:
            add_term(term)
        if label.state_raw:
            add_aliases(label.state_raw, STATE_ALIAS_MAP)
            state_code = US_STATE_TO_CODE.get(label.state_raw)
            if state_code:
                add_term(state_code.lower())
    elif label.top_label == TOP_NON_US and label.country_raw:
        add_aliases(label.country_raw, COUNTRY_ALIAS_MAP)

    return {t for t in terms if t}

def build_masked_word_count_matrix(
    users: List[str],
    selected_words: List[str],
    user_to_label: Dict[str, UserLabel],
) -> sparse.csr_matrix:
    user_index = {u: i for i, u in enumerate(users)}
    word_index = {normalize_text(w): i for i, w in enumerate(selected_words)}
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    t0 = time.time()
    matched = 0
    total_masked_nnz = 0
    n_bad_int = 0
    first_bad_sample: Optional[str] = None
    for uid, raw in iter_vocab_rows_by_source(set(users), WORD_FEATURE_SRC):
        row = user_index.get(uid)
        if row is None:
            continue
        matched += 1
        mask_terms = get_user_mask_terms(user_to_label[uid])
        for k, v in raw.items():
            norm_word = normalize_text(str(k))
            col = word_index.get(norm_word)
            if col is None:
                continue
            try:
                iv = int(v)
            except Exception as e:
                n_bad_int += 1
                if first_bad_sample is None:
                    first_bad_sample = f"uid={uid!r} key={k!r} value={v!r}: {e}"
                continue
            if iv <= 0:
                continue
            if mask_terms and norm_word in mask_terms:
                total_masked_nnz += 1
                continue
            rows.append(row)
            cols.append(col)
            data.append(float(iv))
        if matched and matched % 25000 == 0:
            log(f"[words_masked] built raw rows for {matched:,} users", 1)

    X = sparse.csr_matrix(
        (np.asarray(data, dtype=np.float32), (rows, cols)),
        shape=(len(users), len(selected_words)),
        dtype=np.float32,
    )
    X.sum_duplicates()
    X.sort_indices()
    if n_bad_int:
        log(f"[words_masked] skipped {n_bad_int:,} non-integer count values; first: {first_bad_sample}", 1)
    log(
        f"[words_masked] raw sparse matrix shape={X.shape} nnz={X.nnz:,} masked_terms_removed={total_masked_nnz:,} "
        f"elapsed={(time.time() - t0)/60:.2f} min",
        1,
    )
    return X

def country_to_region(country_raw: str) -> str:
    country = normalize_text(country_raw)
    if not country:
        return REGION_UNKNOWN
    if country in {"united states", "united states of america", "usa", "u.s.", "us"}:
        return REGION_UNKNOWN
    if country == "antarctica":
        return REGION_UNKNOWN
    return REGION_BY_COUNTRY.get(country, REGION_UNKNOWN)

def _safe_log(x: float) -> float:
    return math.log(x) if x > 0.0 else 0.0

### Feature Loading

def load_user_labels(labels_csv: str) -> Dict[str, UserLabel]:
    user_to_label: Dict[str, UserLabel] = {}
    with open(labels_csv, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            if len(row) < 5:
                continue
            uid = normalize_text(row[1])
            if not uid:
                continue
            state_raw = normalize_text(row[3])
            country_raw = normalize_text(row[4])

            if state_raw in US_STATE_TO_CODE:
                top_label = TOP_US
                state_label = US_STATE_TO_CODE[state_raw]
                region_label = REGION_UNKNOWN
            elif "united states" in country_raw or country_raw in {"usa", "u.s.", "us", "united states of america"}:
                top_label = TOP_US
                state_label = STATE_UNKNOWN
                region_label = REGION_UNKNOWN
            elif country_raw:
                top_label = TOP_NON_US
                state_label = STATE_UNKNOWN
                region_label = country_to_region(country_raw)
            else:
                top_label = TOP_UNKNOWN
                state_label = STATE_UNKNOWN
                region_label = REGION_UNKNOWN

            user_to_label[uid] = UserLabel(
                user_id=uid,
                top_label=top_label,
                state_label=state_label,
                region_label=region_label,
                country_raw=country_raw,
                state_raw=state_raw,
            )
    log(f"[labels] loaded {len(user_to_label):,} labeled users", 1)
    return user_to_label

def summarize_split(name: str, users: List[str], top_labels: List[str]):
    log(f"[{name}] users={len(users):,} top_classes={len(set(top_labels)):,}", 1)

def load_subreddit_counts(path: str, users_set: Set[str]) -> Dict[str, Dict[str, int]]:
    log("[subs] loading subreddit counts", 1)
    subs: Dict[str, Dict[str, int]] = {}
    n_bad_int = 0
    first_bad_sample: Optional[str] = None
    with open(path, "rb") as f:
        for i, line in enumerate(f):
            obj = _json_loads(line)
            uid = normalize_text(obj.get("author") or "")
            if uid not in users_set:
                continue
            raw = obj.get("subreddit_counts") or {}
            norm: Dict[str, int] = {}
            if isinstance(raw, dict):
                for k, v in raw.items():
                    try:
                        iv = int(v)
                    except Exception as e:
                        n_bad_int += 1
                        if first_bad_sample is None:
                            first_bad_sample = f"line={i} uid={uid!r} key={k!r} value={v!r}: {e}"
                        continue
                    if iv > 0:
                        norm[str(k)] = iv
            subs[uid] = norm
            if i and i % 50000 == 0:
                log(f"[subs] scanned {i:,} lines | matched {len(subs):,}", 1)
    if n_bad_int:
        log(f"[subs] skipped {n_bad_int:,} non-integer count values; first: {first_bad_sample}", 1)
    log(f"[subs] matched users: {len(subs):,}", 1)
    return subs

def load_hour_counts(path: str, users_set: Set[str]) -> Dict[str, Dict[str, int]]:
    log("[hours] loading hour counts", 1)
    hours: Dict[str, Dict[str, int]] = {}
    n_bad_int = 0
    first_bad_sample: Optional[str] = None
    with open(path, "rb") as f:
        for i, line in enumerate(f):
            obj = _json_loads(line)
            uid = normalize_text(obj.get("author") or "")
            if uid not in users_set:
                continue
            raw = obj.get("hour_counts") or obj.get("gmt_hour_counts") or {}
            norm: Dict[str, int] = {}
            if isinstance(raw, dict):
                for k, v in raw.items():
                    try:
                        hk = int(k)
                        iv = int(v)
                    except Exception as e:
                        n_bad_int += 1
                        if first_bad_sample is None:
                            first_bad_sample = f"line={i} uid={uid!r} key={k!r} value={v!r}: {e}"
                        continue
                    if 0 <= hk <= 23 and iv > 0:
                        norm[f"{hk:02d}"] = iv
            hours[uid] = norm
            if i and i % 50000 == 0:
                log(f"[hours] scanned {i:,} lines | matched {len(hours):,}", 1)
    if n_bad_int:
        log(f"[hours] skipped {n_bad_int:,} non-integer hour/count pairs; first: {first_bad_sample}", 1)
    log(f"[hours] matched users: {len(hours):,}", 1)
    return hours

def iter_vocab_rows(vocab_jsonl: str, users_set: Set[str]):
    with open(vocab_jsonl, "rb") as f:
        for i, line in enumerate(f):
            obj = _json_loads(line)
            uid = normalize_text(obj.get("author") or "")
            if uid not in users_set:
                continue
            raw = obj.get("vocab") or {}
            if not isinstance(raw, dict):
                raw = {}
            yield uid, raw
            if i and i % 100000 == 0:
                log(f"[vocab] scanned {i:,} rows in {os.path.basename(vocab_jsonl)}", 1)

def iter_vocab_rows_by_source(users_set: Set[str], word_feature_src: str):
    if word_feature_src == "comments":
        yield from iter_vocab_rows(VOCAB_FILE_COMMENTS, users_set)
        return
    if word_feature_src == "submissions":
        yield from iter_vocab_rows(VOCAB_FILE_SUBMISSIONS, users_set)
        return
    if word_feature_src == "all":
        tmp: Dict[str, Dict[str, int]] = {}
        n_bad_int = 0
        first_bad_sample: Optional[str] = None
        for path in (VOCAB_FILE_COMMENTS, VOCAB_FILE_SUBMISSIONS):
            for uid, raw in iter_vocab_rows(path, users_set):
                tgt = tmp.setdefault(uid, {})
                for k, v in raw.items():
                    try:
                        iv = int(v)
                    except Exception as e:
                        n_bad_int += 1
                        if first_bad_sample is None:
                            first_bad_sample = f"path={os.path.basename(path)} uid={uid!r} key={k!r} value={v!r}: {e}"
                        continue
                    if iv > 0:
                        tgt[str(k)] = tgt.get(str(k), 0) + iv
        if n_bad_int:
            log(f"[vocab_all] skipped {n_bad_int:,} non-integer count values; first: {first_bad_sample}", 1)
        for uid, raw in tmp.items():
            yield uid, raw
        return
    raise ValueError("word_feature_src must be one of: comments, submissions, all")

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
        elif mode == "l1":
            tv = fv
        elif mode == "binary_l1":
            tv = 1.0
        elif mode == "tfidf":
            tv = fv
        else:
            raise ValueError(f"unsupported mode: {mode}")
        out[f"{prefix}{k}"] = tv
        total += tv
    if mode in {"log1p_l1", "l1", "binary_l1"} and total > 0.0:
        scale = weight / total
        out = {k: v * scale for k, v in out.items()}
    elif mode == "tfidf" and weight != 1.0:
        out = {k: v * weight for k, v in out.items()}
    return out

### Word Feature Selection

def compute_binary_mi(
    present_counts_by_label: Dict[str, Counter],
    label_doc_counts: Counter,
    candidate_words: Set[str],
    n_docs: int,
) -> Dict[str, float]:
    labels = list(label_doc_counts.keys())
    scores: Dict[str, float] = {}
    for word in candidate_words:
        n1dot = sum(present_counts_by_label[label].get(word, 0) for label in labels)
        n0dot = n_docs - n1dot
        if n1dot <= 0 or n_docs <= 0:
            scores[word] = 0.0
            continue
        score = 0.0
        for label in labels:
            n11 = present_counts_by_label[label].get(word, 0)
            n01 = label_doc_counts[label] - n11
            n10 = n1dot - n11
            n00 = n0dot - n01
            # contribution helper
            for nij, ni_dot, n_dot_j in (
                (n11, n1dot, label_doc_counts[label]),
                (n01, n0dot, label_doc_counts[label]),
                (n10, n1dot, n_docs - label_doc_counts[label]),
                (n00, n0dot, n_docs - label_doc_counts[label]),
            ):
                if nij > 0 and ni_dot > 0 and n_dot_j > 0:
                    score += (nij / n_docs) * _safe_log((nij * n_docs) / (ni_dot * n_dot_j))
        scores[word] = score
    return scores

def select_word_vocabulary(
    train_users_set: Set[str],
    train_user_to_state: Dict[str, str],
) -> Tuple[List[str], WordSelectionMetadata]:
    log(f"[words] selector={WORD_SELECTOR} stat={WORD_STAT} candidate_pool={WORD_CANDIDATE_POOL:,} top_k={WORD_TOP_K:,}", 1)

    df_counts: Counter = Counter()
    total_counts: Counter = Counter()
    label_doc_counts: Counter = Counter()
    present_counts_by_label: Dict[str, Counter] = defaultdict(Counter)
    n_train_docs = 0

    t0 = time.time()
    n_bad_int = 0
    first_bad_sample: Optional[str] = None
    for uid, raw in iter_vocab_rows_by_source(train_users_set, WORD_FEATURE_SRC):
        label = train_user_to_state.get(uid)
        if label is None:
            continue
        n_train_docs += 1
        label_doc_counts[label] += 1
        seen_words: Set[str] = set()
        for k, v in raw.items():
            try:
                iv = int(v)
            except Exception as e:
                n_bad_int += 1
                if first_bad_sample is None:
                    first_bad_sample = f"uid={uid!r} key={k!r} value={v!r}: {e}"
                continue
            if iv <= 0:
                continue
            word = str(k)
            total_counts[word] += iv
            if word not in seen_words:
                df_counts[word] += 1
                present_counts_by_label[label][word] += 1
                seen_words.add(word)
        if n_train_docs and n_train_docs % 25000 == 0:
            log(f"[words] pass1 processed {n_train_docs:,} training users", 1)
    if n_bad_int:
        log(f"[words] skipped {n_bad_int:,} non-integer count values during vocab pass; first: {first_bad_sample}", 1)

    stat_counter = df_counts if WORD_STAT == "df" else total_counts
    candidates: List[str] = []
    for word, score in stat_counter.items():
        if df_counts[word] < WORD_MIN_DF:
            continue
        if total_counts[word] < WORD_MIN_TOTAL_COUNT:
            continue
        candidates.append(word)

    candidates.sort(key=lambda w: (stat_counter[w], df_counts[w], total_counts[w], w), reverse=True)
    if WORD_CANDIDATE_POOL > 0:
        candidates = candidates[:WORD_CANDIDATE_POOL]
    candidate_set = set(candidates)
    log(f"[words] eligible candidates after thresholds: {len(candidates):,}", 1)

    if WORD_SELECTOR == "freq":
        selected = candidates[:WORD_TOP_K]
    else:
        mi_scores = compute_binary_mi(
            present_counts_by_label=present_counts_by_label,
            label_doc_counts=label_doc_counts,
            candidate_words=candidate_set,
            n_docs=n_train_docs,
        )
        selected = sorted(
            candidate_set,
            key=lambda w: (mi_scores.get(w, 0.0), df_counts[w], total_counts[w], w),
            reverse=True,
        )[:WORD_TOP_K]

    meta = WordSelectionMetadata(
        selector=WORD_SELECTOR,
        stat=WORD_STAT,
        min_df=WORD_MIN_DF,
        min_total_count=WORD_MIN_TOTAL_COUNT,
        candidate_pool=WORD_CANDIDATE_POOL,
        top_k=WORD_TOP_K,
        selected_size=len(selected),
    )
    log(f"[words] selected final vocabulary of {len(selected):,} words in {(time.time() - t0)/60:.2f} min", 1)
    return selected, meta

### Feature Matrix Building

def build_word_count_matrix(
    users: List[str],
    selected_words: List[str],
) -> sparse.csr_matrix:
    user_index = {u: i for i, u in enumerate(users)}
    word_index = {w: i for i, w in enumerate(selected_words)}
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    t0 = time.time()
    matched = 0
    n_bad_int = 0
    first_bad_sample: Optional[str] = None
    for uid, raw in iter_vocab_rows_by_source(set(users), WORD_FEATURE_SRC):
        row = user_index.get(uid)
        if row is None:
            continue
        matched += 1
        for k, v in raw.items():
            col = word_index.get(str(k))
            if col is None:
                continue
            try:
                iv = int(v)
            except Exception as e:
                n_bad_int += 1
                if first_bad_sample is None:
                    first_bad_sample = f"uid={uid!r} key={k!r} value={v!r}: {e}"
                continue
            if iv > 0:
                rows.append(row)
                cols.append(col)
                data.append(float(iv))
        if matched and matched % 25000 == 0:
            log(f"[words] pass2 built raw rows for {matched:,} users", 1)

    X = sparse.csr_matrix((np.asarray(data, dtype=np.float32), (rows, cols)), shape=(len(users), len(selected_words)), dtype=np.float32)
    X.sum_duplicates()
    X.sort_indices()
    if n_bad_int:
        log(f"[words] skipped {n_bad_int:,} non-integer count values during pass2; first: {first_bad_sample}", 1)
    log(f"[words] raw sparse matrix shape={X.shape} nnz={X.nnz:,} elapsed={(time.time() - t0)/60:.2f} min", 1)
    return X

def build_struct_matrix(
    users: List[str],
    subs_by_user: Dict[str, Dict[str, int]],
    hours_by_user: Dict[str, Dict[str, int]],
    train_users_set: Set[str],
) -> Tuple[sparse.csr_matrix, DictVectorizer, Optional[TfidfTransformer]]:
    # Select subreddit vocabulary by training DF only.
    sub_df = Counter()
    for uid in train_users_set:
        raw = subs_by_user.get(uid) or {}
        for sub in raw.keys():
            sub_df[sub] += 1
    selected_subs = {
        sub for sub, df in sub_df.most_common(SUBREDDIT_TOP_K) if df >= SUBREDDIT_MIN_DF
    }
    log(f"[struct] selected {len(selected_subs):,} subreddit features", 1)

    dict_rows: List[Dict[str, float]] = []
    for uid in users:
        feats: Dict[str, float] = {}
        raw_subs = {k: v for k, v in (subs_by_user.get(uid) or {}).items() if k in selected_subs}
        raw_hours = hours_by_user.get(uid) or {}
        feats.update(_normalize_struct_counts(raw_subs, STRUCT_SUB_MODE, STRUCT_SUB_WEIGHT, "s:"))
        feats.update(_normalize_struct_counts(raw_hours, STRUCT_HOUR_MODE, STRUCT_HOUR_WEIGHT, "h:"))
        dict_rows.append(feats)

    vectorizer = DictVectorizer(sparse=True)
    X = vectorizer.fit_transform(dict_rows).astype(np.float32).tocsr()
    tfidf = None
    if STRUCT_SUB_MODE == "tfidf":
        sub_cols = [i for i, name in enumerate(vectorizer.feature_names_) if name.startswith("s:")]
        if sub_cols:
            tfidf = TfidfTransformer(norm=WORD_NORM_OPT, use_idf=True, smooth_idf=True, sublinear_tf=True)
            X_sub = tfidf.fit_transform(X[:, sub_cols])
            X = X.tolil(copy=True)
            X[:, sub_cols] = X_sub
            X = X.tocsr()
    log(f"[struct] matrix shape={X.shape} nnz={X.nnz:,}", 1)
    return X, vectorizer, tfidf

### Label array processing

def build_label_arrays(
    users: List[str],
    user_to_label: Dict[str, UserLabel],
) -> Dict[str, np.ndarray]:
    top_labels = np.array([user_to_label[u].top_label for u in users], dtype=object)
    state_labels = np.array([user_to_label[u].state_label for u in users], dtype=object)
    region_labels = np.array([user_to_label[u].region_label for u in users], dtype=object)

    state_mask = np.array([lab == TOP_US and user_to_label[u].state_label != STATE_UNKNOWN for u, lab in zip(users, top_labels)], dtype=np.bool_)
    region_mask = np.array([lab == TOP_NON_US and user_to_label[u].region_label != REGION_UNKNOWN for u, lab in zip(users, top_labels)], dtype=np.bool_)

    return {
        "y_top": top_labels,
        "y_state": state_labels,
        "y_region": region_labels,
        "mask_state": state_mask,
        "mask_region": region_mask,
    }

### Feature Cache
# prevents re-generation of existing processed feature files. 

def _artifact_tag() -> str:
    return f"src-{WORD_FEATURE_SRC}"

def get_artifact_paths(output_dir: str) -> Dict[str, str]:
    tag = _artifact_tag()
    return {
        "users": os.path.join(output_dir, f"users__{tag}.npy"),
        "x_words": os.path.join(output_dir, f"X_words__{tag}.npz"),
        "x_words_masked": os.path.join(output_dir, f"X_words_masked__{tag}.npz"),
        "x_struct": os.path.join(output_dir, f"X_struct__{tag}.npz"),
        "labels_splits": os.path.join(output_dir, f"labels_and_splits__{tag}.npz"),
        "metadata": os.path.join(output_dir, f"metadata__{tag}.json"),
        "preprocessor": os.path.join(output_dir, f"preprocessor__{tag}.pkl"),
    }

def _base_required_paths(output_dir: str, save_preprocessor: bool) -> List[str]:
    paths = get_artifact_paths(output_dir)
    required = [
        paths["users"],
        paths["x_words"],
        paths["x_struct"],
        paths["labels_splits"],
        paths["metadata"],
    ]
    if save_preprocessor:
        required.append(paths["preprocessor"])
    return required

def base_cache_is_complete(output_dir: str, save_preprocessor: bool) -> bool:
    missing = [p for p in _base_required_paths(output_dir, save_preprocessor) if not os.path.exists(p)]
    if not missing:
        log("[cache] Found existing non-masked preprocessing artifacts.", 1)
        return True
    log(f"[cache] Base artifact miss: {', '.join(os.path.basename(p) for p in missing)}", 1)
    return False

def masked_cache_is_complete(output_dir: str) -> bool:
    paths = get_artifact_paths(output_dir)
    if os.path.exists(paths["x_words_masked"]):
        log("[cache] Found existing masked word artifact.", 1)
        return True
    log(f"[cache] Missing masked word artifact: {os.path.basename(paths['x_words_masked'])}", 1)
    return False

def cache_is_complete(output_dir: str, save_preprocessor: bool) -> bool:
    log(f"[cache] artifact tag: {_artifact_tag()}", 1)
    if not base_cache_is_complete(output_dir, save_preprocessor):
        return False
    if MASK_TIER1_WORDS and not masked_cache_is_complete(output_dir):
        return False
    log("[cache] Reusing existing preprocessed artifacts; no rebuild needed.", 1)
    return True

def _update_existing_metadata_for_masked_artifact(metadata_path: str, masked_shape: Tuple[int, int]) -> None:
    metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            log(f"[mask] failed to read existing metadata for update: {e}", 1)
            metadata = {}
    metadata["has_masked_words"] = True
    metadata["n_word_features_masked"] = int(masked_shape[1])
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

def _update_existing_preprocessor_for_masked_artifact(preprocessor_path: str) -> None:
    if not os.path.exists(preprocessor_path):
        return
    try:
        with open(preprocessor_path, "rb") as f:
            preprocessor = pickle.load(f)
    except Exception as e:
        log(f"[mask] failed to read existing preprocessor for update: {e}", 1)
        return
    if not isinstance(preprocessor, dict):
        return
    preprocessor["mask_tier1_words"] = MASK_TIER1_WORDS
    preprocessor["masking_notes"] = "Tier-1 masked word artifact removes obvious mentions of each user's own country/state label from word features, plus a small alias map for common country/state variants."
    try:
        with open(preprocessor_path, "wb") as f:
            pickle.dump(preprocessor, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        log(f"[mask] failed to update existing preprocessor with masking metadata: {e}", 1)

def maybe_build_masked_only(output_dir: str, labels_csv: str) -> bool:
    if not MASK_TIER1_WORDS:
        return False

    artifact_paths = get_artifact_paths(output_dir)
    if not base_cache_is_complete(output_dir, SAVE_PREPROCESSOR):
        return False
    if masked_cache_is_complete(output_dir):
        log(f"[cache] using artifacts in {output_dir}", 1)
        for key, path in artifact_paths.items():
            if key == "preprocessor" and not SAVE_PREPROCESSOR:
                continue
            log(f"[cache] {key}: {path}", 1)
        return True

    log("[mask] Base artifacts found and masked artifact missing; generating masked words from existing preprocessing outputs.", 1)

    if not os.path.exists(artifact_paths["preprocessor"]):
        log("[mask] Cannot build masked artifact from cache because preprocessor.pkl is missing. Falling back to full rebuild.", 1)
        return False

    with open(artifact_paths["preprocessor"], "rb") as f:
        preprocessor = pickle.load(f)
    if not isinstance(preprocessor, dict):
        log("[mask] Existing preprocessor artifact is not a dict. Falling back to full rebuild.", 1)
        return False

    selected_words = preprocessor.get("selected_words")
    word_tfidf = preprocessor.get("word_tfidf")
    if not selected_words or word_tfidf is None:
        log("[mask] Existing preprocessor artifact lacks selected_words or word_tfidf. Falling back to full rebuild.", 1)
        return False

    users = np.load(artifact_paths["users"], allow_pickle=True).tolist()
    user_to_label_all = load_user_labels(labels_csv)
    user_to_label = {u: v for u, v in user_to_label_all.items() if u in set(users)}
    missing_users = [u for u in users if u not in user_to_label]
    if missing_users:
        preview = ", ".join(missing_users[:5])
        log(f"[mask] Missing labels for {len(missing_users):,} cached users while building masked artifact (e.g. {preview}). Falling back to full rebuild.", 1)
        return False

    X_words_counts_masked = build_masked_word_count_matrix(users, list(selected_words), user_to_label)
    X_words_masked = word_tfidf.transform(X_words_counts_masked).astype(np.float32).tocsr()
    log(f"[words_masked] tfidf matrix shape={X_words_masked.shape} nnz={X_words_masked.nnz:,}", 1)
    save_npz(artifact_paths["x_words_masked"], X_words_masked)

    _update_existing_metadata_for_masked_artifact(artifact_paths["metadata"], X_words_masked.shape)
    _update_existing_preprocessor_for_masked_artifact(artifact_paths["preprocessor"])

    log(f"[mask] wrote masked artifact to {artifact_paths['x_words_masked']}", 1)
    return True

### Main Preprocessing Function

def preprocess_location_data():

    # log preprocessing hyperparameters
    log(f"[args] resource={args.resource} type={args.type} -> WORD_FEATURE_SRC={WORD_FEATURE_SRC}", 1)
    log(
        f"[config] source={WORD_FEATURE_SRC} selector={WORD_SELECTOR}/{WORD_STAT} "
        f"word_min_df={WORD_MIN_DF} word_min_total_count={WORD_MIN_TOTAL_COUNT} "
        f"word_candidate_pool={WORD_CANDIDATE_POOL} word_top_k={WORD_TOP_K} "
        f"word_norm={WORD_NORM} use_idf={int(WORD_USE_IDF)} smooth_idf={int(WORD_SMOOTH_IDF)} sublinear_tf={int(WORD_SUBLINEAR_TF)} "
        f"subreddit_top_k={SUBREDDIT_TOP_K} subreddit_min_df={SUBREDDIT_MIN_DF} "
        f"struct_sub_mode={STRUCT_SUB_MODE} struct_hour_mode={STRUCT_HOUR_MODE} "
        f"mask_tier1_words={int(MASK_TIER1_WORDS)}",
        1,
    )
    log(f"[config] artifact tag={_artifact_tag()}", 1)

    ## Label Processing

    # load the labeled user data
    labels_csv = os.path.join(input_path, "combined_geohash.csv")

    # cache check/use
    t0_all = time.time()
    if cache_is_complete(PREPROC_PATH, SAVE_PREPROCESSOR):
        log(f"[cache] using artifacts in {PREPROC_PATH}", 1)
        for key, path in get_artifact_paths(PREPROC_PATH).items():
            if key == "preprocessor" and not SAVE_PREPROCESSOR:
                continue
            log(f"[cache] {key}: {path}", 1)
        return
    if maybe_build_masked_only(PREPROC_PATH, labels_csv):
        log(f"[done] total elapsed {(time.time() - t0_all)/60:.2f} min", 1)
        return
    log("[cache] Building preprocessing artifacts from raw inputs.", 1)

    # load user labels
    user_to_label = load_user_labels(labels_csv)
    all_users = list(user_to_label.keys())
    all_top_labels = [user_to_label[u].top_label for u in all_users]

    ## Feature loading

    subs_by_user = load_subreddit_counts(SUBS_JSONL, set(all_users))
    hours_by_user = load_hour_counts(HOURS_JSONL, set(all_users))
    vocab_present_users: Set[str] = set()
    for uid, _ in iter_vocab_rows_by_source(set(all_users), WORD_FEATURE_SRC):
        vocab_present_users.add(uid)
    log(f"[vocab] users with requested word source(s): {len(vocab_present_users):,}", 1)

    featured_users_set = set(all_users) & set(subs_by_user.keys()) & set(hours_by_user.keys()) & vocab_present_users
    log(f"[align] users with labels + words + subs + hours: {len(featured_users_set):,}", 1)
    log(f"[align] dropped labeled users missing >=1 feature family: {len(set(all_users) - featured_users_set):,}", 1)

    user_to_label = {u: v for u, v in user_to_label.items() if u in featured_users_set}
    all_users = list(user_to_label.keys())
    all_top_labels = [user_to_label[u].top_label for u in all_users]

    ## training/valid/test split

    train_users, train_top, val_users, val_top, test_users, test_top = prepare_splits(
        all_users,
        all_top_labels,
        split_dir=SPLIT_DIR
    )

    summarize_split("train", train_users, train_top)
    summarize_split("valid", val_users, val_top)
    summarize_split("test", test_users, test_top)

    users = list(train_users) + list(val_users) + list(test_users)

    ## Training feature processing

    # label mapping for users in the training/valid/test sets
    train_users_set = set(train_users)
    train_user_to_state = {u: user_to_label[u].state_label for u in train_users if user_to_label[u].top_label == TOP_US and user_to_label[u].state_label != STATE_UNKNOWN}
    for u in train_users:
        if user_to_label[u].top_label == TOP_NON_US:
            train_user_to_state[u] = user_to_label[u].region_label if user_to_label[u].region_label != REGION_UNKNOWN else TOP_NON_US
        elif user_to_label[u].top_label == TOP_US and user_to_label[u].state_label == STATE_UNKNOWN:
            train_user_to_state[u] = TOP_US

    label_arrays = build_label_arrays(users, user_to_label)

    split_arrays = {
    "train_idx": np.arange(0, len(train_users), dtype=np.int64),
    "val_idx": np.arange(len(train_users), len(train_users) + len(val_users), dtype=np.int64),
    "test_idx": np.arange(len(train_users) + len(val_users), len(users), dtype=np.int64),
    }

    # word feature transformation and selection for the training set
    selected_words, word_meta = select_word_vocabulary(train_users_set, train_user_to_state)
    X_words_counts = build_word_count_matrix(users, selected_words)
    word_tfidf = TfidfTransformer(
        norm=WORD_NORM_OPT,
        use_idf=WORD_USE_IDF,
        smooth_idf=WORD_SMOOTH_IDF,
        sublinear_tf=WORD_SUBLINEAR_TF,
    )
    X_words = word_tfidf.fit_transform(X_words_counts[: len(train_users), :])
    X_words_all = word_tfidf.transform(X_words_counts)
    X_words_all = X_words_all.astype(np.float32).tocsr()

    log(f"[words] tfidf matrix shape={X_words_all.shape} nnz={X_words_all.nnz:,}", 1)
    
    del X_words

    # training words feature set masking
    if MASK_TIER1_WORDS:
        X_words_counts_masked = build_masked_word_count_matrix(users, selected_words, user_to_label)
        X_words_masked = word_tfidf.transform(X_words_counts_masked)
        X_words_masked = X_words_masked.astype(np.float32).tocsr()
        log(f"[words_masked] tfidf matrix shape={X_words_masked.shape} nnz={X_words_masked.nnz:,}", 1)
    else:
        X_words_masked = X_words_all.copy()
        log("[words_masked] masking disabled; masked artifact mirrors unmasked X_words", 1)

    # structural feature processing for the training set
    X_struct, struct_vectorizer, struct_tfidf = build_struct_matrix(users, subs_by_user, hours_by_user, train_users_set)

    # save feature artifacts
    artifact_paths = get_artifact_paths(PREPROC_PATH)
    np.save(artifact_paths["users"], np.array(users, dtype=object), allow_pickle=True)
    save_npz(artifact_paths["x_words"], X_words_all)
    save_npz(artifact_paths["x_words_masked"], X_words_masked)
    save_npz(artifact_paths["x_struct"], X_struct)
    np.savez_compressed(artifact_paths["labels_splits"], **label_arrays, **split_arrays)

    # define the preprocessor object
    preprocessor = {
        "selected_words": selected_words,
        "word_tfidf": word_tfidf,
        "struct_vectorizer": struct_vectorizer,
        "struct_tfidf": struct_tfidf,
        "word_selection": asdict(word_meta),
        "mask_tier1_words": MASK_TIER1_WORDS,
        "masking_notes": "Tier-1 masked word artifact removes obvious mentions of each user's own country/state label from word features, plus a small alias map for common country/state variants.",
    }
    if SAVE_PREPROCESSOR: # save the preprocessor object
        with open(artifact_paths["preprocessor"], "wb") as f:
            pickle.dump(preprocessor, f, protocol=pickle.HIGHEST_PROTOCOL)

    # save metadata for the preprocessing
    metadata = PreprocessMetadata(
        word_feature_src=WORD_FEATURE_SRC,
        word_selector=WORD_SELECTOR,
        word_stat=WORD_STAT,
        word_min_df=WORD_MIN_DF,
        word_min_total_count=WORD_MIN_TOTAL_COUNT,
        word_candidate_pool=WORD_CANDIDATE_POOL,
        word_top_k=WORD_TOP_K,
        subreddit_top_k=SUBREDDIT_TOP_K,
        struct_sub_mode=STRUCT_SUB_MODE,
        struct_hour_mode=STRUCT_HOUR_MODE,
        n_users_total=len(set(all_users)),
        n_users_feature_aligned=len(users),
        n_train=len(train_users),
        n_val=len(val_users),
        n_test=len(test_users),
        n_word_features=X_words_all.shape[1],
        n_struct_features=X_struct.shape[1],
        n_state_classes=len(sorted({lab for lab in label_arrays["y_state"] if lab != STATE_UNKNOWN})),
        n_region_classes=len(sorted({lab for lab in label_arrays["y_region"] if lab != REGION_UNKNOWN})),
        has_masked_words=MASK_TIER1_WORDS,
        n_word_features_masked=X_words_masked.shape[1],
    )
    with open(artifact_paths["metadata"], "w", encoding="utf-8") as f:
        json.dump(asdict(metadata), f, indent=2, sort_keys=True)

    # log the end of the task
    log(f"[done] wrote artifacts to {PREPROC_PATH}", 1)
    for key, path in artifact_paths.items():
        if key == "preprocessor" and not SAVE_PREPROCESSOR:
            continue
        log(f"[done] {key}: {path}", 1)
    log(f"[done] total elapsed {(time.time() - t0_all)/60:.2f} min", 1)

### Main Execution

if __name__ == "__main__":
    preprocess_location_data()