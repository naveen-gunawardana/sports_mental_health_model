"""isaac-data: a thin Python loader for the ISAAC Reddit corpus.

Reads the ISAAC Direct Download endpoint
(https://isaac.psychology.illinois.edu/direct-download/) using the published
manifest as a catalog. Parquet reads support column pushdown over HTTP.

Quick start
-----------
>>> import isaac_data as isaac
>>> isaac.files("race", "2018-01", "2018-12")          # what's available
>>> df = isaac.load("race", "2018-03", "2018-03",       # one month, two columns
...                 columns=["text", "score"])
>>> isaac.download("age", "2015-01", "2015-12", dest="./age2015")  # bulk fetch
"""
from .core import (
    BASE_URL,
    CATEGORIES,
    DATA_BASE,
    MANIFEST_URL,
    cache_dir,
    catalog,
    download,
    files,
    load,
    read_parquet,
    set_cache_dir,
)
from .terms import (
    TermsNotAccepted,
    accept_terms,
    is_accepted,
    status as terms_status,
    withdraw as withdraw_terms,
)

__version__ = "0.1.1"

__all__ = [
    "__version__",
    "BASE_URL", "DATA_BASE", "MANIFEST_URL", "CATEGORIES",
    "cache_dir", "set_cache_dir", "catalog", "files",
    "download", "read_parquet", "load",
    "accept_terms", "is_accepted", "terms_status", "withdraw_terms", "TermsNotAccepted",
]
