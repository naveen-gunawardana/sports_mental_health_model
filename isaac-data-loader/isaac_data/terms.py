"""First-run Terms-of-Use acceptance (recorded locally).

Data access (`load`, `download`, remote `read_parquet`) requires a one-time
acceptance of the ISAAC Terms of Use. Acceptance is recorded ONLY on the local
machine (OS-native config dir). Browsing the catalog (`catalog`, `files`) does
not require acceptance.

Non-interactive use (CI, notebooks without a TTY) must either accept beforehand
via `isaac-data accept-terms` or set the environment variable
``ISAAC_ACCEPT_TERMS=1``; otherwise data access raises ``TermsNotAccepted``.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

TERMS_PAGE = "https://github.com/BabakHemmatian/Illinois_Social_Attitudes/blob/main/Terms_of_Use.md"
TERMS_RAW = "https://raw.githubusercontent.com/BabakHemmatian/Illinois_Social_Attitudes/main/Terms_of_Use.md"
_ENV = "ISAAC_ACCEPT_TERMS"


class TermsNotAccepted(RuntimeError):
    """Raised when ISAAC Terms of Use have not been accepted."""


def _config_dir() -> Path:
    override = os.environ.get("ISAAC_DATA_CONFIG")
    if override:
        d = Path(override).expanduser()
    else:
        import platformdirs
        d = Path(platformdirs.user_config_dir("isaac-data"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _record_file() -> Path:
    return _config_dir() / "accepted.json"


def is_accepted() -> bool:
    return _record_file().exists()


def status() -> Optional[dict]:
    """Return the local acceptance record, or None if not yet accepted."""
    f = _record_file()
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return {"accepted": True, "record": str(f)}


def fetch_terms(timeout: int = 15) -> Optional[str]:
    """Best-effort fetch of the current Terms of Use text (None if unavailable)."""
    import requests
    try:
        r = requests.get(TERMS_RAW, timeout=timeout)
        if r.ok and r.text.strip():
            return r.text
    except Exception:
        pass
    return None


def withdraw() -> bool:
    """Delete the local acceptance record. Returns True if one existed."""
    f = _record_file()
    if f.exists():
        f.unlink()
        return True
    return False


def _write_record(text: Optional[str]) -> dict:
    from . import __version__
    rec = {
        "accepted": True,
        "accepted_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "terms_url": TERMS_PAGE,
        "terms_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None,
        "package_version": __version__,
    }
    _record_file().write_text(json.dumps(rec, indent=2) + "\n")
    return rec


def accept_terms(assume_yes: bool = False) -> dict:
    """Show the Terms of Use and record acceptance locally.

    assume_yes : skip the prompt (equivalent to ``ISAAC_ACCEPT_TERMS=1``).
    Raises TermsNotAccepted if the user declines or no terminal is available.
    """
    out = sys.stderr
    text = fetch_terms()
    print("\n" + "=" * 72, file=out)
    print("ISAAC dataset — Terms of Use", file=out)
    print("=" * 72, file=out)
    if text:
        print(text.strip(), file=out)
        print("-" * 72, file=out)
    else:
        print("By using the ISAAC dataset and this package you agree to the ISAAC", file=out)
        print(f"Terms of Use:\n  {TERMS_PAGE}", file=out)
        print("-" * 72, file=out)

    if assume_yes or os.environ.get(_ENV, "").strip().lower() in ("1", "true", "yes"):
        rec = _write_record(text)
        print(f"Terms accepted (recorded at {_record_file()}).", file=out)
        return rec

    if not (sys.stdin and sys.stdin.isatty()):
        raise TermsNotAccepted(
            "ISAAC Terms of Use not accepted and no interactive terminal is available.\n"
            f"Read {TERMS_PAGE}, then run `isaac-data accept-terms` or set {_ENV}=1."
        )

    resp = input("Do you accept the ISAAC Terms of Use? [y/N] ").strip().lower()
    if resp in ("y", "yes"):
        rec = _write_record(text)
        print(f"Thank you — acceptance recorded at {_record_file()}.", file=out)
        return rec
    raise TermsNotAccepted("ISAAC Terms of Use were not accepted; aborting.")


def require_acceptance() -> None:
    """Fast gate called before data access. No-op once accepted."""
    if is_accepted():
        return
    if os.environ.get(_ENV, "").strip().lower() in ("1", "true", "yes"):
        _write_record(fetch_terms())
        return
    accept_terms()
