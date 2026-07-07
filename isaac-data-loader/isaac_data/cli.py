"""Command-line interface for isaac-data.

Examples
--------
    isaac-data ls --category race --start 2018-01 --end 2018-12
    isaac-data info
    isaac-data download --category age --start 2015-01 --end 2015-12 --dest ./age2015
"""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .core import CATEGORIES, download, files


def _add_selection_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--category", choices=CATEGORIES, help="social group (default: all)")
    p.add_argument("-s", "--start", help="start month, YYYY-MM (inclusive)")
    p.add_argument("-e", "--end", help="end month, YYYY-MM (inclusive)")
    p.add_argument("-f", "--format", dest="fmt", default="parquet",
                   choices=["parquet", "csv", "both"], help="file format (default: parquet)")


def _fmt(args) -> str | None:
    return None if args.fmt == "both" else args.fmt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="isaac-data", description="ISAAC corpus direct-download helper.")
    parser.add_argument("--version", action="version", version=f"isaac-data {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ls = sub.add_parser("ls", help="list matching files")
    _add_selection_args(p_ls)

    p_info = sub.add_parser("info", help="summary of the whole corpus")

    p_dl = sub.add_parser("download", help="download matching files (resumable)")
    _add_selection_args(p_dl)
    p_dl.add_argument("-d", "--dest", help="destination directory (default: cache)")

    p_acc = sub.add_parser("accept-terms", help="review & accept the Terms of Use (recorded locally)")
    p_acc.add_argument("-y", "--yes", action="store_true", help="accept without the interactive prompt")
    p_acc.add_argument("--status", action="store_true", help="show current acceptance record and exit")
    p_acc.add_argument("--withdraw", action="store_true", help="delete the local acceptance record")

    args = parser.parse_args(argv)

    if args.cmd == "accept-terms":
        from .terms import accept_terms, status, withdraw, TermsNotAccepted
        if args.status:
            s = status()
            print(json.dumps(s, indent=2) if s else "Terms of Use not yet accepted on this machine.")
            return 0
        if args.withdraw:
            print("Removed local acceptance record." if withdraw() else "No acceptance record to remove.")
            return 0
        try:
            accept_terms(assume_yes=args.yes)
            return 0
        except TermsNotAccepted as e:
            print(str(e), file=sys.stderr)
            return 2

    if args.cmd == "ls":
        df = files(args.category, args.start, args.end, _fmt(args))
        cols = ["category", "year", "month", "format", "size_bytes", "num_rows", "url"]
        with_pd_print(df[cols])
        print(f"\n{len(df)} files, {df['size_bytes'].sum()/1e9:.2f} GB", file=sys.stderr)
        return 0

    if args.cmd == "info":
        df = files(None, None, None, None)
        import pandas as pd  # noqa
        g = df.groupby(["category", "format"]).agg(files=("url", "size"),
                                                   gb=("size_bytes", lambda s: round(s.sum() / 1e9, 1)))
        with_pd_print(g.reset_index())
        print(f"\nTotal: {len(df)} files, {df['size_bytes'].sum()/1e9:.1f} GB", file=sys.stderr)
        return 0

    if args.cmd == "download":
        paths = download(args.category, args.start, args.end, _fmt(args), dest=args.dest)
        for p in paths:
            print(p)
        print(f"\nDownloaded {len(paths)} file(s).", file=sys.stderr)
        return 0

    return 1


def with_pd_print(df) -> None:
    import pandas as pd
    with pd.option_context("display.max_rows", 200, "display.width", 200):
        print(df.to_string(index=False))


if __name__ == "__main__":
    raise SystemExit(main())
