"""Preprocess a URL CSV and write feature-engineered data to a CSV file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from url_phishing_classification.preprocessing import deduplicate_urls, engineer_features, normalize_url


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the preprocessing job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_csv", type=Path, help="Input CSV containing a URL column.")
    parser.add_argument("output_csv", type=Path, help="Destination for the preprocessed CSV.")
    parser.add_argument("--url-column", default="url", help="Name of the URL column (default: url).")
    parser.add_argument(
        "--label-column",
        default="label",
        help="Name of the label column used to detect conflicting duplicates (default: label).",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Skip normalized-URL deduplication.",
    )
    return parser.parse_args()


def main() -> None:
    """Read the source CSV, preprocess it, and write the result."""
    args = parse_args()
    df = pd.read_csv(args.source_csv)

    # Step 1: normalize and deduplicate
    if args.url_column not in df:
        raise ValueError(f"Missing required URL column: {args.url_column!r}")
    df["normalized_url"] = df[args.url_column].fillna("").astype(str).map(normalize_url)
    if not args.keep_duplicates:
        df = deduplicate_urls(df, label_column=args.label_column)

    # Step 2: derive URL features
    processed = engineer_features(df, url_column=args.url_column)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(processed):,} rows and {len(processed.columns)} columns to {args.output_csv}")


if __name__ == "__main__":
    main()
