"""Export saved-model probabilities for one CSV of URLs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from predict import predict_probabilities


def parse_args() -> argparse.Namespace:
    """Parse the input and output CSV paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_csv", type=Path, help="CSV containing a url column.")
    parser.add_argument("output_csv", type=Path, help="Destination for url,probability predictions.")
    return parser.parse_args()


def main() -> None:
    """Score the input file and preserve its URL row order in the output."""
    args = parse_args()
    if args.source_csv.resolve() == args.output_csv.resolve():
        raise ValueError("Input and output CSV paths must be different.")

    source = pd.read_csv(args.source_csv, usecols=["url"], keep_default_na=False)
    source["probability"] = predict_probabilities(source["url"].tolist())
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    source.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(source):,} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
