"""Train and save the two finalized URL phishing logistic-regression models."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from url_phishing_classification.config import FINAL_MODEL_SPEC
from url_phishing_classification.training import fit_final_model

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"


def parse_args() -> argparse.Namespace:
    """Parse the path to a labelled, feature-engineered training CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "training_csv",
        type=Path,
        help="CSV produced by the preprocessing workflow, containing label and engineered feature columns.",
    )
    return parser.parse_args()


def main() -> None:
    """Fit finalized model and save their artifact."""
    args = parse_args()
    if not args.training_csv.exists():
        raise FileNotFoundError(f"Training data not found: {args.training_csv}")

    training_data = pd.read_csv(args.training_csv)
    model = fit_final_model(training_data)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    artifact_path = MODELS_DIR / FINAL_MODEL_SPEC.artifact_name
    joblib.dump(model, artifact_path)
    print(
        f"Saved all_engineered: {len(FINAL_MODEL_SPEC.features)} features, "
        f"C={FINAL_MODEL_SPEC.C:g}, {artifact_path.relative_to(PROJECT_ROOT)}"
    )
    print(f"Training rows read: {len(training_data):,}.")


if __name__ == "__main__":
    main()
