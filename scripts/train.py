"""Command-line training script for the BIST order-flow signal model.

Usage::

    python scripts/train.py --input historical_features.csv --output models/model.pkl

The input CSV must contain columns produced by
:class:`~bist_bot.features.engineer.FeatureEngineer` plus a ``mid_price``
column.  The timestamp column (if present) should be named ``timestamp`` or be
the DataFrame index.

Examples::

    # Basic usage — default output path and 60-second forward horizon
    python scripts/train.py --input data/features.csv

    # Custom output path and horizon
    python scripts/train.py --input data/features.csv \\
        --output models/my_model.pkl \\
        --horizon 120 \\
        --n-splits 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from bist_bot.ml.trainer import SignalModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the BIST order-flow XGBoost signal model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        metavar="CSV",
        help="Path to a CSV file containing feature columns and a 'mid_price' column.",
    )
    parser.add_argument(
        "--output", "-o",
        default="models/bofa_a1_tera_xgb.pkl",
        metavar="PKL",
        help="Destination path for the saved model bundle (.pkl).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=60,
        metavar="SECS",
        help="Forward horizon in rows (seconds when data is 1-second sampled).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        metavar="N",
        help="Number of walk-forward cross-validation folds.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    logger.info("Loading data from %s …", input_path)
    df = pd.read_csv(input_path)

    # Use 'timestamp' column as the index if present.
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index)

    if "mid_price" not in df.columns:
        logger.error(
            "CSV must contain a 'mid_price' column.  "
            "Found columns: %s",
            list(df.columns),
        )
        sys.exit(1)

    price_series = df["mid_price"]
    feature_df = df.drop(columns=["mid_price"])

    logger.info(
        "Loaded %d rows × %d feature columns.", len(df), feature_df.shape[1]
    )

    trainer = SignalModelTrainer(
        forward_horizon_sec=args.horizon,
        n_splits=args.n_splits,
    )

    X, y = trainer.build_dataset(feature_df, price_series)
    metrics = trainer.train(X, y)

    logger.info(
        "Training complete — CV AUC: %.4f ± %.4f",
        metrics["mean_cv_auc"],
        metrics["std_cv_auc"],
    )

    trainer.save(args.output)
    logger.info("Model saved to %s", args.output)


if __name__ == "__main__":
    main()
