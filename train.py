"""Entry point — click the Run ▶ button in your IDE to train the model.

You can also run it from the terminal::

    python train.py --input historical_features.csv

The input CSV must contain feature columns (produced by FeatureEngineer) plus
a ``mid_price`` column.  A ``timestamp`` column is used as the index when present.

Options::

    --input  / -i   Path to the feature CSV  (required)
    --output / -o   Output model path  (default: models/bofa_a1_tera_xgb.pkl)
    --horizon       Forward-look window in rows/seconds  (default: 60)
    --n-splits      Walk-forward CV folds  (default: 5)

Example::

    python train.py --input data/features.csv
"""

from __future__ import annotations

import os
import sys

# Make sure the project root is on sys.path so 'bist_bot' can always be found,
# regardless of where Python is launched from.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.train import main  # noqa: E402

if __name__ == "__main__":
    main()
