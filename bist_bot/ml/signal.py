"""Real-time signal generation from a trained XGBoost model.

:class:`SignalGenerator` loads a persisted model bundle and generates a
directional trading signal (``"BUY"``, ``"SELL"``, or ``"FLAT"``) from a
feature dictionary produced by
:class:`~bist_bot.features.engineer.FeatureEngineer`.

Usage::

    gen = SignalGenerator("models/bofa_a1_tera_xgb.pkl")
    signal = gen.generate(features)
    # {"signal": "BUY", "confidence": 0.7312}
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Wraps a trained classifier and produces actionable trading signals.

    Args:
        model_path: Path to a ``.pkl`` bundle saved by
            :class:`~bist_bot.ml.trainer.SignalModelTrainer`.
        threshold_buy: Minimum predicted probability of an up-move to emit
            a ``"BUY"`` signal.  Default ``0.62``.
        threshold_sell: Maximum predicted probability to emit a ``"SELL"``
            signal.  Default ``0.38``.
    """

    def __init__(
        self,
        model_path: str | Path,
        threshold_buy: float = 0.62,
        threshold_sell: float = 0.38,
    ) -> None:
        bundle = joblib.load(model_path)
        self._model = bundle["model"]
        self._scaler = bundle["scaler"]
        self.threshold_buy = threshold_buy
        self.threshold_sell = threshold_sell
        logger.info("SignalGenerator loaded model from %s", model_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, features: dict[str, float]) -> dict:
        """Produce a signal dict from a feature snapshot.

        Args:
            features: Flat dictionary mapping feature name → float, as
                returned by
                :meth:`~bist_bot.features.engineer.FeatureEngineer.compute_features`.

        Returns:
            Dictionary with keys:

            * ``signal`` – ``"BUY"``, ``"SELL"``, or ``"FLAT"``
            * ``confidence`` – probability of an up-move (0 – 1)
        """
        X = pd.DataFrame([features])
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_scaled = self._scaler.transform(X)

        proba_up = float(self._model.predict_proba(X_scaled)[0, 1])

        if proba_up >= self.threshold_buy:
            signal = "BUY"
        elif proba_up <= self.threshold_sell:
            signal = "SELL"
        else:
            signal = "FLAT"

        return {"signal": signal, "confidence": round(proba_up, 4)}
