"""XGBoost-based training pipeline for the BIST order-flow signal model.

:class:`SignalModelTrainer` builds a binary classifier that predicts whether
the mid-price will be *higher* at ``forward_horizon_sec`` seconds in the
future.  Features must be the output of
:class:`~bist_bot.features.engineer.FeatureEngineer`.

Walk-forward validation is used via :class:`~sklearn.model_selection.TimeSeriesSplit`
to avoid look-ahead bias.

Typical training workflow::

    trainer = SignalModelTrainer(forward_horizon_sec=60)
    X, y = trainer.build_dataset(feature_df, price_series)
    trainer.train(X, y)
    trainer.save("models/bofa_a1_tera_xgb.pkl")
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class SignalModelTrainer:
    """Train and persist an XGBoost classifier on historical broker flow data.

    Args:
        forward_horizon_sec: Number of *rows* (ticks / samples) to look
            ahead when constructing the binary target label.  When the
            feature DataFrame has regular 1-second sampling this equals
            the number of seconds.
        n_splits: Number of folds for walk-forward cross-validation.
        xgb_params: Optional override for XGBoost hyper-parameters.
    """

    _DEFAULT_XGB_PARAMS: dict = {
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.02,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": 42,
    }

    def __init__(
        self,
        forward_horizon_sec: int = 60,
        n_splits: int = 5,
        xgb_params: dict | None = None,
    ) -> None:
        self.forward_horizon_sec = forward_horizon_sec
        self.n_splits = n_splits

        params = {**self._DEFAULT_XGB_PARAMS, **(xgb_params or {})}
        self.model = xgb.XGBClassifier(**params)
        self.scaler = StandardScaler()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_dataset(
        self,
        feature_df: pd.DataFrame,
        price_series: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Align features with a forward-looking binary target.

        Args:
            feature_df: Rows indexed by timestamp; columns are feature names
                produced by :class:`~bist_bot.features.engineer.FeatureEngineer`.
            price_series: Mid-price (or last trade price) aligned to the same
                index as *feature_df*.

        Returns:
            ``(X, y)`` — the cleaned feature matrix and binary target series.
            Rows where the future price is unknown are dropped.
        """
        future_price = price_series.shift(-self.forward_horizon_sec)
        target = (future_price > price_series).astype(int)
        valid_mask = ~future_price.isna()

        X = feature_df.loc[valid_mask].copy()
        y = target.loc[valid_mask]

        # Replace any remaining NaN/Inf in features with 0.
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        logger.info(
            "Dataset built: %d samples, %d features, %.1f%% positive.",
            len(X),
            X.shape[1],
            100.0 * y.mean(),
        )
        return X, y

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Fit the model and return walk-forward cross-validation AUC scores.

        Args:
            X: Feature matrix (output of :meth:`build_dataset`).
            y: Binary target series.

        Returns:
            Dictionary with ``mean_cv_auc`` and ``std_cv_auc``.
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        X_scaled = self.scaler.fit_transform(X)

        cv_scores = cross_val_score(
            self.model, X_scaled, y, cv=tscv, scoring="roc_auc", n_jobs=-1
        )
        logger.info(
            "Walk-forward CV AUC: %.4f ± %.4f", cv_scores.mean(), cv_scores.std()
        )

        # Final fit on full training set.
        self.model.fit(X_scaled, y, verbose=False)

        return {"mean_cv_auc": float(cv_scores.mean()), "std_cv_auc": float(cv_scores.std())}

    def save(self, path: str | Path) -> None:
        """Persist model and scaler to *path*.

        Args:
            path: Destination file path (e.g. ``"models/bofa_a1_tera_xgb.pkl"``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "SignalModelTrainer":
        """Load a previously saved trainer from *path*.

        Args:
            path: File path written by :meth:`save`.

        Returns:
            A :class:`SignalModelTrainer` instance with model and scaler
            restored.
        """
        bundle = joblib.load(path)
        obj = cls.__new__(cls)
        obj.model = bundle["model"]
        obj.scaler = bundle["scaler"]
        return obj
