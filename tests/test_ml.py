"""Unit tests for SignalModelTrainer and SignalGenerator."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from bist_bot.ml.trainer import SignalModelTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_toy_dataset(n=400, n_features=18, seed=42):
    """Create a small synthetic dataset mimicking broker-flow features."""
    rng = np.random.default_rng(seed)
    feature_names = [f"feat_{i}" for i in range(n_features)]
    X = pd.DataFrame(rng.standard_normal((n, n_features)), columns=feature_names)
    # Simple rule: target=1 when first feature is positive (learnable signal)
    y = (X["feat_0"] > 0).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# SignalModelTrainer tests
# ---------------------------------------------------------------------------

class TestSignalModelTrainer:
    def test_build_dataset_drops_tail_rows(self):
        trainer = SignalModelTrainer(forward_horizon_sec=5)
        idx = pd.date_range("2024-01-01", periods=100, freq="s")
        price = pd.Series(np.linspace(10, 20, 100), index=idx)
        feat_df = pd.DataFrame({"f1": np.ones(100)}, index=idx)
        X, y = trainer.build_dataset(feat_df, price)
        # Last 5 rows should be dropped (no future price)
        assert len(X) == 95
        assert len(y) == 95

    def test_build_dataset_target_is_binary(self):
        trainer = SignalModelTrainer(forward_horizon_sec=3)
        idx = pd.date_range("2024-01-01", periods=50, freq="s")
        price = pd.Series(np.linspace(10, 20, 50), index=idx)
        feat_df = pd.DataFrame({"f1": np.ones(50)}, index=idx)
        _, y = trainer.build_dataset(feat_df, price)
        assert set(y.unique()).issubset({0, 1})

    def test_train_returns_cv_metrics(self):
        trainer = SignalModelTrainer(forward_horizon_sec=1, n_splits=3)
        X, y = _make_toy_dataset(n=200, n_features=10)
        metrics = trainer.train(X, y)
        assert "mean_cv_auc" in metrics
        assert "std_cv_auc" in metrics
        assert 0.0 <= metrics["mean_cv_auc"] <= 1.0

    def test_save_and_load_roundtrip(self):
        trainer = SignalModelTrainer(forward_horizon_sec=1, n_splits=3)
        X, y = _make_toy_dataset(n=200, n_features=10)
        trainer.train(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.pkl")
            trainer.save(model_path)
            assert os.path.exists(model_path)

            loaded = SignalModelTrainer.load(model_path)
            original_preds = trainer.model.predict_proba(
                trainer.scaler.transform(X.iloc[:5])
            )
            loaded_preds = loaded.model.predict_proba(
                loaded.scaler.transform(X.iloc[:5])
            )
            np.testing.assert_allclose(original_preds, loaded_preds, rtol=1e-5)


# ---------------------------------------------------------------------------
# SignalGenerator tests
# ---------------------------------------------------------------------------

class TestSignalGenerator:
    def _train_and_save(self, tmpdir):
        trainer = SignalModelTrainer(forward_horizon_sec=1, n_splits=3)
        X, y = _make_toy_dataset(n=200, n_features=10)
        trainer.train(X, y)
        path = os.path.join(tmpdir, "model.pkl")
        trainer.save(path)
        return path, [f"feat_{i}" for i in range(10)]

    def test_signal_is_buy_sell_or_flat(self):
        from bist_bot.ml.signal import SignalGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            path, feature_names = self._train_and_save(tmpdir)
            gen = SignalGenerator(path)
            features = {name: float(val) for name, val in
                        zip(feature_names, np.random.randn(10))}
            result = gen.generate(features)
            assert result["signal"] in ("BUY", "SELL", "FLAT")
            assert 0.0 <= result["confidence"] <= 1.0

    def test_high_confidence_buys(self):
        """With threshold_buy=0.0, every prediction should be BUY."""
        from bist_bot.ml.signal import SignalGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            path, feature_names = self._train_and_save(tmpdir)
            gen = SignalGenerator(path, threshold_buy=0.0, threshold_sell=-1.0)
            features = {name: 0.0 for name in feature_names}
            result = gen.generate(features)
            assert result["signal"] == "BUY"

    def test_low_confidence_sells(self):
        """With threshold_sell=1.0, every prediction should be SELL."""
        from bist_bot.ml.signal import SignalGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            path, feature_names = self._train_and_save(tmpdir)
            gen = SignalGenerator(path, threshold_buy=2.0, threshold_sell=1.0)
            features = {name: 0.0 for name in feature_names}
            result = gen.generate(features)
            assert result["signal"] == "SELL"
