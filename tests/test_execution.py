"""Unit tests for RiskManager and ExecutionEngine."""

import pytest
from unittest.mock import MagicMock, call

from bist_bot.execution.risk import RiskManager
from bist_bot.execution.engine import ExecutionEngine


# ---------------------------------------------------------------------------
# RiskManager tests
# ---------------------------------------------------------------------------

class TestRiskManager:
    def setup_method(self):
        self.rm = RiskManager(
            max_position_pct=0.02,
            atr_multiplier_sl=1.5,
            atr_multiplier_tp=3.0,
            max_latency_ms=50.0,
        )

    def test_buy_stop_loss_below_entry(self):
        stops = self.rm.compute_stops(100.0, 2.0, "BUY")
        assert stops["stop_loss"] == pytest.approx(97.0)   # 100 - 1.5*2
        assert stops["take_profit"] == pytest.approx(106.0)  # 100 + 3.0*2

    def test_sell_stop_loss_above_entry(self):
        stops = self.rm.compute_stops(100.0, 2.0, "SELL")
        assert stops["stop_loss"] == pytest.approx(103.0)
        assert stops["take_profit"] == pytest.approx(94.0)

    def test_invalid_side_raises(self):
        with pytest.raises(ValueError):
            self.rm.compute_stops(100.0, 2.0, "FLAT")

    def test_invalid_atr_raises(self):
        with pytest.raises(ValueError):
            self.rm.compute_stops(100.0, 0.0, "BUY")

    def test_position_size_positive(self):
        qty = self.rm.size_position(1_000_000, 45.0, 1.0)
        # risk_amount = 20000; risk_per_share = 1.5 → shares = 13333
        assert qty == 13333

    def test_position_size_minimum_one(self):
        # Extremely large ATR should still give at least 1 share
        qty = self.rm.size_position(100, 45.0, 1_000_000.0)
        assert qty == 1

    def test_invalid_max_position_pct(self):
        with pytest.raises(ValueError):
            RiskManager(max_position_pct=0.0)
        with pytest.raises(ValueError):
            RiskManager(max_position_pct=1.1)

    def test_invalid_atr_multiplier(self):
        with pytest.raises(ValueError):
            RiskManager(atr_multiplier_sl=-1.0)

    def test_invalid_portfolio_value(self):
        with pytest.raises(ValueError):
            self.rm.size_position(0.0, 45.0, 1.0)


# ---------------------------------------------------------------------------
# ExecutionEngine tests
# ---------------------------------------------------------------------------

class TestExecutionEngine:
    def setup_method(self):
        self.mock_api = MagicMock()
        self.mock_api.submit_order.return_value = {"order_id": "ORD-001"}
        self.rm = RiskManager(max_position_pct=0.02, atr_multiplier_sl=1.5,
                              atr_multiplier_tp=3.0, max_latency_ms=5000.0)
        self.engine = ExecutionEngine(
            broker_api_client=self.mock_api,
            risk_manager=self.rm,
        )

    def test_flat_signal_no_order_submitted(self):
        result = self.engine.execute(
            {"signal": "FLAT", "confidence": 0.50},
            "GARAN", 45.0, 1.0, 1_000_000,
        )
        assert result is None
        self.mock_api.submit_order.assert_not_called()

    def test_buy_signal_submits_order(self):
        result = self.engine.execute(
            {"signal": "BUY", "confidence": 0.70},
            "GARAN", 45.0, 1.0, 1_000_000,
        )
        assert result == {"order_id": "ORD-001"}
        self.mock_api.submit_order.assert_called_once_with(
            symbol="GARAN", side="BUY",
            quantity=self.rm.size_position(1_000_000, 45.0, 1.0),
            order_type="MARKET",
        )

    def test_sell_signal_submits_order(self):
        self.engine.execute(
            {"signal": "SELL", "confidence": 0.30},
            "GARAN", 45.0, 1.0, 1_000_000,
        )
        self.mock_api.submit_order.assert_called_once()
        args = self.mock_api.submit_order.call_args
        assert args.kwargs["side"] == "SELL"

    def test_bracket_order_submitted_after_market_order(self):
        self.engine.execute(
            {"signal": "BUY", "confidence": 0.70},
            "GARAN", 100.0, 2.0, 1_000_000,
        )
        self.mock_api.submit_bracket.assert_called_once_with(
            symbol="GARAN",
            parent_order_id="ORD-001",
            stop_loss=pytest.approx(97.0),
            take_profit=pytest.approx(106.0),
        )

    def test_unknown_signal_direction_returns_none(self):
        result = self.engine.execute(
            {"signal": "HOLD", "confidence": 0.55},
            "GARAN", 45.0, 1.0, 1_000_000,
        )
        assert result is None
        self.mock_api.submit_order.assert_not_called()

    def test_high_latency_cancels_order(self):
        """Simulate high latency by configuring a very tight limit."""
        rm = RiskManager(max_latency_ms=0.0)  # 0 ms → always cancel
        engine = ExecutionEngine(self.mock_api, rm)
        result = engine.execute(
            {"signal": "BUY", "confidence": 0.70},
            "GARAN", 45.0, 1.0, 1_000_000,
        )
        assert result is None
        self.mock_api.cancel_order.assert_called_once_with("ORD-001")
        self.mock_api.submit_bracket.assert_not_called()
