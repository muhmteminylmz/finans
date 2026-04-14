"""Unit tests for OrderFlowProcessor."""

import pytest
from bist_bot.data.models import BrokerTick
from bist_bot.processing.order_flow import OrderFlowProcessor


def _tick(side, vol, price, broker="BOFA", order_type="LIMIT", symbol="GARAN"):
    return BrokerTick(
        timestamp="2024-01-15T10:00:00Z",
        symbol=symbol,
        broker_code=broker,
        side=side,
        volume=vol,
        price=price,
        order_type=order_type,
    )


class TestOrderFlowProcessor:
    def setup_method(self):
        self.ofp = OrderFlowProcessor(symbol="GARAN")

    def test_initial_state_is_zero(self):
        flow = self.ofp.net_flow("BOFA")
        assert flow["net_volume"] == 0.0
        assert flow["buy_vol"] == 0.0
        assert flow["sell_vol"] == 0.0
        assert flow["aggression_ratio"] is None

    def test_buy_tick_updates_buy_vol(self):
        self.ofp.update(_tick("BUY", 1000, 45.20))
        flow = self.ofp.net_flow("BOFA")
        assert flow["buy_vol"] == 1000.0
        assert flow["sell_vol"] == 0.0
        assert flow["net_volume"] == 1000.0

    def test_sell_tick_updates_sell_vol(self):
        self.ofp.update(_tick("SELL", 500, 45.10))
        flow = self.ofp.net_flow("BOFA")
        assert flow["sell_vol"] == 500.0
        assert flow["net_volume"] == -500.0

    def test_net_volume_mixed(self):
        self.ofp.update(_tick("BUY", 1000, 45.20))
        self.ofp.update(_tick("SELL", 400, 45.15))
        flow = self.ofp.net_flow("BOFA")
        assert flow["net_volume"] == pytest.approx(600.0)

    def test_buy_vwap(self):
        self.ofp.update(_tick("BUY", 100, 10.0))
        self.ofp.update(_tick("BUY", 200, 20.0))
        flow = self.ofp.net_flow("BOFA")
        expected_vwap = (100 * 10 + 200 * 20) / 300
        assert flow["buy_vwap"] == pytest.approx(expected_vwap)

    def test_aggression_ratio(self):
        self.ofp.update(_tick("BUY", 300, 45.0))
        self.ofp.update(_tick("SELL", 100, 45.0))
        flow = self.ofp.net_flow("BOFA")
        assert flow["aggression_ratio"] == pytest.approx(3.0)

    def test_different_brokers_isolated(self):
        self.ofp.update(_tick("BUY", 1000, 45.0, broker="BOFA"))
        self.ofp.update(_tick("SELL", 500, 45.0, broker="A1KP"))
        assert self.ofp.net_flow("BOFA")["buy_vol"] == 1000.0
        assert self.ofp.net_flow("A1KP")["sell_vol"] == 500.0
        assert self.ofp.net_flow("BOFA")["sell_vol"] == 0.0

    def test_tick_for_wrong_symbol_ignored(self):
        self.ofp.update(_tick("BUY", 1000, 45.0, symbol="THYAO"))
        assert self.ofp.net_flow("BOFA")["buy_vol"] == 0.0

    def test_reset_clears_state(self):
        self.ofp.update(_tick("BUY", 1000, 45.0))
        self.ofp.reset()
        assert self.ofp.net_flow("BOFA")["buy_vol"] == 0.0

    def test_all_net_flows(self):
        self.ofp.update(_tick("BUY", 100, 10.0, broker="BOFA"))
        self.ofp.update(_tick("SELL", 50, 10.0, broker="A1KP"))
        flows = {f["broker"]: f for f in self.ofp.all_net_flows()}
        assert "BOFA" in flows
        assert "A1KP" in flows

    def test_tick_count(self):
        self.ofp.update(_tick("BUY", 100, 10.0))
        self.ofp.update(_tick("BUY", 200, 11.0))
        assert self.ofp.net_flow("BOFA")["tick_count"] == 2
