"""Unit tests for FeatureEngineer."""

import time

import pytest
from bist_bot.data.models import BrokerTick, TARGET_BROKERS
from bist_bot.features.engineer import FeatureEngineer


def _now_iso():
    from datetime import timezone, datetime
    return datetime.now(timezone.utc).isoformat()


def _tick(side, vol, price, broker="BOFA", order_type="LIMIT"):
    return BrokerTick(
        timestamp=_now_iso(),
        symbol="GARAN",
        broker_code=broker,
        side=side,
        volume=vol,
        price=price,
        order_type=order_type,
    )


class TestFeatureEngineer:
    def setup_method(self):
        self.fe = FeatureEngineer(windows=[60, 300])

    def test_empty_features_are_zero(self):
        features = self.fe.compute_features()
        assert features["BOFA_w60s_net_vol"] == 0.0
        assert features["smart_money_index_w60s"] == 0.0

    def test_buy_tick_shows_positive_net_vol(self):
        self.fe.ingest(_tick("BUY", 1000, 45.0))
        features = self.fe.compute_features()
        assert features["BOFA_w60s_net_vol"] == pytest.approx(1000.0)
        assert features["BOFA_w300s_net_vol"] == pytest.approx(1000.0)

    def test_sell_tick_shows_negative_net_vol(self):
        self.fe.ingest(_tick("SELL", 500, 45.0))
        features = self.fe.compute_features()
        assert features["BOFA_w60s_net_vol"] == pytest.approx(-500.0)

    def test_market_order_captured_in_aggr_pressure(self):
        self.fe.ingest(_tick("BUY", 200, 45.0, order_type="MARKET"))
        self.fe.ingest(_tick("SELL", 100, 45.0, order_type="MARKET"))
        features = self.fe.compute_features()
        assert features["BOFA_w60s_aggr_pressure"] == pytest.approx(100.0)

    def test_limit_order_not_in_market_vol(self):
        self.fe.ingest(_tick("BUY", 500, 45.0, order_type="LIMIT"))
        features = self.fe.compute_features()
        assert features["BOFA_w60s_market_buy_vol"] == 0.0

    def test_smart_money_index_aggregates_brokers(self):
        self.fe.ingest(_tick("BUY", 200, 45.0, broker="BOFA", order_type="MARKET"))
        self.fe.ingest(_tick("BUY", 100, 45.0, broker="A1KP", order_type="MARKET"))
        features = self.fe.compute_features()
        assert features["smart_money_index_w60s"] == pytest.approx(300.0)

    def test_aggression_ratio_zero_when_no_sells(self):
        self.fe.ingest(_tick("BUY", 100, 45.0))
        features = self.fe.compute_features()
        assert features["BOFA_w60s_aggr_ratio"] == 0.0

    def test_aggression_ratio_calculated(self):
        self.fe.ingest(_tick("BUY", 300, 45.0))
        self.fe.ingest(_tick("SELL", 100, 45.0))
        features = self.fe.compute_features()
        assert features["BOFA_w60s_aggr_ratio"] == pytest.approx(3.0)

    def test_feature_keys_present_for_all_brokers_and_windows(self):
        features = self.fe.compute_features()
        for broker in TARGET_BROKERS.values():
            for w in [60, 300]:
                assert f"{broker}_w{w}s_net_vol" in features
                assert f"{broker}_w{w}s_aggr_pressure" in features
        assert "smart_money_index_w60s" in features
        assert "smart_money_index_w300s" in features

    def test_vwap_spread_feature_present(self):
        self.fe.ingest(_tick("BUY", 100, 50.0))
        self.fe.ingest(_tick("SELL", 100, 49.0))
        features = self.fe.compute_features()
        assert "BOFA_w60s_vwap_spread" in features
        assert features["BOFA_w60s_vwap_spread"] == pytest.approx(1.0)

    def test_tick_count_feature(self):
        self.fe.ingest(_tick("BUY", 100, 45.0))
        self.fe.ingest(_tick("SELL", 50, 45.0))
        features = self.fe.compute_features()
        assert features["BOFA_w60s_tick_count"] == 2.0
