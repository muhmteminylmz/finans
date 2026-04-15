"""Rolling-window feature engineering from raw broker ticks.

:class:`FeatureEngineer` maintains an in-memory ring buffer of recent ticks
and computes a rich set of features whenever :meth:`compute_features` is
called.  Features are grouped by broker and by time window, and a composite
*smart money pressure index* aggregates signals across all tracked brokers.

Feature naming convention::

    {BROKER_CODE}_{w<N>s}_{metric}

e.g. ``BOFA_w60s_net_vol``, ``A1KP_w300s_aggr_pressure``.

Composite features::

    smart_money_index_w60s
    smart_money_index_w300s
"""

from __future__ import annotations

import logging

import pandas as pd

from bist_bot.data.models import BrokerTick, TARGET_BROKERS

logger = logging.getLogger(__name__)

# Column layout of the internal tick buffer rows.
_COLS = ["ts", "broker", "side", "vol", "price", "is_market"]


class FeatureEngineer:
    """Compute rolling broker order-flow features from a live tick stream.

    Args:
        windows: List of window lengths in seconds.  Default ``[60, 300]``
            produces 1-minute and 5-minute features.
    """

    def __init__(self, windows: list[int] | None = None) -> None:
        self.windows: list[int] = windows if windows is not None else [60, 300]
        # Each entry: (unix_ts, broker_code, side, volume, price, is_market:int)
        self._tick_buffer: list[tuple] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, tick: BrokerTick) -> None:
        """Append a new tick to the rolling buffer.

        Args:
            tick: Incoming broker tick.
        """
        ts = pd.Timestamp(tick.timestamp).timestamp()
        is_market = 1 if tick.order_type == "MARKET" else 0
        self._tick_buffer.append(
            (ts, tick.broker_code, tick.side, tick.volume, tick.price, is_market)
        )

    def compute_features(self) -> dict[str, float]:
        """Compute all features for the current instant.

        Uses the timestamp of the most recent tick as the reference point
        so that feature windows are aligned to the live data stream rather
        than to local wall-clock time.  This prevents window miscalculation
        when feed timestamps lag behind or run ahead of the host clock.

        Also evicts ticks older than the largest configured window to keep
        memory usage bounded.

        Returns:
            A flat dictionary mapping feature name → float value.
        """
        if self._tick_buffer:
            now_ts = self._tick_buffer[-1][0]
        else:
            now_ts = pd.Timestamp.now("UTC").timestamp()
        max_window = max(self.windows)

        # Evict stale ticks.
        self._tick_buffer = [r for r in self._tick_buffer if r[0] >= now_ts - max_window]

        features: dict[str, float] = {}

        for window_sec in self.windows:
            label = f"w{window_sec}s"
            df = self._window_df(now_ts, window_sec)

            for broker_code in TARGET_BROKERS.values():
                bdf = df[df["broker"] == broker_code]
                buy_df = bdf[bdf["side"] == "BUY"]
                sell_df = bdf[bdf["side"] == "SELL"]

                buy_vol = float(buy_df["vol"].sum())
                sell_vol = float(sell_df["vol"].sum())
                market_buy_vol = float(buy_df.loc[buy_df["is_market"] == 1, "vol"].sum())
                market_sell_vol = float(sell_df.loc[sell_df["is_market"] == 1, "vol"].sum())

                # Tick count
                features[f"{broker_code}_{label}_tick_count"] = float(len(bdf))

                # Net volume (signed; positive = net buying)
                features[f"{broker_code}_{label}_net_vol"] = buy_vol - sell_vol

                # Aggression ratio (buy / sell); 0.0 when no sells
                features[f"{broker_code}_{label}_aggr_ratio"] = (
                    buy_vol / sell_vol if sell_vol > 0.0 else 0.0
                )

                # Market-order volumes (aggressive orders)
                features[f"{broker_code}_{label}_market_buy_vol"] = market_buy_vol
                features[f"{broker_code}_{label}_market_sell_vol"] = market_sell_vol

                # Smart-money pressure: aggressive buys minus aggressive sells
                features[f"{broker_code}_{label}_aggr_pressure"] = (
                    market_buy_vol - market_sell_vol
                )

                # VWAP spread proxy (buy_vwap − sell_vwap)
                buy_vwap = self._vwap(buy_df)
                sell_vwap = self._vwap(sell_df)
                features[f"{broker_code}_{label}_vwap_spread"] = buy_vwap - sell_vwap

        # Composite cross-broker smart money index per window.
        for window_sec in self.windows:
            label = f"w{window_sec}s"
            features[f"smart_money_index_{label}"] = sum(
                features.get(f"{code}_{label}_aggr_pressure", 0.0)
                for code in TARGET_BROKERS.values()
            )

        return features

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _window_df(self, now_ts: float, window_sec: int) -> pd.DataFrame:
        cutoff = now_ts - window_sec
        rows = [r for r in self._tick_buffer if r[0] >= cutoff]
        if rows:
            return pd.DataFrame(rows, columns=_COLS)
        return pd.DataFrame(columns=_COLS).astype(
            {"ts": float, "broker": str, "side": str,
             "vol": float, "price": float, "is_market": int}
        )

    @staticmethod
    def _vwap(df: pd.DataFrame) -> float:
        """Volume-weighted average price for a slice of the tick buffer."""
        total_vol = df["vol"].sum()
        if total_vol == 0.0:
            return 0.0
        return float((df["vol"] * df["price"]).sum() / total_vol)
