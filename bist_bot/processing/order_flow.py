"""Order-flow state machine for tracking per-broker net positioning.

The :class:`OrderFlowProcessor` maintains a live, cumulative snapshot of each
tracked broker's buying and selling activity for a single symbol.  Call
:meth:`update` for every incoming :class:`~bist_bot.data.models.BrokerTick`
and query :meth:`net_flow` at any time to obtain current metrics.

This component intentionally holds *session-level* state (from market open).
For *window-based* metrics (e.g. rolling 1-min pressure) see
:mod:`bist_bot.features.engineer`.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from bist_bot.data.models import BrokerTick

logger = logging.getLogger(__name__)


def _empty_broker_state() -> dict:
    return {
        "buy_vol": 0.0,
        "sell_vol": 0.0,
        "buy_vwap_num": 0.0,   # numerator for VWAP (sum of price × volume)
        "sell_vwap_num": 0.0,
        "tick_count": 0,
    }


class OrderFlowProcessor:
    """Maintains a cumulative, session-scoped order-flow snapshot per broker.

    Args:
        symbol: BIST ticker this processor is tracking (informational only).
    """

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self._state: dict[str, dict] = defaultdict(_empty_broker_state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, tick: BrokerTick) -> None:
        """Incorporate a new broker tick into the running state.

        Args:
            tick: A :class:`~bist_bot.data.models.BrokerTick` for this symbol.
        """
        if tick.symbol != self.symbol:
            logger.debug(
                "OrderFlowProcessor for %s received tick for %s — ignored.",
                self.symbol,
                tick.symbol,
            )
            return

        s = self._state[tick.broker_code]
        s["tick_count"] += 1
        if tick.side == "BUY":
            s["buy_vol"] += tick.volume
            s["buy_vwap_num"] += tick.volume * tick.price
        else:
            s["sell_vol"] += tick.volume
            s["sell_vwap_num"] += tick.volume * tick.price

    def net_flow(self, broker_code: str) -> dict:
        """Return current flow metrics for *broker_code*.

        Returns a dictionary with the following keys:

        * ``broker`` – broker code string
        * ``net_volume`` – buy_vol − sell_vol (positive = net buying)
        * ``buy_vol`` / ``sell_vol`` – cumulative volumes
        * ``buy_vwap`` / ``sell_vwap`` – volume-weighted average prices
        * ``aggression_ratio`` – buy_vol / sell_vol (``None`` if no sells)
        * ``tick_count`` – total number of ticks processed

        Args:
            broker_code: One of the codes in
                :data:`~bist_bot.data.models.TARGET_BROKERS` values.
        """
        s = self._state[broker_code]
        buy_vol = s["buy_vol"]
        sell_vol = s["sell_vol"]

        buy_vwap = s["buy_vwap_num"] / buy_vol if buy_vol > 0 else 0.0
        sell_vwap = s["sell_vwap_num"] / sell_vol if sell_vol > 0 else 0.0
        aggression_ratio = (buy_vol / sell_vol) if sell_vol > 0 else None

        return {
            "broker": broker_code,
            "net_volume": buy_vol - sell_vol,
            "buy_vol": buy_vol,
            "sell_vol": sell_vol,
            "buy_vwap": buy_vwap,
            "sell_vwap": sell_vwap,
            "aggression_ratio": aggression_ratio,
            "tick_count": s["tick_count"],
        }

    def all_net_flows(self) -> list[dict]:
        """Return :meth:`net_flow` results for every broker seen so far."""
        return [self.net_flow(code) for code in self._state]

    def reset(self) -> None:
        """Clear all accumulated state (e.g. called at market open each day)."""
        self._state.clear()
        logger.info("OrderFlowProcessor for %s reset.", self.symbol)
