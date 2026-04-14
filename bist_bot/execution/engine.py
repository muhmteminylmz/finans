"""Order submission and latency-gated execution engine.

:class:`ExecutionEngine` translates a signal dict from
:class:`~bist_bot.ml.signal.SignalGenerator` into live orders via an
abstract broker API client.  It enforces the latency limit configured in
:class:`~bist_bot.execution.risk.RiskManager` by cancelling an order that
was submitted too slowly.

The *broker_api_client* is intentionally left as a duck-typed interface so
the engine can be used with any broker SDK (e.g. Matriks, IS Yatırım REST
API, IBKR TWS).  The client must implement:

* ``submit_order(symbol, side, quantity, order_type) → dict``  (returns dict
  with ``"order_id"`` key)
* ``cancel_order(order_id) → None``
* ``submit_bracket(symbol, parent_order_id, stop_loss, take_profit) → None``
"""

from __future__ import annotations

import logging
import time

from bist_bot.execution.risk import RiskManager

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """Submit market orders and attach ATR-based bracket orders.

    Args:
        broker_api_client: Duck-typed object implementing ``submit_order``,
            ``cancel_order``, and ``submit_bracket``.
        risk_manager: A configured :class:`~bist_bot.execution.risk.RiskManager`
            instance shared with the caller.
    """

    def __init__(self, broker_api_client, risk_manager: RiskManager) -> None:
        self._api = broker_api_client
        self._risk = risk_manager

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        signal: dict,
        symbol: str,
        entry_price: float,
        atr: float,
        portfolio_value: float,
    ) -> dict | None:
        """Execute a signal if it is not ``"FLAT"``.

        Args:
            signal: Output of
                :meth:`~bist_bot.ml.signal.SignalGenerator.generate`, e.g.
                ``{"signal": "BUY", "confidence": 0.71}``.
            symbol: BIST ticker (e.g. ``"GARAN"``).
            entry_price: Current mid-price used to compute stop levels.
            atr: Average True Range at time of signal.
            portfolio_value: Current total portfolio value in TRY.

        Returns:
            The order dict returned by the broker API, or ``None`` if the
            signal is ``"FLAT"`` or the order was cancelled due to latency.
        """
        direction = signal.get("signal")
        if direction == "FLAT":
            return None

        if direction not in ("BUY", "SELL"):
            logger.warning("Unknown signal direction %r — skipping.", direction)
            return None

        qty = self._risk.size_position(portfolio_value, entry_price, atr)
        stops = self._risk.compute_stops(entry_price, atr, direction)

        t0 = time.perf_counter()

        order = self._api.submit_order(
            symbol=symbol,
            side=direction,
            quantity=qty,
            order_type="MARKET",
        )

        latency_ms = (time.perf_counter() - t0) * 1_000
        logger.debug("Order %s submitted in %.2f ms", order.get("order_id"), latency_ms)

        if latency_ms > self._risk.max_latency_ms:
            logger.warning(
                "Latency %.2f ms > limit %.2f ms — cancelling order %s.",
                latency_ms,
                self._risk.max_latency_ms,
                order.get("order_id"),
            )
            self._api.cancel_order(order["order_id"])
            return None

        self._api.submit_bracket(
            symbol=symbol,
            parent_order_id=order["order_id"],
            stop_loss=stops["stop_loss"],
            take_profit=stops["take_profit"],
        )

        logger.info(
            "Executed %s %s x%d @ ~%.2f | SL=%.2f TP=%.2f (conf=%.4f)",
            direction, symbol, qty, entry_price,
            stops["stop_loss"], stops["take_profit"],
            signal.get("confidence", 0.0),
        )
        return order
