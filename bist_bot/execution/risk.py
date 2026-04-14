"""Dynamic stop-loss / take-profit sizing and position sizing.

:class:`RiskManager` encapsulates all risk parameters and exposes two key
methods:

* :meth:`compute_stops` — derive stop-loss and take-profit levels based on
  the current ATR.
* :meth:`size_position` — calculate the number of shares to trade based on
  a fixed-fractional risk model.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class RiskManager:
    """ATR-based dynamic stop-loss, take-profit, and position sizing.

    Args:
        max_position_pct: Maximum fraction of portfolio value to risk per
            trade.  Default ``0.02`` (2 %).
        atr_multiplier_sl: ATR multiplier for stop-loss distance.
            Default ``1.5``.
        atr_multiplier_tp: ATR multiplier for take-profit distance.
            Default ``3.0`` (gives a 1:2 risk/reward ratio).
        max_latency_ms: Maximum acceptable round-trip latency in milliseconds
            before an order is cancelled.  Default ``50``.
    """

    def __init__(
        self,
        max_position_pct: float = 0.02,
        atr_multiplier_sl: float = 1.5,
        atr_multiplier_tp: float = 3.0,
        max_latency_ms: float = 50.0,
    ) -> None:
        if not (0.0 < max_position_pct <= 1.0):
            raise ValueError("max_position_pct must be in (0, 1]")
        if atr_multiplier_sl <= 0 or atr_multiplier_tp <= 0:
            raise ValueError("ATR multipliers must be positive")

        self.max_position_pct = max_position_pct
        self.atr_multiplier_sl = atr_multiplier_sl
        self.atr_multiplier_tp = atr_multiplier_tp
        self.max_latency_ms = max_latency_ms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_stops(
        self, entry_price: float, atr: float, side: str
    ) -> dict[str, float]:
        """Compute stop-loss and take-profit prices.

        Args:
            entry_price: Expected fill price.
            atr: Average True Range of the instrument (in price units).
            side: ``"BUY"`` or ``"SELL"``.

        Returns:
            Dictionary with ``stop_loss`` and ``take_profit`` keys.

        Raises:
            ValueError: If *side* is not ``"BUY"`` or ``"SELL"``.
        """
        if side not in ("BUY", "SELL"):
            raise ValueError(f"side must be 'BUY' or 'SELL', got {side!r}")
        if atr <= 0:
            raise ValueError("ATR must be positive")

        sl_dist = atr * self.atr_multiplier_sl
        tp_dist = atr * self.atr_multiplier_tp

        if side == "BUY":
            stop_loss = round(entry_price - sl_dist, 2)
            take_profit = round(entry_price + tp_dist, 2)
        else:
            stop_loss = round(entry_price + sl_dist, 2)
            take_profit = round(entry_price - tp_dist, 2)

        logger.debug(
            "Stops for %s @ %.2f (ATR=%.4f): SL=%.2f TP=%.2f",
            side, entry_price, atr, stop_loss, take_profit,
        )
        return {"stop_loss": stop_loss, "take_profit": take_profit}

    def size_position(
        self, portfolio_value: float, entry_price: float, atr: float
    ) -> int:
        """Compute the number of shares to buy/sell.

        Uses a fixed-fractional model: risk at most ``max_position_pct`` of
        portfolio value, where the risk per share equals the stop-loss
        distance (``atr × atr_multiplier_sl``).

        Args:
            portfolio_value: Current total portfolio value in TRY.
            entry_price: Expected fill price (used for sanity-checking only).
            atr: Average True Range of the instrument.

        Returns:
            Integer number of shares (minimum 1).
        """
        if portfolio_value <= 0:
            raise ValueError("portfolio_value must be positive")

        risk_amount = portfolio_value * self.max_position_pct
        risk_per_share = atr * self.atr_multiplier_sl
        shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 1
        shares = max(1, shares)

        logger.debug(
            "Position size: %d shares (portfolio=%.0f, risk_amount=%.0f, atr=%.4f)",
            shares, portfolio_value, risk_amount, atr,
        )
        return shares
