"""Orchestration entry point for the BIST trading bot.

This module wires together all components:

1. :class:`~bist_bot.data.websocket_client.MatriksWebSocketClient` — real-time
   WebSocket ingestion.
2. :class:`~bist_bot.processing.order_flow.OrderFlowProcessor` — session-level
   order-flow state.
3. :class:`~bist_bot.features.engineer.FeatureEngineer` — rolling-window feature
   computation.
4. :class:`~bist_bot.ml.signal.SignalGenerator` — XGBoost inference.
5. :class:`~bist_bot.execution.engine.ExecutionEngine` with
   :class:`~bist_bot.execution.risk.RiskManager` — order submission.

Configuration is read from environment variables so that secrets are never
hard-coded.  Required variables:

* ``MATRIKS_WS_URI`` — WebSocket endpoint (e.g.
  ``wss://ws.matriksiq.com/v2/stream``)
* ``MATRIKS_API_KEY`` — API key for Bearer authentication.
* ``BROKER_API_CLASS`` — Dotted import path for the broker API client class
  (must implement ``submit_order``, ``cancel_order``, ``submit_bracket``).

Optional variables (all have sensible defaults):

* ``BIST_SYMBOLS`` — Comma-separated BIST tickers (default: ``GARAN``).
* ``MODEL_PATH`` — Path to the trained model bundle
  (default: ``models/bofa_a1_tera_xgb.pkl``).
* ``REDIS_URL`` — Redis connection URL (default: ``redis://localhost:6379``).
* ``FEATURE_WINDOWS`` — Comma-separated window lengths in seconds
  (default: ``60,300``).
* ``ATR_LOOKBACK`` — Number of price samples for ATR calculation
  (default: ``14``).

Usage::

    export MATRIKS_WS_URI="wss://ws.matriksiq.com/v2/stream"
    export MATRIKS_API_KEY="<your-key>"
    export BROKER_API_CLASS="mybroker.api.BrokerClient"
    python -m bist_bot.main
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import os

import redis.asyncio as aioredis

from bist_bot.data.models import BrokerTick
from bist_bot.data.websocket_client import MatriksWebSocketClient
from bist_bot.execution.engine import ExecutionEngine
from bist_bot.execution.risk import RiskManager
from bist_bot.features.engineer import FeatureEngineer
from bist_bot.ml.signal import SignalGenerator
from bist_bot.processing.order_flow import OrderFlowProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ATR helper (simple rolling ATR over recent mid-price changes)
# ---------------------------------------------------------------------------

class _ATRTracker:
    """Lightweight ATR approximation using successive absolute returns."""

    def __init__(self, period: int = 14) -> None:
        self._period = period
        self._prices: list[float] = []

    def update(self, price: float) -> None:
        self._prices.append(price)
        if len(self._prices) > self._period + 1:
            self._prices.pop(0)

    def atr(self) -> float:
        if len(self._prices) < 2:
            return 1.0  # fallback: 1 TRY
        diffs = [abs(self._prices[i] - self._prices[i - 1]) for i in range(1, len(self._prices))]
        return sum(diffs) / len(diffs)


# ---------------------------------------------------------------------------
# Portfolio value stub (replace with real broker account query)
# ---------------------------------------------------------------------------

def _get_portfolio_value() -> float:
    """Return current portfolio value in TRY.

    Replace this stub with a live call to your broker account API.
    """
    return float(os.environ.get("PORTFOLIO_VALUE_TRY", "1000000"))


# ---------------------------------------------------------------------------
# Main async entry point
# ---------------------------------------------------------------------------

async def run(symbol: str) -> None:
    """Run the trading bot for *symbol* until cancelled."""
    model_path = os.environ.get("MODEL_PATH", "models/bofa_a1_tera_xgb.pkl")
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    windows = [
        int(w) for w in os.environ.get("FEATURE_WINDOWS", "60,300").split(",")
    ]
    atr_period = int(os.environ.get("ATR_LOOKBACK", "14"))

    # Broker API client — loaded dynamically from BROKER_API_CLASS env var.
    broker_api_client = _load_broker_api()

    # Component instances.
    ofp = OrderFlowProcessor(symbol=symbol)
    fe = FeatureEngineer(windows=windows)
    signal_gen = SignalGenerator(model_path)
    risk_mgr = RiskManager()
    exec_engine = ExecutionEngine(
        broker_api_client=broker_api_client,
        risk_manager=risk_mgr,
    )
    atr_tracker = _ATRTracker(period=atr_period)

    async def on_tick(tick: BrokerTick) -> None:
        atr_tracker.update(tick.price)
        ofp.update(tick)
        fe.ingest(tick)
        features = fe.compute_features()
        signal = signal_gen.generate(features)
        exec_engine.execute(
            signal=signal,
            symbol=symbol,
            entry_price=tick.price,
            atr=atr_tracker.atr(),
            portfolio_value=_get_portfolio_value(),
        )

    ws_uri = os.environ["MATRIKS_WS_URI"]
    api_key = os.environ["MATRIKS_API_KEY"]

    ws_client = MatriksWebSocketClient(
        uri=ws_uri,
        api_key=api_key,
        symbols=[symbol],
        redis_url=redis_url,
        on_tick=on_tick,
    )

    logger.info("Starting BIST trading bot for symbol=%s", symbol)
    await ws_client.run()


def _load_broker_api():
    """Instantiate the broker API client from the ``BROKER_API_CLASS`` env var.

    Falls back to a :class:`_NullBrokerAPI` stub if the variable is unset so
    that the bot can be tested without a live brokerage connection.
    """
    class_path = os.environ.get("BROKER_API_CLASS", "")
    if not class_path:
        logger.warning(
            "BROKER_API_CLASS not set — using NullBrokerAPI stub.  "
            "Orders will NOT be submitted to a real broker."
        )
        return _NullBrokerAPI()

    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()


class _NullBrokerAPI:
    """No-op broker API used when no real broker client is configured."""

    def submit_order(self, symbol, side, quantity, order_type):
        logger.info("[NullBroker] SUBMIT %s %s x%d (%s)", side, symbol, quantity, order_type)
        return {"order_id": "NULL-0"}

    def cancel_order(self, order_id):
        logger.info("[NullBroker] CANCEL %s", order_id)

    def submit_bracket(self, symbol, parent_order_id, stop_loss, take_profit):
        logger.info(
            "[NullBroker] BRACKET for %s: SL=%.2f TP=%.2f",
            parent_order_id, stop_loss, take_profit,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    symbols_raw = os.environ.get("BIST_SYMBOLS", "GARAN")
    symbols = [s.strip() for s in symbols_raw.split(",") if s.strip()]

    async def _run_all():
        await asyncio.gather(*[run(sym) for sym in symbols])

    asyncio.run(_run_all())


if __name__ == "__main__":
    # When the file is run directly (e.g. from an IDE "Run" button or
    # `python bist_bot/main.py`), make sure the project root is on sys.path
    # so that `import bist_bot.*` resolves correctly.
    import pathlib as _pathlib
    _project_root = str(_pathlib.Path(__file__).parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    main()
