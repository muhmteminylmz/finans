"""Async WebSocket client for the Matriks IQ broker-distribution feed.

Connects to the Matriks IQ (or compatible) WebSocket endpoint, subscribes to
the ``broker_distribution`` channel for the requested symbols, and pushes
every matching broker tick into a Redis Stream for downstream consumers.

Usage example::

    import asyncio
    from bist_bot.data.websocket_client import MatriksWebSocketClient

    client = MatriksWebSocketClient(
        uri="wss://ws.matriksiq.com/v2/stream",
        api_key="YOUR_API_KEY",
        symbols=["GARAN", "AKBNK"],
    )
    asyncio.run(client.run())
"""

import asyncio
import json
import logging
from dataclasses import asdict

import redis.asyncio as aioredis
import websockets
from websockets.exceptions import ConnectionClosedError, WebSocketException

from bist_bot.data.models import BrokerTick, TARGET_BROKERS

logger = logging.getLogger(__name__)

# Seconds to wait before attempting a reconnect after a connection failure.
_RECONNECT_BASE_DELAY: float = 1.0
_RECONNECT_MAX_DELAY: float = 60.0

# Redis stream key template — one stream per symbol.
_STREAM_KEY_TPL = "broker_ticks:{symbol}"
# Maximum number of entries kept in each Redis stream (FIFO eviction).
_STREAM_MAXLEN = 100_000


class MatriksWebSocketClient:
    """Async WebSocket client that ingests broker-level tick data into Redis.

    Args:
        uri: WebSocket endpoint URL.
        api_key: Matriks IQ API key used for Bearer authentication.
        symbols: List of BIST tickers to subscribe to (e.g. ``["GARAN"]``).
        redis_url: Redis connection URL (default: ``redis://localhost:6379``).
        on_tick: Optional async callback invoked for every parsed
            :class:`~bist_bot.data.models.BrokerTick`.  Useful for hooking the
            downstream processing pipeline without polling Redis.
    """

    def __init__(
        self,
        uri: str,
        api_key: str,
        symbols: list[str],
        redis_url: str = "redis://localhost:6379",
        on_tick=None,
    ) -> None:
        self.uri = uri
        self._api_key = api_key
        self.symbols = symbols
        self._redis_url = redis_url
        self._on_tick = on_tick
        self._redis: aioredis.Redis | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Connect and keep running with automatic reconnect on failure."""
        self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
        delay = _RECONNECT_BASE_DELAY
        while True:
            try:
                await self._connect_once()
                delay = _RECONNECT_BASE_DELAY  # reset on clean disconnect
            except (ConnectionClosedError, WebSocketException, OSError) as exc:
                logger.warning("WebSocket disconnected: %s — retrying in %.1fs", exc, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, _RECONNECT_MAX_DELAY)
            except asyncio.CancelledError:
                logger.info("WebSocket client cancelled — shutting down.")
                break

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _connect_once(self) -> None:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        async with websockets.connect(self.uri, additional_headers=headers) as ws:
            logger.info("WebSocket connected to %s", self.uri)
            await self._subscribe(ws)
            async for raw in ws:
                await self._handle_message(raw)

    async def _subscribe(self, ws) -> None:
        payload = {
            "action": "subscribe",
            "channels": ["broker_distribution"],
            "symbols": self.symbols,
        }
        await ws.send(json.dumps(payload))
        logger.debug("Subscribed to broker_distribution for %s", self.symbols)

    async def _handle_message(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Malformed JSON from feed — skipping: %.120s", raw)
            await self._dead_letter(raw)
            return

        if data.get("type") != "BROKER_DIST":
            return

        broker_code = data.get("broker_code", "")
        if broker_code not in TARGET_BROKERS.values():
            return

        try:
            tick = BrokerTick(
                timestamp=data["timestamp"],
                symbol=data["symbol"],
                broker_code=broker_code,
                side=data["side"],
                volume=float(data["volume"]),
                price=float(data["price"]),
                order_type=data.get("order_type", "LIMIT"),
            )
        except (KeyError, ValueError) as exc:
            logger.warning("Cannot parse tick: %s — raw: %.120s", exc, raw)
            await self._dead_letter(raw)
            return

        # Persist to Redis Stream.
        stream_key = _STREAM_KEY_TPL.format(symbol=tick.symbol)
        await self._redis.xadd(stream_key, asdict(tick), maxlen=_STREAM_MAXLEN)

        # Fire optional downstream callback.
        if self._on_tick is not None:
            await self._on_tick(tick)

    async def _dead_letter(self, raw: str) -> None:
        """Push unparseable messages to a dead-letter stream for inspection."""
        await self._redis.xadd("broker_ticks:dead_letter", {"raw": raw}, maxlen=10_000)
