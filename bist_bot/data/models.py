"""Shared constants and data models for the BIST trading bot."""

from dataclasses import dataclass, field

# Broker codes used in the Matriks IQ / Aracı Kurum Dağılımı feed.
# Keys are human-readable names; values are the codes sent by the data provider.
TARGET_BROKERS: dict[str, str] = {
    "BofA": "BOFA",
    "A1Capital": "A1KP",
    "Tera": "TERA",
}


@dataclass
class BrokerTick:
    """One broker-level execution event arriving from the market-data feed."""

    timestamp: str    # ISO-8601 string from the feed
    symbol: str       # BIST ticker, e.g. "GARAN"
    broker_code: str  # One of TARGET_BROKERS.values()
    side: str         # "BUY" or "SELL"
    volume: float     # Executed lot size
    price: float      # Execution price (TRY)
    order_type: str = field(default="LIMIT")  # "MARKET" or "LIMIT"
