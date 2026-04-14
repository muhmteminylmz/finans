"""Entry point — click the Run ▶ button in your IDE to start the trading bot.

You can also run it from the terminal::

    python run.py

Configuration is done via environment variables (all optional except the
WebSocket credentials when connecting to a live feed):

    MATRIKS_WS_URI      WebSocket endpoint  (e.g. wss://ws.matriksiq.com/v2/stream)
    MATRIKS_API_KEY     API key for the data feed
    BIST_SYMBOLS        Comma-separated tickers  (default: GARAN)
    MODEL_PATH          Path to the trained model  (default: models/bofa_a1_tera_xgb.pkl)
    BROKER_API_CLASS    Dotted import path for the broker client  (omit → NullBroker stub)
    REDIS_URL           Redis connection URL  (default: redis://localhost:6379)
    PORTFOLIO_VALUE_TRY Portfolio value in TRY for position sizing  (default: 1000000)
    FEATURE_WINDOWS     Comma-separated window sizes in seconds  (default: 60,300)
    ATR_LOOKBACK        ATR period  (default: 14)

If MODEL_PATH does not exist yet, train a model first by running train.py.
"""

from __future__ import annotations

import os
import sys

# Make sure the project root is on sys.path so 'bist_bot' can always be found,
# regardless of where Python is launched from.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bist_bot.main import main  # noqa: E402

if __name__ == "__main__":
    main()
