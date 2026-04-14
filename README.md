# BIST Algorithmic Trading Bot

A real-time algorithmic trading system for **Borsa Istanbul (BIST)** that tracks the institutional order flow of specific major brokerage firms — **Bank of America (BofA)**, **A1 Capital**, and **Tera Yatırım** — and generates buy/sell signals using an XGBoost classifier.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        BIST Trading Bot                          │
├──────────────┬──────────────┬────────────────┬──────────────────┤
│  Data Layer  │ Processing   │  ML / Signal   │  Execution       │
│              │   Layer      │   Layer        │  Layer           │
│  WebSocket   │  Order Flow  │  Feature Eng.  │  Broker API      │
│  (Matriks/   │  Processor   │  XGBoost       │  Order Manager   │
│   IdealData) │  Normalizer  │  Signal Gen.   │  Risk Manager    │
└──────────────┴──────────────┴────────────────┴──────────────────┘
         │             │               │                │
         └─────────────┴───────────────┴────────────────┘
                              Redis (tick cache)
                              PostgreSQL / TimescaleDB (historical)
```

---

## Project Layout

```
bist_bot/
├── data/
│   ├── models.py             # BrokerTick dataclass & TARGET_BROKERS constant
│   └── websocket_client.py   # Async WebSocket client → Redis Stream ingestion
├── processing/
│   └── order_flow.py         # Session-level per-broker net flow state machine
├── features/
│   └── engineer.py           # Rolling-window feature engineering
├── ml/
│   ├── trainer.py            # XGBoost training pipeline (walk-forward CV)
│   └── signal.py             # Real-time signal generation from trained model
├── execution/
│   ├── risk.py               # ATR-based stop-loss / take-profit / position sizing
│   └── engine.py             # Order submission with latency gating
└── main.py                   # Async orchestration loop
tests/
├── test_order_flow.py
├── test_features.py
├── test_ml.py
└── test_execution.py
```

---

## Components

### 1 · Data Ingestion (`bist_bot/data/`)

`MatriksWebSocketClient` connects to the **Matriks IQ** (or compatible) WebSocket
endpoint and subscribes to the `broker_distribution` channel.  Every tick whose
`broker_code` matches one of BofA / A1 Capital / Tera is parsed into a
`BrokerTick` and pushed to a **Redis Stream** (capped at 100 k entries per symbol).
Malformed messages go to a dead-letter stream.  Automatic exponential-backoff
reconnect handles transient network failures.

### 2 · Order Flow Processing (`bist_bot/processing/`)

`OrderFlowProcessor` maintains a cumulative, session-scoped snapshot of each
broker's buying and selling activity (volumes, VWAP numerators).  Call
`update(tick)` on every tick; query `net_flow(broker_code)` for the current
snapshot including net volume, per-side VWAP, and aggression ratio.

### 3 · Feature Engineering (`bist_bot/features/`)

`FeatureEngineer` computes rolling-window metrics for configurable time windows
(default: 1 min and 5 min):

| Feature | Description |
|---|---|
| `{BROKER}_w{N}s_net_vol` | Signed net volume (buy − sell) |
| `{BROKER}_w{N}s_aggr_ratio` | Buy volume / sell volume |
| `{BROKER}_w{N}s_market_buy_vol` | Aggressive market-buy volume |
| `{BROKER}_w{N}s_market_sell_vol` | Aggressive market-sell volume |
| `{BROKER}_w{N}s_aggr_pressure` | Market buy − market sell ("smart money") |
| `{BROKER}_w{N}s_vwap_spread` | Buy VWAP − sell VWAP |
| `{BROKER}_w{N}s_tick_count` | Number of ticks in window |
| `smart_money_index_w{N}s` | Sum of `aggr_pressure` across all brokers |

### 4 · ML Signal Generation (`bist_bot/ml/`)

`SignalModelTrainer` trains an `XGBClassifier` to predict whether the mid-price
will be higher in `forward_horizon_sec` seconds.  Walk-forward cross-validation
(`TimeSeriesSplit`) prevents look-ahead bias.

`SignalGenerator` loads the persisted model and maps the probability estimate
to a directional signal:

| Condition | Signal |
|---|---|
| P(up) ≥ 0.62 | **BUY** |
| P(up) ≤ 0.38 | **SELL** |
| otherwise | **FLAT** |

### 5 · Execution & Risk Management (`bist_bot/execution/`)

`RiskManager` computes:
- **Stop-loss / take-profit** levels using ATR × configurable multipliers
- **Position size** via fixed-fractional risk (default 2 % of portfolio per trade)

`ExecutionEngine` submits a market order, measures round-trip latency, and
cancels the order if latency exceeds `max_latency_ms`.  On success it attaches
bracket orders for automatic stop/TP management.

---

## Quickstart

### Prerequisites

- Python ≥ 3.11
- Redis (local or remote)
- Matriks IQ / IdealData API key with `broker_distribution` channel access
- A broker SDK that exposes `submit_order`, `cancel_order`, `submit_bracket`

### Install

```bash
pip install -r requirements.txt
```

### Train the model

Provide a CSV with columns matching the output of `FeatureEngineer.compute_features()`
plus a `mid_price` column, then:

```python
import pandas as pd
from bist_bot.ml.trainer import SignalModelTrainer

df = pd.read_csv("historical_features.csv", index_col="timestamp", parse_dates=True)
trainer = SignalModelTrainer(forward_horizon_sec=60)
X, y = trainer.build_dataset(df.drop(columns=["mid_price"]), df["mid_price"])
trainer.train(X, y)
trainer.save("models/bofa_a1_tera_xgb.pkl")
```

### Run the bot

```bash
export MATRIKS_WS_URI="wss://ws.matriksiq.com/v2/stream"
export MATRIKS_API_KEY="<your-key>"
export BROKER_API_CLASS="mybroker.api.BrokerClient"  # omit to use NullBroker stub
export BIST_SYMBOLS="GARAN,AKBNK"
export MODEL_PATH="models/bofa_a1_tera_xgb.pkl"

python -m bist_bot.main
```

### Run tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MATRIKS_WS_URI` | *(required)* | WebSocket endpoint URL |
| `MATRIKS_API_KEY` | *(required)* | API key for Bearer auth |
| `BROKER_API_CLASS` | *(unset → NullBroker)* | Dotted import path for broker client |
| `BIST_SYMBOLS` | `GARAN` | Comma-separated tickers |
| `MODEL_PATH` | `models/bofa_a1_tera_xgb.pkl` | Trained model bundle |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `FEATURE_WINDOWS` | `60,300` | Rolling window sizes in seconds |
| `ATR_LOOKBACK` | `14` | Periods for ATR calculation |
| `PORTFOLIO_VALUE_TRY` | `1000000` | Portfolio value for position sizing |

---

## Operational Notes

| Concern | Recommendation |
|---|---|
| **Market Data** | Matriks IQ or Foreks for *Aracı Kurum Dağılımı*; confirm broker-level tick data is included in your subscription |
| **Latency** | Co-locate or use a VPS in Istanbul (Türk Telekom data centre); target < 10 ms round-trip to exchange |
| **Historical Data** | Store ticks in TimescaleDB; use `time_bucket()` to build training datasets |
| **Model Retraining** | Retrain daily after market close; always use walk-forward validation |
| **Regulatory** | BIST algorithmic trading requires SPK compliance and must be registered with your broker |
| **Feature Drift** | Monitor KL-divergence of broker flow distributions; trigger retraining when threshold is exceeded |
| **Backtest** | Use an event-driven backtester (VectorBT) with exact broker tick replay; account for T+2 settlement and ≈ 0.1 % commission |
