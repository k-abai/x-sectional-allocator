# Cross-Sectional MA200 Allocation Engine

A lean, standalone cross-sectional allocation engine that selects the top-K performing ETFs from a broad universe based on proximity to their 200-day moving average. Monthly rebalancing with risk gating.

## Strategy Overview

1. **Cross-Sectional MA200 Selection** — Compute `score = Price / MA200 - 1` for each asset. Filter eligible (score > 0), rank descending, select top-K.
2. **CS Risk Gating** — If ≥70% of universe has negative 21-day returns, switch to defensive assets (SHY, IEF) + cash.
3. **Execution** — Netted rebalance via Alpaca (paper or live).

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
copy .env.example .env
# Edit .env with your Alpaca paper keys

# 4. Run (dry-run mode by default)
python scripts/run_production.py

# 5. Continuous loop
python scripts/run_loop.py
```

## Project Structure

```
cross_sectional/
├── config/
│   ├── strategy.yaml    # CS allocator + risk config
│   └── system.yaml      # Execution, safety, notifications
├── core/
│   ├── alpaca_broker.py  # Alpaca API wrapper
│   ├── broker_interface.py  # Abstract broker
│   ├── cross_sectional_allocator.py  # MA200 selection engine
│   ├── execution.py      # Order execution + guardrails
│   ├── fetch_bars.py     # OHLCV data fetching
│   ├── load_keys.py      # .env loader
│   ├── logger.py         # Structured JSON logging
│   ├── notifier.py       # Email notifications
│   ├── safety.py         # Kill switch + guards
│   └── strategy.py       # CS-only orchestrator
├── scripts/
│   ├── run_production.py # Single cycle
│   ├── run_loop.py       # Continuous service
│   └── verify_cs_allocator.py  # Smoke test
├── .env.example
├── requirements.txt
└── README.md
```

## Configuration

Edit `config/strategy.yaml` to adjust:
- **Universe**: ETFs to screen (`cs_allocator.universe`)
- **Top-K**: Number of assets to hold (`cs_allocator.top_k`)
- **Allocation Mode**: `equal` or `proportional`
- **Risk Gating**: Breadth threshold, defensive assets, cash fraction
- **DRY_RUN_ALLOC**: Set `true` to log intents without executing
