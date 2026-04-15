"""
Cross-Sectional Allocation Engine — Single Production Cycle

1. Load configs
2. Fetch OHLCV history for CS universe + defensive assets
3. Run CS-only StrategyOrchestrator → intents
4. Validate exposure invariants
5. Execute netted rebalance (or dry-run)
"""

import sys
import os
import json
import yaml
import pandas as pd
from datetime import datetime as dt_datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Load environment
load_dotenv(PROJECT_ROOT / ".env")

from core.strategy import StrategyOrchestrator
from core.execution import ExecutionEngine
from core.fetch_bars import fetch_bar_alpaca
from core.logger import SystemLogger


def run_production():
    logger = SystemLogger()
    logger.log_event("PRODUCTION_RUN_START", {})
    print("\n" + "=" * 50)
    print("  CROSS-SECTIONAL ALLOCATION — PRODUCTION CYCLE")
    print("=" * 50)

    # 1. Load Configurations
    try:
        with open(PROJECT_ROOT / 'config' / 'strategy.yaml', 'r') as f:
            strat_config = yaml.safe_load(f)
        with open(PROJECT_ROOT / 'config' / 'system.yaml', 'r') as f:
            sys_config = yaml.safe_load(f)
    except Exception as e:
        logger.log_error(f"Config load failed: {e}")
        return

    # 2. Init Components
    orchestrator = StrategyOrchestrator(strat_config)
    engine = ExecutionEngine(sys_config)
    # Attach strategy config so execution can read DRY_RUN_ALLOC
    try:
        engine.config['strategy'] = strat_config
    except Exception:
        engine.config = engine.config if isinstance(engine.config, dict) else {}
        engine.config['strategy'] = strat_config

    # 3. Define Universe
    cs_universe = strat_config.get('cs_allocator', {}).get('universe', [])
    full_universe = list(set(cs_universe))

    # Include defensive assets if CS risk is enabled
    cs_risk_conf = strat_config.get('cs_risk', {})
    if cs_risk_conf.get('enabled', False):
        defensive_assets = cs_risk_conf.get('defensive_assets', ["SHY", "IEF"])
        for a in defensive_assets:
            if a not in full_universe:
                full_universe.append(a)

    # 4. Fetch History
    print(f"[INFO] Fetching 750 days of history for {len(full_universe)} symbols...")
    from core.load_keys import load_keys
    keys = load_keys()

    try:
        end_dt = dt_datetime.now()
        start_dt = end_dt - timedelta(days=750)

        bars, meta = fetch_bar_alpaca(
            full_universe,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            api_key=keys["api_key"],
            secret_key=keys["secret_key"],
            base_url=keys["base_url"],
            plan_feed=os.getenv("APCA_DATA_FEED", "iex"),
            wide=True
        )

        multi_history = {}
        for s in full_universe:
            if s in bars['close'].columns:
                multi_history[s] = bars['close'][s].dropna()
            else:
                print(f"[WARNING] Missing data for {s}")

    except Exception as e:
        logger.log_error(f"Data fetch failed: {e}")
        return

    # 5. Get Current State
    print("[INFO] Fetching current account state...")
    last_trade_timestamps = {}
    runtime_state = {}

    try:
        account = engine.broker.get_account()
        equity = account['equity']
        positions = engine.broker.get_positions()
        pos_dict = {p['symbol']: p['qty'] for p in positions}
    except Exception as e:
        logger.log_error(f"Account fetch failed: {e}")
        return

    # Load runtime state
    runtime_state_file = PROJECT_ROOT / 'runtime_state.json'
    try:
        if runtime_state_file.exists():
            with open(runtime_state_file, 'r') as f:
                runtime_state = json.load(f) or {}
                last_trade_timestamps = runtime_state.get('last_trade_timestamps', {})
        else:
            runtime_state = {}
    except Exception as e:
        logger.log_error(f"State load failed (non-blocking): {e}")

    # 6. Generate Strategy Intents
    print("[INFO] Generating CS strategy intents...")
    today = pd.Timestamp.now()
    try:
        intents = orchestrator.generate_portfolio_intents(
            multi_history=multi_history,
            current_equity=equity,
            current_positions=pos_dict,
            current_date=today,
            runtime_state=runtime_state,
        )
    except Exception as e:
        logger.log_error(f"Intent generation failed: {e}")
        return

    # 7. Validate Exposure Invariants
    print("[INFO] Validating exposure invariants...")
    try:
        engine.validate_exposure_invariants(intents, strat_config)
    except Exception as e:
        logger.log_error(f"Exposure validation failed: {e}")
        return

    # 8. Execute Netted Rebalance
    if engine.is_dry_run():
        print("[INFO] DRY RUN enabled: intentions logged, no orders submitted.")
    print(f"[INFO] Executing netted rebalance for {len(intents)} intents...")

    # Print intent summary
    for intent in intents:
        if intent['weight'] > 0:
            print(f"  {intent['symbol']:6s}  w={intent['weight']:.4f}  reason={intent['reason']}")

    try:
        engine.execute_netted_rebalance(intents, last_trade_timestamps)
    except Exception as e:
        logger.log_error(f"Netted rebalance failed: {e}")

    # Save runtime state
    runtime_state['last_trade_timestamps'] = last_trade_timestamps
    with open(runtime_state_file, 'w') as f:
        json.dump(runtime_state, f, indent=2)

    print("\n[OK] Production cycle complete.")
    logger.log_event("PRODUCTION_RUN_COMPLETE", {"equity": equity})


if __name__ == "__main__":
    run_production()
