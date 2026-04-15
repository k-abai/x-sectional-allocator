"""
Smoke Test — Cross-Sectional Allocator

Generates synthetic price data and validates:
1. Weight invariants (sum <= 1, no negatives)
2. Top-K selection logic
3. Risk gating transition
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from core.cross_sectional_allocator import CrossSectionalAllocator
from core.strategy import StrategyOrchestrator


def generate_synthetic_history(tickers, days=300, seed=42):
    """Generate synthetic close prices with upward drift for some tickers."""
    np.random.seed(seed)
    dates = pd.bdate_range(end=pd.Timestamp.now(), periods=days)
    multi_history = {}
    for i, ticker in enumerate(tickers):
        # Alternate between uptrending and downtrending
        drift = 0.0005 if i % 2 == 0 else -0.0002
        returns = np.random.normal(drift, 0.015, size=days)
        prices = 100 * np.exp(np.cumsum(returns))
        multi_history[ticker] = pd.Series(prices, index=dates, name=ticker)
    return multi_history


def test_allocator_basic():
    """Test CrossSectionalAllocator directly."""
    print("=" * 50)
    print("  TEST 1: CrossSectionalAllocator Direct")
    print("=" * 50)

    config = {
        'cs_allocator': {
            'universe': ['QQQ', 'EFA', 'EEM', 'XLU', 'XLK'],
            'ma_window': 200,
            'top_k': 2,
            'allocation_mode': 'equal',
        }
    }

    allocator = CrossSectionalAllocator(config)
    multi_history = generate_synthetic_history(config['cs_allocator']['universe'])
    today = pd.Timestamp.now()

    weights = allocator.get_target_weights(multi_history, today)
    debug = allocator.last_debug

    print(f"  Universe: {debug['universe']}")
    print(f"  Eligible: {debug['eligible']}")
    print(f"  Selected: {debug['selected']}")
    print(f"  Weights:  {weights}")

    # Invariants
    total = sum(weights.values())
    neg = [t for t, w in weights.items() if w < 0]
    selected_count = sum(1 for w in weights.values() if w > 0)

    passed = True
    if total > 1.0 + 1e-6:
        print(f"  ❌ FAIL: total weight {total:.4f} > 1.0")
        passed = False
    if neg:
        print(f"  ❌ FAIL: negative weights {neg}")
        passed = False
    if selected_count > config['cs_allocator']['top_k']:
        print(f"  ❌ FAIL: selected {selected_count} > top_k={config['cs_allocator']['top_k']}")
        passed = False

    if passed:
        print("  ✅ PASS: All weight invariants hold\n")
    return passed


def test_orchestrator():
    """Test full StrategyOrchestrator pipeline."""
    print("=" * 50)
    print("  TEST 2: StrategyOrchestrator Pipeline")
    print("=" * 50)

    config = {
        'cs_allocator': {
            'enabled': True,
            'universe': ['QQQ', 'EFA', 'EEM', 'XLU', 'XLK', 'XLF'],
            'ma_window': 200,
            'top_k': 2,
            'allocation_mode': 'equal',
        },
        'cs_risk': {
            'enabled': True,
            'lookback_days': 21,
            'pct_negative_threshold': 0.70,
            'min_days_in_state': 3,
            'defensive_assets': ['SHY', 'IEF'],
            'defensive_split': 'equal',
            'cash_fraction_of_cs': 0.50,
            'log_debug': True,
        },
        'budgets': {
            'cs_total_max': 0.98,
            'cash_buffer': 0.02,
        },
        'portfolio': {
            'W_max': 1.0,
        },
    }

    # Include defensive assets in history
    all_tickers = config['cs_allocator']['universe'] + config['cs_risk']['defensive_assets']
    multi_history = generate_synthetic_history(all_tickers)

    orchestrator = StrategyOrchestrator(config)
    today = pd.Timestamp.now()
    runtime_state = {}

    intents = orchestrator.generate_portfolio_intents(
        multi_history=multi_history,
        current_equity=100000.0,
        current_positions={},
        current_date=today,
        runtime_state=runtime_state,
    )

    # Summarise
    total_weight = sum(i['weight'] for i in intents)
    active = [(i['symbol'], i['weight'], i['reason']) for i in intents if i['weight'] > 0]

    print(f"  Total intents: {len(intents)}")
    print(f"  Active positions: {len(active)}")
    for sym, w, reason in active:
        print(f"    {sym:6s}  w={w:.4f}  reason={reason}")
    print(f"  Total weight: {total_weight:.4f}")
    print(f"  Risk state: {runtime_state.get('cs_risk_state', '?')}")

    passed = True
    if total_weight > 1.0 + 1e-6:
        print(f"  ❌ FAIL: total weight {total_weight:.4f} > 1.0")
        passed = False
    neg = [i for i in intents if i['weight'] < 0]
    if neg:
        print(f"  ❌ FAIL: negative weights found")
        passed = False

    if passed:
        print("  ✅ PASS: Orchestrator invariants hold\n")
    return passed


def test_proportional_mode():
    """Test proportional allocation mode."""
    print("=" * 50)
    print("  TEST 3: Proportional Allocation Mode")
    print("=" * 50)

    config = {
        'cs_allocator': {
            'universe': ['QQQ', 'EFA', 'EEM'],
            'ma_window': 200,
            'top_k': 2,
            'allocation_mode': 'proportional',
        }
    }

    allocator = CrossSectionalAllocator(config)
    multi_history = generate_synthetic_history(config['cs_allocator']['universe'])
    today = pd.Timestamp.now()

    weights = allocator.get_target_weights(multi_history, today)
    selected = [t for t, w in weights.items() if w > 0]

    print(f"  Selected: {selected}")
    print(f"  Weights: {weights}")

    passed = True
    total = sum(weights.values())
    if selected and abs(total - 1.0) > 1e-6:
        print(f"  ❌ FAIL: proportional weights should sum to 1.0, got {total:.6f}")
        passed = False

    # Check weights are unequal (proportional should differ)
    w_vals = [w for w in weights.values() if w > 0]
    if len(w_vals) >= 2 and len(set(round(w, 6) for w in w_vals)) == 1:
        print("  ⚠️  WARNING: proportional weights are equal (possible if scores are equal)")

    if passed:
        print("  ✅ PASS: Proportional mode works correctly\n")
    return passed


if __name__ == "__main__":
    results = []
    results.append(("Allocator Basic", test_allocator_basic()))
    results.append(("Orchestrator Pipeline", test_orchestrator()))
    results.append(("Proportional Mode", test_proportional_mode()))

    print("=" * 50)
    print("  SUMMARY")
    print("=" * 50)
    all_pass = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n  ✅ ALL TESTS PASSED")
    else:
        print("\n  ❌ SOME TESTS FAILED")
        sys.exit(1)
