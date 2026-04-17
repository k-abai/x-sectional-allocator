"""
Generate synthetic CSV price data for walk-forward backtest testing.
Creates enough history (2010-2025) for MA200 to warm up.
"""
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import numpy as np

def generate_prices(ticker, days=4000, drift=0.0003, vol=0.012, seed=None):
    """Generate synthetic daily close prices with optional drift."""
    if seed is not None:
        np.random.seed(seed)
    dates = pd.bdate_range(end=pd.Timestamp('2025-12-31'), periods=days)
    returns = np.random.normal(drift, vol, size=days)
    prices = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame({'date': dates, 'close': prices})

tickers_config = {
    # CS universe — mix of up/sideways/down trends
    'QQQ':  {'drift':  0.00050, 'vol': 0.013},
    'EFA':  {'drift':  0.00015, 'vol': 0.011},
    'EEM':  {'drift':  0.00010, 'vol': 0.014},
    'XLU':  {'drift':  0.00020, 'vol': 0.009},
    'XLF':  {'drift':  0.00025, 'vol': 0.013},
    'XLI':  {'drift':  0.00025, 'vol': 0.011},
    'XLV':  {'drift':  0.00030, 'vol': 0.010},
    'XLRE': {'drift':  0.00018, 'vol': 0.012},
    'XLE':  {'drift':  0.00005, 'vol': 0.015},
    'XLK':  {'drift':  0.00055, 'vol': 0.014},
    'XLB':  {'drift':  0.00015, 'vol': 0.012},
    'XLP':  {'drift':  0.00018, 'vol': 0.008},
    'XLY':  {'drift':  0.00030, 'vol': 0.013},
    'XLC':  {'drift':  0.00025, 'vol': 0.011},
    # Defensive
    'SHY':  {'drift':  0.00005, 'vol': 0.003},
    'IEF':  {'drift':  0.00010, 'vol': 0.005},
    # Benchmark
    'SPY':  {'drift':  0.00030, 'vol': 0.012},
}

data_dir = PROJECT_ROOT / 'data'
data_dir.mkdir(exist_ok=True)

for ticker, params in tickers_config.items():
    df = generate_prices(ticker, days=4000, seed=hash(ticker) % (2**16), **params)
    path = data_dir / f'{ticker}.csv'
    df.to_csv(path, index=False)
    print(f"  Created {path}  ({len(df)} rows)")

print(f"\n[OK] Synthetic data written to {data_dir}/")
