"""
Walk-Forward Backtest — Cross-Sectional MA200 Allocation Engine

Self-contained vectorised simulation. No live broker calls.

Pipeline:
  1. Load multi-symbol price history (CSV or Alpaca)
  2. Slice into expanding walk-forward windows
  3. Each month-end: run CrossSectionalAllocator to get target weights
  4. Apply CS risk gating (breadth check → defensive)
  5. Simulate daily P&L from weight × daily return
  6. Aggregate metrics across full backtest and per-window
  7. Save equity curve, trades, metrics CSV, and performance chart

Usage:
  python scripts/run_walkforward.py                      # uses CSV data in data/
  python scripts/run_walkforward.py --source alpaca      # fetch from Alpaca
  python scripts/run_walkforward.py --start 2015-01-01  # custom start date
"""

import sys
import os
import argparse
import yaml
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')           # headless – no display required
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# ── Project root on path ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from core.cross_sectional_allocator import CrossSectionalAllocator


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def _load_csv_prices(tickers: list, data_dir: Path) -> dict:
    """Load close prices from CSV files. Returns {ticker: pd.Series}."""
    multi = {}
    for ticker in tickers:
        path = data_dir / f"{ticker}.csv"
        if not path.exists():
            print(f"  [WARN] No CSV for {ticker} at {path}")
            continue
        df = pd.read_csv(path, parse_dates=True)
        # Accept common date column names
        date_cols = [c for c in df.columns if c.lower() in ('date', 'datetime', 'timestamp', 'time')]
        if date_cols:
            df = df.set_index(date_cols[0])
            df.index = pd.to_datetime(df.index).tz_localize(None)
        close_cols = [c for c in df.columns if c.lower() in ('close', 'adj close', 'adj_close')]
        if close_cols:
            series = df[close_cols[0]].dropna().sort_index()
            multi[ticker] = series
            print(f"  [OK] {ticker}: {len(series)} bars from CSV")
        else:
            print(f"  [WARN] No 'close' column found in {path}")
    return multi


def _fetch_alpaca_prices(tickers: list, start: str, end: str) -> dict:
    """Fetch close prices via Alpaca. Returns {ticker: pd.Series}."""
    import alpaca_trade_api as tradeapi
    from core.load_keys import load_keys
    keys = load_keys()

    def _sanitize(url):
        url = url.rstrip('/')
        while url.lower().endswith('/v2'):
            url = url[:-3]
        return url

    api = tradeapi.REST(
        keys['api_key'], keys['secret_key'],
        _sanitize(keys['base_url']), api_version='v2'
    )
    feed = os.getenv("APCA_DATA_FEED", "iex")

    multi = {}
    for ticker in tickers:
        try:
            raw = api.get_bars(ticker, '1Day', start=start, end=end,
                               adjustment='all', feed=feed).df
            if raw.empty:
                print(f"  [WARN] No data from Alpaca for {ticker}")
                continue
            series = raw['close'].dropna()
            series.index = pd.to_datetime(series.index).tz_localize(None)
            series = series.sort_index()
            multi[ticker] = series
            # Also save as CSV for future runs
            os.makedirs(PROJECT_ROOT / 'data', exist_ok=True)
            raw.to_csv(PROJECT_ROOT / 'data' / f'{ticker}.csv')
            print(f"  [OK] {ticker}: {len(series)} bars from Alpaca")
        except Exception as e:
            print(f"  [WARN] Failed to fetch {ticker}: {e}")
    return multi


# ══════════════════════════════════════════════════════════════════════════════
# CS Risk Gate Helper
# ══════════════════════════════════════════════════════════════════════════════

def _apply_cs_risk(
    w_raw: dict,
    multi_history: dict,
    cs_universe: list,
    cs_risk_conf: dict,
    cs_budget: float,
) -> dict:
    """
    Mirror of the production cs_risk logic, applied to a single backtest date.
    Returns final weights (may include defensive assets).
    """
    if not cs_risk_conf.get('enabled', False):
        return _cap(w_raw, cs_budget)

    lookback = cs_risk_conf.get('lookback_days', 21)
    threshold = cs_risk_conf.get('pct_negative_threshold', 0.70)
    defensive_assets = cs_risk_conf.get('defensive_assets', ['SHY', 'IEF'])
    cash_fraction = cs_risk_conf.get('cash_fraction_of_cs', 0.50)

    neg_count = 0
    considered = 0
    for t in cs_universe:
        hist = multi_history.get(t)
        if hist is None or len(hist) <= lookback:
            continue
        r = (hist.iloc[-1] / hist.iloc[-1 - lookback]) - 1.0
        considered += 1
        if r < 0:
            neg_count += 1

    pct_neg = (neg_count / considered) if considered > 0 else 0.0
    risk_off = pct_neg >= threshold

    if risk_off:
        avail_def = [a for a in defensive_assets if a in multi_history and len(multi_history[a]) > 0]
        if not avail_def:
            return {t: 0.0 for t in cs_universe}
        def_weight = ((1.0 - cash_fraction) * cs_budget) / len(avail_def)
        weights = {t: 0.0 for t in cs_universe}
        for a in avail_def:
            weights[a] = def_weight
        return weights
    else:
        return _cap(w_raw, cs_budget)


def _cap(weights: dict, cap: float) -> dict:
    """Scale weights so sum <= cap."""
    total = sum(weights.values())
    if total <= 0:
        return {k: 0.0 for k in weights}
    if total > cap:
        scale = cap / total
        return {k: v * scale for k, v in weights.items()}
    return dict(weights)


# ══════════════════════════════════════════════════════════════════════════════
# Walk-Forward Engine
# ══════════════════════════════════════════════════════════════════════════════

def run_walkforward(config: dict, multi_prices: dict,
                    start_date: str = None, end_date: str = None):
    """
    Run the walk-forward CS backtest.

    Returns:
        equity_curve    pd.Series  (daily portfolio value, starts at initial_capital)
        trades_log      pd.DataFrame  (one row per rebalance event)
        metrics         dict
    """
    initial_capital = float(config.get('initial_capital', 100_000.0))
    cs_conf         = config.get('cs_allocator', {})
    cs_risk_conf    = config.get('cs_risk', {})
    budgets         = config.get('budgets', {})
    cs_total_max    = budgets.get('cs_total_max', 0.98)
    cash_buffer     = budgets.get('cash_buffer', 0.02)
    cs_budget       = min(cs_total_max, 1.0 - cash_buffer)

    cs_universe      = cs_conf.get('universe', [])
    defensive_assets = cs_risk_conf.get('defensive_assets', ['SHY', 'IEF'])
    all_tickers      = list(set(cs_universe + defensive_assets))

    # ── Build aligned price matrix ────────────────────────────────────────────
    available = {t: multi_prices[t] for t in all_tickers if t in multi_prices}
    if not available:
        raise ValueError("No price data available for any ticker in the universe.")

    price_df = pd.DataFrame(available).sort_index()
    price_df.index = pd.to_datetime(price_df.index)

    # Restrict to date range
    if start_date:
        price_df = price_df[price_df.index >= pd.Timestamp(start_date)]
    if end_date:
        price_df = price_df[price_df.index <= pd.Timestamp(end_date)]

    # Forward fill short gaps (max 2 days), then drop rows still missing
    price_df = price_df.ffill(limit=2)

    # Daily returns (simple)
    returns_df = price_df.pct_change().fillna(0.0)

    all_dates = price_df.index
    print(f"  Backtest range: {all_dates[0].date()} -> {all_dates[-1].date()} ({len(all_dates)} trading days)")

    # ── MA200 allocator ────────────────────────────────────────────────────────
    allocator = CrossSectionalAllocator(config)

    # ── Simulation ────────────────────────────────────────────────────────────
    equity     = initial_capital
    portfolio_weights: dict = {}          # {ticker: weight}
    equity_curve = []
    trades_log   = []
    last_rebalance_month = None

    # CS risk state persistence
    cs_risk_state  = 'RISK_ON'
    cs_risk_streak = 0
    cs_min_days    = int(cs_risk_conf.get('min_days_in_state', 3))

    for i, date in enumerate(all_dates):
        # Skip if insufficient history for MA200
        if i < cs_conf.get('ma_window', 200):
            equity_curve.append({'date': date, 'equity': equity})
            continue

        current_month = date.month

        # ── Monthly rebalance ────────────────────────────────────────────────
        if current_month != last_rebalance_month:
            last_rebalance_month = current_month

            # Build history slice (up to current row, inclusive)
            history_slice = {}
            for t in all_tickers:
                if t in price_df.columns:
                    s = price_df[t].iloc[:i+1].dropna()
                    if len(s) > 0:
                        history_slice[t] = s

            # 1. CS raw weights
            w_raw = allocator.get_target_weights(history_slice, date)

            # 2. CS risk gating
            w_final = _apply_cs_risk(
                w_raw, history_slice, cs_universe,
                cs_risk_conf, cs_budget
            )

            # 3. Persist cs_risk state
            # (simplified: recalculate pct_negative each month)
            lookback = cs_risk_conf.get('lookback_days', 21)
            neg_count = sum(
                1 for t in cs_universe
                if t in history_slice and len(history_slice[t]) > lookback
                and (history_slice[t].iloc[-1] / history_slice[t].iloc[-1-lookback] - 1.0) < 0
            )
            considered = sum(
                1 for t in cs_universe
                if t in history_slice and len(history_slice[t]) > lookback
            )
            pct_neg = (neg_count / considered) if considered > 0 else 0.0
            desired = 'RISK_OFF' if pct_neg >= cs_risk_conf.get('pct_negative_threshold', 0.70) else 'RISK_ON'
            if desired == cs_risk_state:
                cs_risk_streak += 1
            else:
                cs_risk_streak = 1
            if desired != cs_risk_state and cs_risk_streak >= cs_min_days:
                cs_risk_state = desired
                cs_risk_streak = 0

            selected = [t for t, w in w_final.items() if w > 0]
            cs_debug = allocator.last_debug

            trades_log.append({
                'date':         date,
                'selected':     str(selected),
                'weights':      str({t: round(w, 4) for t, w in w_final.items() if w > 0}),
                'eligible':     str(cs_debug.get('eligible', [])),
                'cs_mode':      cs_risk_state,
                'pct_negative': round(pct_neg, 4),
                'n_selected':   len(selected),
            })

            portfolio_weights = dict(w_final)

        # ── Daily P&L ────────────────────────────────────────────────────────
        daily_return = 0.0
        for ticker, weight in portfolio_weights.items():
            if ticker in returns_df.columns:
                ret = returns_df.iloc[i][ticker]
                if not np.isnan(ret):
                    daily_return += weight * ret

        equity *= (1.0 + daily_return)
        equity_curve.append({'date': date, 'equity': equity})

    # ── Build output DataFrames ───────────────────────────────────────────────
    equity_df = pd.DataFrame(equity_curve).set_index('date')['equity']
    trades_df = pd.DataFrame(trades_log) if trades_log else pd.DataFrame()

    return equity_df, trades_df


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def calculate_metrics(equity: pd.Series, initial_capital: float = 100_000.0) -> dict:
    """Compute standard performance metrics from an equity curve."""
    if len(equity) < 2:
        return {}

    daily_returns = equity.pct_change().dropna()
    total_return  = (equity.iloc[-1] / equity.iloc[0]) - 1.0
    n_years       = len(equity) / 252.0
    cagr          = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else 0.0
    ann_vol       = daily_returns.std() * np.sqrt(252)
    sharpe        = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0.0
    running_max   = equity.cummax()
    drawdowns     = (equity - running_max) / running_max
    max_dd        = drawdowns.min()
    calmar        = cagr / abs(max_dd) if max_dd != 0 else 0.0

    return {
        'total_return_pct':  round(total_return * 100, 2),
        'cagr_pct':          round(cagr * 100, 2),
        'ann_volatility_pct': round(ann_vol * 100, 2),
        'sharpe_ratio':      round(sharpe, 3),
        'max_drawdown_pct':  round(max_dd * 100, 2),
        'calmar_ratio':      round(calmar, 3),
        'n_trading_days':    len(equity),
        'final_equity':      round(equity.iloc[-1], 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Build-in SPY Buy-and-Hold Benchmark
# ══════════════════════════════════════════════════════════════════════════════

def build_benchmark(price_series: pd.Series, equity_index: pd.DatetimeIndex,
                    initial_capital: float) -> pd.Series:
    """Buy-and-hold on price_series, reindexed to equity_index."""
    aligned = price_series.reindex(equity_index).ffill()
    aligned = aligned.dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    bh = initial_capital * (aligned / aligned.iloc[0])
    return bh


# ══════════════════════════════════════════════════════════════════════════════
# Chart
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(equity: pd.Series, benchmark: pd.Series, output_dir: Path):
    """Save equity curve chart."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [3, 1]})
    ax1, ax2 = axes

    ax1.plot(equity.index, equity.values, label='CS Allocator', color='#2196F3', linewidth=2)
    if not benchmark.empty:
        ax1.plot(benchmark.index, benchmark.values, label='SPY B&H', color='#FF9800',
                 linewidth=1.5, linestyle='--', alpha=0.85)
    ax1.set_title('Cross-Sectional MA200 Allocation Engine — Walk-Forward Backtest',
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    # Drawdown
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max * 100
    ax2.fill_between(dd.index, dd.values, 0, color='#F44336', alpha=0.6, label='Drawdown %')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = output_dir / 'cs_walkforward_chart.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved: {chart_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='CS Walk-Forward Backtest')
    parser.add_argument('--source', choices=['csv', 'alpaca'], default='csv',
                        help='Data source (default: csv)')
    parser.add_argument('--start',  default='2015-01-01', help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end',    default=datetime.now().strftime('%Y-%m-%d'),
                        help='Backtest end date (YYYY-MM-DD)')
    parser.add_argument('--benchmark', default='SPY', help='Benchmark ticker (default: SPY)')
    parser.add_argument('--no-chart', action='store_true', help='Skip chart generation')
    args = parser.parse_args()

    # ── Load Config ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  CS MA200 WALK-FORWARD BACKTEST")
    print("=" * 60)

    config_path = PROJECT_ROOT / 'config' / 'strategy.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    initial_capital = float(config.get('initial_capital', 100_000.0))
    cs_universe     = config.get('cs_allocator', {}).get('universe', [])
    defensive_assets = config.get('cs_risk', {}).get('defensive_assets', ['SHY', 'IEF'])
    benchmark_ticker = args.benchmark

    all_tickers = list(set(cs_universe + defensive_assets + [benchmark_ticker]))

    print(f"\n  Config:      {config_path}")
    print(f"  Universe:    {cs_universe}")
    print(f"  Defensive:   {defensive_assets}")
    print(f"  Date range:  {args.start} -> {args.end}")
    print(f"  Data source: {args.source}")
    print(f"  Capital:     ${initial_capital:,.0f}")

    # ── Load Data ─────────────────────────────────────────────────────────────
    print("\n[1] Loading price data...")
    data_dir = PROJECT_ROOT / 'data'

    if args.source == 'alpaca':
        multi_prices = _fetch_alpaca_prices(all_tickers, args.start, args.end)
    else:
        multi_prices = _load_csv_prices(all_tickers, data_dir)
        # If any tickers missing from CSV, warn — don't auto-fetch (no keys needed)
        missing = [t for t in cs_universe if t not in multi_prices]
        if missing:
            print(f"  [WARN] Missing CSV data for: {missing}")
            print(f"  Tip: run with --source alpaca to auto-fetch, or place CSV files in data/")

    if not multi_prices:
        print("  [ERROR] No price data loaded. Exiting.")
        return

    # ── Run Backtest ──────────────────────────────────────────────────────────
    print("\n[2] Running walk-forward simulation...")
    try:
        equity, trades = run_walkforward(
            config, multi_prices,
            start_date=args.start,
            end_date=args.end,
        )
    except Exception as e:
        print(f"  [ERROR] Backtest failed: {e}")
        raise

    if equity.empty:
        print("  [ERROR] Equity curve is empty — check data coverage.")
        return

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("\n[3] Computing metrics...")
    metrics = calculate_metrics(equity, initial_capital)

    # Benchmark metrics
    bm_series = multi_prices.get(benchmark_ticker)
    benchmark_equity = pd.Series(dtype=float)
    if bm_series is not None:
        benchmark_equity = build_benchmark(bm_series, equity.index, initial_capital)
        bm_metrics = calculate_metrics(benchmark_equity, initial_capital)
    else:
        bm_metrics = {}

    # ── Print Results ─────────────────────────────────────────────────────────
    print(f"\n{'Metric':<28} {'CS Allocator':>14} {benchmark_ticker + ' B&H':>14}")
    print("-" * 58)
    metric_keys = ['total_return_pct', 'cagr_pct', 'ann_volatility_pct',
                   'sharpe_ratio', 'max_drawdown_pct', 'calmar_ratio']
    labels = ['Total Return (%)', 'CAGR (%)', 'Ann. Volatility (%)',
              'Sharpe Ratio', 'Max Drawdown (%)', 'Calmar Ratio']
    for key, label in zip(metric_keys, labels):
        cs_val = metrics.get(key, float('nan'))
        bm_val = bm_metrics.get(key, float('nan'))
        print(f"  {label:<26} {cs_val:>14.3f} {bm_val:>14.3f}")

    print(f"\n  Final equity:    ${metrics['final_equity']:>14,.2f}")
    print(f"  Trading days:    {metrics['n_trading_days']:>14,}")
    print(f"  Rebalances:      {len(trades):>14,}")

    # ── Save Outputs ──────────────────────────────────────────────────────────
    output_dir = PROJECT_ROOT / 'output'
    output_dir.mkdir(exist_ok=True)

    equity.to_csv(output_dir / 'cs_walkforward_equity.csv', header=['equity'])
    print(f"\n[4] Saving outputs to {output_dir}/")
    print(f"  > Equity curve  -> cs_walkforward_equity.csv  ({len(equity)} rows)")

    if not trades.empty:
        trades.to_csv(output_dir / 'cs_walkforward_trades.csv', index=False)
        print(f"  > Rebalance log -> cs_walkforward_trades.csv  ({len(trades)} rows)")

    metrics_out = {'cs_allocator': metrics}
    if bm_metrics:
        metrics_out[benchmark_ticker] = bm_metrics
    pd.DataFrame(metrics_out).T.to_csv(output_dir / 'cs_walkforward_metrics.csv')
    print(f"  > Metrics       -> cs_walkforward_metrics.csv")

    if not args.no_chart:
        plot_results(equity, benchmark_equity, output_dir)

    print("\n[OK] Walk-forward backtest complete.")

    # Display recent rebalance table
    if not trades.empty:
        print("\n  Last 5 rebalances:")
        display_cols = ['date', 'n_selected', 'selected', 'cs_mode', 'pct_negative']
        available_cols = [c for c in display_cols if c in trades.columns]
        tail = trades[available_cols].tail(5).to_string(index=False)
        for line in tail.split('\n'):
            print(f"    {line}")


if __name__ == "__main__":
    main()
