import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from datetime import datetime, timedelta
import time

import alpaca_trade_api as tradeapi
import logging
from alpaca_trade_api.rest import REST


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def nan_lengths(s):
    """
    For each index where s is NaN, return the length of that NaN run; else 0.
    Used for gap guardrails (drop long missing runs).
    """
    isna = s.isna()
    nan_len = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(nan_len).transform("size")
    return run_len.where(isna, 0)

"""
This module fetches OHLCV bars from Alpaca for multiple symbols and returns a tidy DataFrame of log scaled returns.
"""
def fetch_bar_alpaca(symbols, start=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
                     end=datetime.now().strftime("%Y-%m-%d"),
                     timeframe="1Day", wide=False, 
                     api_key="YOUR_KEY",
                     secret_key="YOUR_SECRET",
                     base_url="https://paper-api.alpaca.markets",
                     plan_feed=None,
                     enforce_overlap=True

):
    """
    Fetch OHLCV bars for multiple symbols and return a tidy DataFrame.
    timeframe: '1Min','5Min','15Min','1Hour','1Day'
    Returns columns: 'symbol', 'datetime', 'open', 'high', 'low', 'close', 'volume', and, if available, 'trade_count', 'vwap'.
    """
    freq = "B"                  # business-day grid
    ffill_limit = 1             # allow short carry for brief gaps
    max_gap = 2                 # drop rows that lie within longer NA runs (>max_gap)
    min_overlap = 252           # require ~1Y of aligned obs after cleaning
    winsor_limits = [0.01,0.01] # winsorize extreme returns at 1% level
    
    # Normalize base_url
    def _sanitize(url: str) -> str:
        if not url:
            return url
        cleaned = url.rstrip('/')
        while cleaned.lower().endswith('/v2'):
            cleaned = cleaned[:-3]
        return cleaned
    base_url = _sanitize(base_url)

    # Create a client using the Alpaca API
    api = tradeapi.REST(api_key, secret_key, base_url, api_version="v2")

    frames = []
    dropped_rows_long_gap_total = 0
    for symbol in symbols:
        max_retries = 5
        base_delay = 1.0
        df_raw = None
        for attempt in range(max_retries):
            try:
                df_raw = api.get_bars(
                    symbol = symbol,
                    timeframe=timeframe,
                    start=pd.to_datetime(start).isoformat() if not isinstance(start, str) else start,
                    end=pd.to_datetime(end).isoformat() if not isinstance(end, str) else end,
                    adjustment="all",
                    feed=plan_feed
                )
                break
            except Exception as e:
                err_str = str(e).lower()
                if '429' in err_str or 'rate limit' in err_str or 'too many requests' in err_str:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Rate limit hit for {symbol}, retrying in {delay}s...")
                    time.sleep(delay)
                    if attempt == max_retries - 1:
                        raise e
                else:
                    raise e
                    
        df = df_raw.df if df_raw is not None else pd.DataFrame()
        if df.empty:
            logger.warning(f"No data for {symbol}. Check symbols/date range/plan.")
        
        expected_cols = ["symbol", "open", "high", "low", "close", "volume", "trade_count", "vwap"]
        for col in expected_cols:
            if col not in df.columns:
                logger.warning(f"Column {col} missing for {symbol}. Column is dropped.")

        if "symbol" not in df.columns:
            df["symbol"] = symbol
        # Resample to business-day frequency
        df = df.resample(freq).last() 
        # Restrict cleaning to numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            logger.warning(f"No numeric columns found for {symbol}; skipping numeric cleaning steps.")
            df_ff = df
            rows_with_long_gap = pd.Series(False, index=df.index)
        else:
            long_gap_masks = pd.DataFrame({col: (nan_lengths(df[col]) > max_gap) for col in numeric_cols})
            rows_with_long_gap = long_gap_masks.any(axis=1)
            df_ff = df.ffill(limit=ffill_limit)
            df_ff = df_ff[~rows_with_long_gap].dropna(subset=list(numeric_cols), how="any")
            dropped_rows_long_gap_total += int(rows_with_long_gap.sum())
        # Check for ample overlap
        if len(df_ff) < min_overlap:
            logger.warning(
            f"Not enough overlap after cleaning: {len(df_ff)} rows (< {min_overlap}). "
            f"Consider widening the date range or relaxing ffill_limit/max_gap."
        )
        if not isinstance(df_ff.index, pd.DatetimeIndex):
            df_ff.index = pd.to_datetime(df_ff.index)
        frames.append(df_ff)
        if not frames:
            print("No data fetched for any symbol.")
            return pd.DataFrame()
        continue    
    
    # Concatenate all frames
    bars = pd.concat(frames)
    bars = bars.reset_index()
    cols = ["symbol", "timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"]
    available_cols = [c for c in cols if c in bars.columns]
    df = bars[available_cols]
    df = df.rename(columns={"timestamp": "datetime"})

    # Compute log returns
    df_values = df.select_dtypes(include=[np.number])
    df_labels = df.select_dtypes(exclude=[np.number])
    log_prices = np.log(df_values)
    df_log_returns = log_prices.diff().fillna(0.0)

    def _winsorize(s, limits=(0.01, 0.01)):
        w = winsorize(s.values, limits=limits)
        return pd.Series(np.asarray(w, dtype=float), index=s.index)

    df_logret_w = df_log_returns.apply(_winsorize, axis=0)
    df_logret_w = df_values.apply(lambda x: winsorize(x, limits=winsor_limits), axis=0) 

    # Final combined output (long)
    df_w = df_labels.join(df_logret_w, how="inner", rsuffix="_ret")
    
    if wide is True:
        df_w = df_w.pivot(index="datetime", columns="symbol", values=["open", "close"]).sort_index()

    meta = {
        "columns": list(df_w.columns),
        "rows_final": int(len(df_w)),
        "start_final": df_w.index.min(),
        "end_final": df_w.index.max(),
        "wide": wide,
        "ffill_limit": ffill_limit,
        "max_gap": max_gap,
        "winsor_limits": winsor_limits,
        "insufficient_data_cols": list(df_w.columns[df_w.isna().any()]),
        "dropped_rows_long_gap": int(dropped_rows_long_gap_total),
    }
    return df_w, meta
