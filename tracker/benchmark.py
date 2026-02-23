"""
tracker/benchmark.py
=====================
Computes portfolio performance and compares it against market indices.

Key concepts introduced here:
  - Time-weighted returns  : the standard way to measure portfolio performance
                             regardless of when cash was added/withdrawn
  - Reindexing             : aligning two DataFrames to the same dates
  - Drawdown               : how far a portfolio has fallen from its peak
  - Sharpe ratio           : return per unit of risk (higher = better)
  - pd.date_range          : generating sequences of dates
"""

from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from tracker.portfolio import Portfolio, Holding, Transaction


# ── Available benchmark indices ───────────────────────────────────────────────

INDICES = {
    "MSCI World":          "URTH",
    "S&P 500":             "SPY",
    "NASDAQ 100":          "QQQ",
    "MSCI Emerging Mkts":  "EEM",
}


# ── Portfolio return series ───────────────────────────────────────────────────

def build_portfolio_value_series(
    portfolio: Portfolio,
    start_date: date,
) -> Optional[pd.Series]:
    """
    Build a daily portfolio value series from the start_date to today.

    How it works:
      1. Collect every ticker ever held
      2. Download daily price history for all of them
      3. For each day, work out how many shares are held and multiply by price
      4. Sum across all holdings to get total portfolio value

    This approach gives us true time-weighted performance — if you bought
    more shares halfway through, the chart reflects that correctly.

    Returns a pd.Series with dates as index and portfolio value (€) as values.
    """
    holdings = portfolio.all_holdings()
    if not holdings:
        return None

    # All tickers in the portfolio (including zero-quantity ones for history)
    all_tickers = list(portfolio.holdings.keys())
    start_str   = start_date.strftime("%Y-%m-%d")

    # Download daily prices for all tickers from start_date to today
    try:
        raw = yf.download(
            all_tickers,
            start=start_str,
            progress=False,
            auto_adjust=True,
        )
    except Exception:
        return None

    if raw.empty:
        return None

    # Extract close prices — structure differs for 1 vs multiple tickers
    if len(all_tickers) == 1:
        prices_df = raw[["Close"]].copy()
        prices_df.columns = [all_tickers[0]]
    else:
        prices_df = raw["Close"].copy()
        prices_df.columns = [c.upper() for c in prices_df.columns]

    prices_df.index = pd.to_datetime(prices_df.index).tz_localize(None)

    # Build a transaction log sorted by date
    all_transactions: List[Tuple[date, str, str, float, float]] = []
    for ticker, holding in portfolio.holdings.items():
        for t in holding.transactions:
            all_transactions.append((
                datetime.strptime(t.date, "%Y-%m-%d").date(),
                ticker,
                t.action,
                t.quantity,
                t.price,
            ))
    all_transactions.sort(key=lambda x: x[0])

    # For each trading day, compute cumulative holdings up to that day
    # then multiply by price to get portfolio value
    dates      = prices_df.index
    values     = []
    # Running position: ticker -> quantity held
    positions: Dict[str, float] = {t: 0.0 for t in all_tickers}

    txn_idx = 0
    n_txns  = len(all_transactions)

    for dt in dates:
        dt_date = dt.date()

        # Apply all transactions up to and including this date
        while txn_idx < n_txns and all_transactions[txn_idx][0] <= dt_date:
            _, ticker, action, qty, _ = all_transactions[txn_idx]
            if action == "buy":
                positions[ticker] = positions.get(ticker, 0) + qty
            elif action == "sell":
                positions[ticker] = positions.get(ticker, 0) - qty
            txn_idx += 1

        # Sum: quantity * price for each ticker
        total = 0.0
        for ticker, qty in positions.items():
            if qty <= 0:
                continue
            ticker_upper = ticker.upper()
            if ticker_upper in prices_df.columns:
                price = prices_df.loc[dt, ticker_upper]
                if pd.notna(price):
                    total += qty * float(price)

        values.append(total)

    series = pd.Series(values, index=dates, name="Portfolio")

    # Trim leading zeros (before any transactions)
    first_nonzero = series[series > 0].index
    if len(first_nonzero) == 0:
        return None
    series = series[first_nonzero[0]:]

    return series


# ── Index data ────────────────────────────────────────────────────────────────

def fetch_index_series(
    ticker: str,
    start_date: date,
) -> Optional[pd.Series]:
    """Download daily close prices for a benchmark index from start_date."""
    try:
        raw = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
        if raw.empty:
            return None
        series = raw["Close"].squeeze()
        series.index = pd.to_datetime(series.index).tz_localize(None)
        series.name  = ticker
        return series
    except Exception:
        return None


# ── Normalisation (rebase to €100) ───────────────────────────────────────────

def normalise(series: pd.Series) -> pd.Series:
    """
    Rebase a price/value series so it starts at 100.
    This lets us compare portfolio and indices on the same scale —
    'growth of €100 invested at the start'.
    """
    first = series.dropna().iloc[0]
    if first == 0:
        return series
    return (series / first) * 100


# ── Drawdown ──────────────────────────────────────────────────────────────────

def compute_drawdown(series: pd.Series) -> pd.Series:
    """
    Compute the drawdown series: how far (in %) below the rolling peak.
    A value of -0.20 means the series is 20% below its highest point so far.

    Formula: drawdown = (current - rolling_max) / rolling_max
    """
    rolling_max = series.cummax()      # Highest value seen so far at each point
    drawdown    = (series - rolling_max) / rolling_max * 100
    return drawdown


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_stats(series: pd.Series, label: str) -> dict:
    """
    Compute key performance statistics for a price/value series.

    Concepts:
      - Total return     : (end / start) - 1
      - Ann. return      : total return compounded over years
      - Volatility       : std of daily returns, annualised by sqrt(252)
      - Sharpe ratio     : (ann. return - risk free rate) / ann. volatility
                           We use 0% as the risk-free rate for simplicity.
      - Max drawdown     : worst peak-to-trough decline over the period
    """
    series   = series.dropna()
    if len(series) < 2:
        return {}

    daily_returns = series.pct_change().dropna()
    n_days        = len(series)
    n_years       = n_days / 252

    total_return  = (series.iloc[-1] / series.iloc[0]) - 1
    ann_return    = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol       = daily_returns.std() * np.sqrt(252)
    sharpe        = ann_return / ann_vol if ann_vol > 0 else 0
    max_dd        = compute_drawdown(series).min()
    best_day      = daily_returns.max() * 100
    worst_day     = daily_returns.min() * 100

    return {
        "Label":          label,
        "Total Return":   f"{total_return:+.2%}",
        "Ann. Return":    f"{ann_return:+.2%}",
        "Ann. Volatility":f"{ann_vol:.2%}",
        "Sharpe Ratio":   f"{sharpe:.2f}",
        "Max Drawdown":   f"{max_dd:.2f}%",
        "Best Day":       f"{best_day:+.2f}%",
        "Worst Day":      f"{worst_day:+.2f}%",
        "Days":           str(n_days),
        # Raw values for colouring
        "_total_return":  total_return,
        "_sharpe":        sharpe,
        "_max_dd":        max_dd,
    }


# ── Oldest transaction date ───────────────────────────────────────────────────

def get_portfolio_start_date(portfolio: Portfolio) -> Optional[date]:
    """Find the date of the earliest transaction across all holdings."""
    dates = []
    for holding in portfolio.holdings.values():
        for t in holding.transactions:
            dates.append(datetime.strptime(t.date, "%Y-%m-%d").date())
    if not dates:
        return None
    return min(dates)
