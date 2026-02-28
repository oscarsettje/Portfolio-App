"""
tracker/benchmark.py  —  Portfolio performance vs market indices
"""

from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

from tracker.portfolio import Portfolio
from tracker.prices import _close_from_download

INDICES = {
    "MSCI World":         "URTH",
    "S&P 500":            "SPY",
    "NASDAQ 100":         "QQQ",
    "MSCI Emerging Mkts": "EEM",
}


def _safe_tz(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Remove timezone info if present — yfinance is inconsistent about this."""
    return index.tz_localize(None) if index.tz is not None else index


def _download_close(tickers: list, start: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Download daily closes from start date.
    Returns (DataFrame, None) on success or (None, error_message) on failure.
    Silently drops tickers that returned no data rather than failing entirely.
    """
    try:
        raw = yf.download(tickers, start=start, progress=False, auto_adjust=True)
        if raw.empty:
            return None, ("Yahoo Finance returned no data. "
                          "You may be rate-limited — wait ~60 min and try again.")
        close = _close_from_download(raw, tickers)
        close.index = _safe_tz(pd.to_datetime(close.index))
        close = close.dropna(axis=1, how="all")
        if close.empty:
            return None, "All tickers returned empty data after processing."
        return close, None
    except KeyError as e:
        return None, (f"Data processing error (not a rate limit): {e}. "
                      f"Try upgrading yfinance: pip install yfinance --upgrade")
    except Exception as e:
        msg = str(e)
        if "rate" in msg.lower() or "too many" in msg.lower() or "429" in msg:
            return None, "Rate-limited by Yahoo Finance — wait ~60 min and try again."
        return None, f"Download error: {msg}"


def build_portfolio_value_series(
        portfolio: Portfolio,
        start_date: date) -> Tuple[Optional[pd.Series], Optional[str]]:
    """
    Returns (series, None) on success or (None, error_message) on failure.
    Partial data is used — tickers with no price history are skipped with a warning.
    """
    if not portfolio.all_holdings():
        return None, "No holdings in portfolio."

    all_tickers = list(portfolio.holdings.keys())
    close, err  = _download_close(all_tickers, start_date.strftime("%Y-%m-%d"))
    if close is None:
        return None, err

    # Warn about tickers we couldn't get data for (but continue)
    missing = [t for t in all_tickers if t.upper() not in close.columns]

    txns: List[Tuple[date, str, str, float]] = sorted(
        [(datetime.strptime(t.date, "%Y-%m-%d").date(), ticker, t.action, t.quantity)
         for ticker, holding in portfolio.holdings.items()
         for t in holding.transactions],
        key=lambda x: x[0]
    )

    positions: Dict[str, float] = {t.upper(): 0.0 for t in all_tickers}
    txn_idx, n_txns = 0, len(txns)
    values = []

    for dt in close.index:
        dt_date = dt.date()
        while txn_idx < n_txns and txns[txn_idx][0] <= dt_date:
            _, ticker, action, qty = txns[txn_idx]
            key = ticker.upper()
            positions[key] = positions.get(key, 0) + (qty if action == "buy" else -qty)
            txn_idx += 1

        total = sum(
            qty * float(close.loc[dt, ticker])
            for ticker, qty in positions.items()
            if qty > 0 and ticker in close.columns and pd.notna(close.loc[dt, ticker])
        )
        values.append(total)

    series  = pd.Series(values, index=close.index, name="Portfolio")
    nonzero = series[series > 0].index
    if not len(nonzero):
        return None, "Portfolio value series is all zeros — check your transaction prices."

    warn = f"No historical data for: {', '.join(missing)}" if missing else None
    return series[nonzero[0]:], warn


def fetch_index_series(ticker: str, start_date: date) -> Tuple[Optional[pd.Series], Optional[str]]:
    close, err = _download_close([ticker], start_date.strftime("%Y-%m-%d"))
    if close is None:
        return None, err
    col = ticker.upper()
    if col not in close.columns:
        col = close.columns[0] if len(close.columns) else None
    if col is None:
        return None, f"No data returned for {ticker}."
    s = close[col].dropna()
    s.name = ticker
    return (s, None) if not s.empty else (None, f"Empty series for {ticker}.")


def normalise(series: pd.Series) -> pd.Series:
    first = series.dropna().iloc[0]
    return series if first == 0 else (series / first) * 100


def compute_drawdown(series: pd.Series) -> pd.Series:
    peak = series.cummax()
    return (series - peak) / peak * 100


def compute_stats(series: pd.Series, label: str) -> dict:
    series = series.dropna()
    if len(series) < 2:
        return {"Label": label, "Total Return": "—", "Ann. Return": "—",
                "Ann. Volatility": "—", "Sharpe Ratio": "—",
                "Max Drawdown": "—", "Best Day": "—", "Worst Day": "—", "Days": "0"}
    dr      = series.pct_change().dropna()
    n_years = len(series) / 252
    tr      = series.iloc[-1] / series.iloc[0] - 1
    ar      = (1 + tr) ** (1 / n_years) - 1 if n_years > 0 else 0
    vol     = dr.std() * np.sqrt(252)
    return {
        "Label":           label,
        "Total Return":    f"{tr:+.2%}",
        "Ann. Return":     f"{ar:+.2%}",
        "Ann. Volatility": f"{vol:.2%}",
        "Sharpe Ratio":    f"{ar/vol:.2f}" if vol > 0 else "—",
        "Max Drawdown":    f"{compute_drawdown(series).min():.2f}%",
        "Best Day":        f"{dr.max()*100:+.2f}%",
        "Worst Day":       f"{dr.min()*100:+.2f}%",
        "Days":            str(len(series)),
    }


def get_portfolio_start_date(portfolio: Portfolio) -> Optional[date]:
    dates = [datetime.strptime(t.date, "%Y-%m-%d").date()
             for h in portfolio.holdings.values() for t in h.transactions]
    return min(dates) if dates else None
