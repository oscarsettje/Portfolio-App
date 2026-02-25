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


def _download_close(tickers: list, start: str) -> Optional[pd.DataFrame]:
    """Download daily closes, return clean (date x ticker) DataFrame."""
    try:
        raw = yf.download(tickers, start=start, progress=False, auto_adjust=True)
        if raw.empty:
            return None
        close = _close_from_download(raw, tickers)
        close.index = pd.to_datetime(close.index).tz_localize(None)
        return close
    except Exception:
        return None


def build_portfolio_value_series(portfolio: Portfolio,
                                  start_date: date) -> Optional[pd.Series]:
    if not portfolio.all_holdings():
        return None

    all_tickers = list(portfolio.holdings.keys())
    close = _download_close(all_tickers, start_date.strftime("%Y-%m-%d"))
    if close is None:
        return None

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

    series = pd.Series(values, index=close.index, name="Portfolio")
    nonzero = series[series > 0].index
    return series[nonzero[0]:] if len(nonzero) else None


def fetch_index_series(ticker: str, start_date: date) -> Optional[pd.Series]:
    close = _download_close([ticker], start_date.strftime("%Y-%m-%d"))
    if close is None:
        return None
    col = ticker.upper()
    if col not in close.columns:
        col = close.columns[0]
    s = close[col].dropna()
    s.name = ticker
    return s if not s.empty else None


def normalise(series: pd.Series) -> pd.Series:
    first = series.dropna().iloc[0]
    return series if first == 0 else (series / first) * 100


def compute_drawdown(series: pd.Series) -> pd.Series:
    peak = series.cummax()
    return (series - peak) / peak * 100


def compute_stats(series: pd.Series, label: str) -> dict:
    series = series.dropna()
    if len(series) < 2:
        return {"Label": label}
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
