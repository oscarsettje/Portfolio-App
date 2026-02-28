"""
tracker/analysis.py  —  Diversification, correlation and stress testing
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

from tracker.models import Holding
from tracker.prices import _close_from_download


def portfolio_weights(holdings: List[Holding],
                      prices: Dict[str, Optional[float]]) -> Dict[str, float]:
    values = {h.ticker: h.current_value(p)
              for h in holdings if (p := prices.get(h.ticker))}
    total  = sum(values.values())
    return {t: v / total for t, v in values.items()} if total else {}


def concentration_hhi(weights: Dict[str, float]) -> float:
    return sum(w ** 2 for w in weights.values()) * 10_000


def by_asset_type(holdings: List[Holding],
                  prices: Dict[str, Optional[float]]) -> pd.DataFrame:
    rows: Dict[str, float] = {}
    for h in holdings:
        if (p := prices.get(h.ticker)):
            rows[h.asset_type.lower()] = rows.get(h.asset_type.lower(), 0) + h.current_value(p)
    total = sum(rows.values())
    return pd.DataFrame([{"Asset Type": k.upper(), "Value (€)": round(v, 2),
                           "Weight (%)": round(v / total * 100, 2)}
                          for k, v in sorted(rows.items(), key=lambda x: -x[1])]) if rows else pd.DataFrame()


def infer_geography(ticker: str) -> str:
    t = ticker.upper()
    for suffix, region in [
        (".DE","Europe"), (".F","Europe"),  (".AS","Europe"), (".PA","Europe"),
        (".MI","Europe"), (".MC","Europe"), (".SW","Europe"), (".L","Europe"),
        (".VI","Europe"), (".BR","Europe"), (".ST","Europe"), (".HE","Europe"),
        (".OL","Europe"), (".CO","Europe"),
        (".T","Asia"),    (".HK","Asia"),   (".SS","Asia"),   (".SZ","Asia"),
        (".KS","Asia"),   (".NS","Asia"),   (".BO","Asia"),
        (".AX","Australia/NZ"),
        (".TO","Canada"), (".V","Canada"),
        (".SA","LatAm"),  (".MX","LatAm"),
    ]:
        if t.endswith(suffix):
            return region
    if any(x in t for x in ["-USD","-EUR","-GBP","-BTC"]):
        return "Crypto (Global)"
    return "North America"


def by_geography(holdings: List[Holding],
                 prices: Dict[str, Optional[float]]) -> pd.DataFrame:
    rows: Dict[str, float] = {}
    for h in holdings:
        if (p := prices.get(h.ticker)):
            r = infer_geography(h.ticker)
            rows[r] = rows.get(r, 0) + h.current_value(p)
    total = sum(rows.values())
    return pd.DataFrame([{"Region": k, "Value (€)": round(v, 2),
                           "Weight (%)": round(v / total * 100, 2)}
                          for k, v in sorted(rows.items(), key=lambda x: -x[1])]) if rows else pd.DataFrame()


def fetch_sectors(tickers: List[str]) -> Dict[str, str]:
    sectors = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            sectors[ticker] = info.get("sector") or info.get("quoteType") or "Unknown"
        except Exception:
            sectors[ticker] = "Unknown"
    return sectors


def by_sector(holdings: List[Holding], prices: Dict[str, Optional[float]],
              sectors: Dict[str, str]) -> pd.DataFrame:
    rows: Dict[str, float] = {}
    for h in holdings:
        if (p := prices.get(h.ticker)):
            s = sectors.get(h.ticker, "Unknown")
            rows[s] = rows.get(s, 0) + h.current_value(p)
    total = sum(rows.values())
    return pd.DataFrame([{"Sector": k, "Value (€)": round(v, 2),
                           "Weight (%)": round(v / total * 100, 2)}
                          for k, v in sorted(rows.items(), key=lambda x: -x[1])]) if rows else pd.DataFrame()


def fetch_return_matrix(tickers: List[str],
                        period: str = "1y") -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Returns (returns_df, None) on success or (None, error_message) on failure.
    Uses weekly interval to reduce data volume and rate-limit pressure.
    Drops tickers with insufficient data rather than failing entirely.
    """
    try:
        raw = yf.download(tickers, period=period, interval="1wk",
                          progress=False, auto_adjust=True)
        if raw.empty:
            return None, ("No data returned — you may be rate-limited. "
                          "Wait ~60 min and try again.")
        close = _close_from_download(raw, tickers)
        threshold = len(close) * 0.8
        close = close.dropna(thresh=int(threshold), axis=1)
        if close.empty or len(close.columns) < 2:
            return None, "Insufficient data (need ≥2 tickers). Check ticker symbols."
        returns = close.pct_change().dropna()
        if len(returns) < 10:
            return None, "Too few data points — try a longer period."
        return returns, None
    except KeyError as e:
        return None, (f"Data processing error (not a rate limit): {e}. "
                      f"Try upgrading yfinance: pip install yfinance --upgrade")
    except Exception as e:
        msg = str(e)
        if "rate" in msg.lower() or "too many" in msg.lower() or "429" in msg:
            return None, "Rate-limited by Yahoo Finance — wait ~60 min and try again."
        return None, f"Download error: {msg}"


def avg_pairwise_correlation(corr: pd.DataFrame) -> float:
    n = len(corr)
    if n < 2:
        return 0.0
    return sum(corr.iloc[i, j] for i in range(n) for j in range(n) if i != j) / (n * (n - 1))


def portfolio_volatility(returns: pd.DataFrame, weights: Dict[str, float]) -> float:
    tickers = [t for t in returns.columns if t in weights]
    if not tickers:
        return 0.0
    w   = np.array([weights.get(t, 0) for t in tickers])
    w   = w / w.sum() if w.sum() > 0 else w
    cov = returns[tickers].cov().values * 52   # weekly → annualised
    return float(np.sqrt(w @ cov @ w))


PRESET_SCENARIOS = {
    "2008 Financial Crisis": {
        "description": "Global equity crash. Stocks -40%, broad ETFs -38%.",
        "shocks": {"stock": -0.40, "etf": -0.38, "crypto": -0.50},
    },
    "2020 COVID Crash": {
        "description": "Sharp 35% equity drawdown in ~5 weeks (Feb–Mar 2020).",
        "shocks": {"stock": -0.34, "etf": -0.32, "crypto": -0.50},
    },
    "2022 Rate Hike Cycle": {
        "description": "Fed raised rates aggressively. Growth stocks and bonds both fell.",
        "shocks": {"stock": -0.19, "etf": -0.18, "crypto": -0.65},
    },
    "Crypto Winter (2022)": {
        "description": "FTX collapse and broad crypto sell-off. Equities mildly affected.",
        "shocks": {"stock": -0.05, "etf": -0.04, "crypto": -0.75},
    },
    "Mild Recession": {
        "description": "Moderate economic slowdown — earnings disappoint, equities dip.",
        "shocks": {"stock": -0.20, "etf": -0.18, "crypto": -0.35},
    },
    "Interest Rate Spike (+2%)": {
        "description": "Sudden 200bps rate rise. Growth stocks hit hardest.",
        "shocks": {"stock": -0.15, "etf": -0.12, "crypto": -0.20},
    },
    "Equity Bull Run": {
        "description": "Strong risk-on rally — tech and growth lead.",
        "shocks": {"stock": +0.30, "etf": +0.25, "crypto": +0.60},
    },
}


def apply_stress(holdings: List[Holding], prices: Dict[str, Optional[float]],
                 shocks: Dict[str, float]) -> List[dict]:
    rows = []
    for h in holdings:
        if not (p := prices.get(h.ticker)):
            continue
        shock       = shocks.get(h.asset_type.lower(), 0.0)
        value_now   = h.current_value(p)
        value_after = value_now * (1 + shock)
        rows.append({
            "Ticker":      h.ticker,
            "Asset Type":  h.asset_type.upper(),
            "Shock":       shock,
            "Value Now":   round(value_now, 2),
            "Value After": round(value_after, 2),
            "Impact (€)":  round(value_after - value_now, 2),
            "Impact (%)":  round(shock * 100, 2),
        })
    return rows
