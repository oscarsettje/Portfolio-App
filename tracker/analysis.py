"""
tracker/analysis.py  —  Portfolio Analysis

Three analysis sections:
  1. Diversification  : how concentrated the portfolio is by asset type,
                        geography (inferred from ticker suffix) and sector
                        (fetched from yfinance when available)
  2. Correlation      : rolling correlation heatmap using daily returns,
                        plus a concentration risk score
  3. Stress testing   : apply historical or custom shocks to each position
                        and compute the total portfolio impact
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

from tracker.portfolio import Holding


# ── Diversification ───────────────────────────────────────────────────────────

def portfolio_weights(holdings: List[Holding],
                      prices: Dict[str, Optional[float]]) -> Dict[str, float]:
    """Return {ticker: weight} where weights sum to 1.0."""
    values = {h.ticker: h.current_value(p)
              for h in holdings if (p := prices.get(h.ticker))}
    total  = sum(values.values())
    return {t: v / total for t, v in values.items()} if total else {}


def concentration_hhi(weights: Dict[str, float]) -> float:
    """
    Herfindahl–Hirschman Index — standard measure of concentration.
    Range: 1/n (perfectly diversified) → 1.0 (one holding = 100%)
    Multiply by 10,000 for the traditional HHI scale (0–10,000).
    A score below 1,500 is generally considered diversified.
    """
    return sum(w ** 2 for w in weights.values()) * 10_000


def by_asset_type(holdings: List[Holding],
                  prices: Dict[str, Optional[float]]) -> pd.DataFrame:
    """Aggregate portfolio value by asset type (stock / crypto / etf)."""
    rows: Dict[str, float] = {}
    for h in holdings:
        p = prices.get(h.ticker)
        if p is None: continue
        key = h.asset_type.lower()
        rows[key] = rows.get(key, 0) + h.current_value(p)
    total = sum(rows.values())
    return pd.DataFrame([
        {"Asset Type": k.upper(), "Value (€)": round(v, 2),
         "Weight (%)": round(v / total * 100, 2)}
        for k, v in sorted(rows.items(), key=lambda x: -x[1])
    ]) if rows else pd.DataFrame()


def infer_geography(ticker: str) -> str:
    """
    Infer broad geography from ticker suffix.
    This is a best-effort heuristic — not 100% accurate.
    """
    t = ticker.upper()
    suffix_map = {
        ".DE": "Europe", ".F": "Europe",  ".AS": "Europe", ".PA": "Europe",
        ".MI": "Europe", ".MC": "Europe", ".SW": "Europe", ".L":  "Europe",
        ".VI": "Europe", ".BR": "Europe", ".LS": "Europe", ".ST": "Europe",
        ".HE": "Europe", ".OL": "Europe", ".CO": "Europe",
        ".T":  "Asia",   ".HK": "Asia",   ".SS": "Asia",   ".SZ": "Asia",
        ".KS": "Asia",   ".NS": "Asia",   ".BO": "Asia",
        ".AX": "Australia/NZ",
        ".TO": "Canada", ".V":  "Canada",
        ".SA": "LatAm",  ".MX": "LatAm",
    }
    for suffix, region in suffix_map.items():
        if t.endswith(suffix):
            return region
    if "-USD" in t or "-EUR" in t or "-GBP" in t or "-BTC" in t:
        return "Crypto (Global)"
    return "North America"   # default — most plain tickers are US


def by_geography(holdings: List[Holding],
                 prices: Dict[str, Optional[float]]) -> pd.DataFrame:
    rows: Dict[str, float] = {}
    for h in holdings:
        p = prices.get(h.ticker)
        if p is None: continue
        region = infer_geography(h.ticker)
        rows[region] = rows.get(region, 0) + h.current_value(p)
    total = sum(rows.values())
    return pd.DataFrame([
        {"Region": k, "Value (€)": round(v, 2),
         "Weight (%)": round(v / total * 100, 2)}
        for k, v in sorted(rows.items(), key=lambda x: -x[1])
    ]) if rows else pd.DataFrame()


def fetch_sectors(tickers: List[str]) -> Dict[str, str]:
    """
    Fetch sector info from yfinance for each ticker.
    Returns {ticker: sector_name}. Falls back to asset type on failure.
    This can be slow — call once and cache in session state.
    """
    sectors = {}
    for ticker in tickers:
        try:
            info   = yf.Ticker(ticker).info
            sector = info.get("sector") or info.get("quoteType") or "Unknown"
            sectors[ticker] = sector
        except Exception:
            sectors[ticker] = "Unknown"
    return sectors


def by_sector(holdings: List[Holding], prices: Dict[str, Optional[float]],
              sectors: Dict[str, str]) -> pd.DataFrame:
    rows: Dict[str, float] = {}
    for h in holdings:
        p = prices.get(h.ticker)
        if p is None: continue
        sector = sectors.get(h.ticker, "Unknown")
        rows[sector] = rows.get(sector, 0) + h.current_value(p)
    total = sum(rows.values())
    return pd.DataFrame([
        {"Sector": k, "Value (€)": round(v, 2),
         "Weight (%)": round(v / total * 100, 2)}
        for k, v in sorted(rows.items(), key=lambda x: -x[1])
    ]) if rows else pd.DataFrame()


# ── Correlation ───────────────────────────────────────────────────────────────

def fetch_return_matrix(tickers: List[str], period: str = "1y") -> Optional[pd.DataFrame]:
    """
    Download daily close prices and compute daily % returns.
    Returns a DataFrame of shape (days, tickers).
    """
    try:
        raw = yf.download(tickers, period=period, progress=False, auto_adjust=True)
        if raw.empty:
            return None
        close = raw["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers[0].upper())
        else:
            close.columns = [c.upper() for c in close.columns]
        return close.pct_change().dropna()
    except Exception:
        return None


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()


def avg_pairwise_correlation(corr: pd.DataFrame) -> float:
    """
    Average of the off-diagonal correlation values.
    0 = uncorrelated, 1 = everything moves together.
    """
    n = len(corr)
    if n < 2:
        return 0.0
    total = sum(corr.iloc[i, j]
                for i in range(n) for j in range(n) if i != j)
    return total / (n * (n - 1))


def portfolio_volatility(returns: pd.DataFrame,
                         weights: Dict[str, float]) -> float:
    """
    Annualised portfolio volatility using the full covariance matrix.
    Formula: σ_p = sqrt(w^T · Σ · w) · sqrt(252)

    This accounts for diversification benefits — a portfolio of correlated
    assets has higher volatility than one of uncorrelated ones with the
    same individual volatilities.
    """
    tickers = [t for t in returns.columns if t in weights]
    if not tickers:
        return 0.0
    w   = np.array([weights.get(t, 0) for t in tickers])
    cov = returns[tickers].cov().values * 252   # annualise
    return float(np.sqrt(w @ cov @ w))


# ── Stress testing ────────────────────────────────────────────────────────────

# Each scenario is a dict mapping asset_type → expected % return in that scenario.
# Values are fractions (e.g. -0.40 = 40% drop).
# Geography and individual overrides can also be layered in.

PRESET_SCENARIOS = {
    "2008 Financial Crisis": {
        "description": "Global equity crash. Stocks -40%, crypto didn't exist, bonds +5%.",
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
        "description": "Sudden 200bps rate rise. Growth stocks and long-duration assets hit hardest.",
        "shocks": {"stock": -0.15, "etf": -0.12, "crypto": -0.20},
    },
    "Equity Bull Run": {
        "description": "Strong risk-on rally — tech and growth lead.",
        "shocks": {"stock": +0.30, "etf": +0.25, "crypto": +0.60},
    },
}


def apply_stress(holdings: List[Holding],
                 prices: Dict[str, Optional[float]],
                 shocks: Dict[str, float]) -> List[dict]:
    """
    Apply a shock dict {asset_type: pct_change} to each holding.
    Returns a list of rows with before/after values and P&L impact.
    """
    rows = []
    for h in holdings:
        p = prices.get(h.ticker)
        if p is None: continue
        shock      = shocks.get(h.asset_type.lower(), 0.0)
        value_now  = h.current_value(p)
        value_after = value_now * (1 + shock)
        rows.append({
            "Ticker":        h.ticker,
            "Asset Type":    h.asset_type.upper(),
            "Shock":         shock,
            "Value Now":     round(value_now, 2),
            "Value After":   round(value_after, 2),
            "Impact (€)":    round(value_after - value_now, 2),
            "Impact (%)":    round(shock * 100, 2),
        })
    return rows
