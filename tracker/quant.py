"""
tracker/quant.py  —  Advanced portfolio metrics

All metrics use weekly returns by default (less noise than daily,
more data points than monthly).

Metrics implemented:
  - Sharpe Ratio        : excess return per unit of total risk
  - Sortino Ratio       : excess return per unit of downside risk only
  - Jensen's Alpha      : return above what CAPM would predict given beta
  - Beta                : sensitivity to benchmark movements
  - Max Drawdown        : worst peak-to-trough decline
  - Calmar Ratio        : annualised return / max drawdown
  - Value at Risk (VaR) : worst expected loss at a given confidence level
  - CVaR / Expected Shortfall : average loss beyond the VaR threshold
  - Rolling metrics     : Sharpe, Sortino, Beta computed over a rolling window
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

from tracker.prices import _close_from_download

# Risk-free rate — approximate ECB deposit rate (update manually if needed)
RISK_FREE_ANNUAL = 0.03
WEEKS_PER_YEAR   = 52


def _weekly_rf() -> float:
    return (1 + RISK_FREE_ANNUAL) ** (1 / WEEKS_PER_YEAR) - 1


def fetch_weekly_returns(tickers: list, period: str = "3y") -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Download weekly close prices and return (returns_df, None) or (None, error_message).
    Drops tickers with >20% missing values rather than failing entirely.
    """
    try:
        raw = yf.download(tickers, period=period, interval="1wk",
                          progress=False, auto_adjust=True)
        if raw.empty:
            return None, ("No data returned — you may be rate-limited. "
                          "Wait ~60 min and try again.")
        close = _close_from_download(raw, tickers)
        close.index = pd.to_datetime(close.index)
        if close.index.tz is not None:
            close.index = close.index.tz_localize(None)
        threshold = len(close) * 0.8
        close = close.dropna(thresh=int(threshold), axis=1)
        if close.empty:
            return None, "All tickers returned insufficient data. Check ticker symbols."
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


# ── Core metrics ──────────────────────────────────────────────────────────────

def sharpe_ratio(returns: pd.Series, rf: float = None) -> float:
    """
    Sharpe = (mean_return - rf) / std(returns)  ×  sqrt(52)  [annualised]

    Higher is better. Above 1.0 is good, above 2.0 is excellent.
    Uses weekly returns × sqrt(52) to annualise.
    """
    if rf is None: rf = _weekly_rf()
    excess = returns - rf
    if excess.std() == 0: return 0.0
    return float((excess.mean() / excess.std()) * np.sqrt(WEEKS_PER_YEAR))


def sortino_ratio(returns: pd.Series, rf: float = None) -> float:
    """
    Sortino = (mean_return - rf) / downside_std  ×  sqrt(52)

    Like Sharpe but only penalises downside volatility — a more
    investor-friendly measure since upside volatility is not a risk.
    """
    if rf is None: rf = _weekly_rf()
    excess        = returns - rf
    downside      = excess[excess < 0]
    downside_std  = downside.std()
    if downside_std == 0 or np.isnan(downside_std): return 0.0
    return float((excess.mean() / downside_std) * np.sqrt(WEEKS_PER_YEAR))


def beta_and_alpha(portfolio_returns: pd.Series,
                   benchmark_returns: pd.Series,
                   rf: float = None) -> Tuple[float, float]:
    """
    Beta  : covariance(portfolio, benchmark) / variance(benchmark)
            > 1 = amplifies market moves, < 1 = dampens them

    Jensen's Alpha : actual return - CAPM expected return
            = mean(portfolio) - [rf + beta × (mean(benchmark) - rf)]
            Positive alpha = outperformance after adjusting for market risk.
            Both annualised (× 52).
    """
    if rf is None: rf = _weekly_rf()
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 10:
        return 0.0, 0.0
    p = aligned.iloc[:, 0]
    b = aligned.iloc[:, 1]
    beta  = float(np.cov(p, b)[0, 1] / np.var(b)) if np.var(b) > 0 else 0.0
    alpha = float((p.mean() - (rf + beta * (b.mean() - rf))) * WEEKS_PER_YEAR)
    return beta, alpha


def max_drawdown(returns: pd.Series) -> float:
    """
    Maximum peak-to-trough decline as a fraction (e.g. -0.35 = -35%).
    Computed on the cumulative return series.
    """
    cum  = (1 + returns).cumprod()
    peak = cum.cummax()
    dd   = (cum - peak) / peak
    return float(dd.min())


def calmar_ratio(returns: pd.Series) -> float:
    """
    Calmar = Annualised Return / |Max Drawdown|

    Measures return earned per unit of drawdown risk.
    Higher is better; above 1.0 is solid.
    """
    ann_return = (1 + returns.mean()) ** WEEKS_PER_YEAR - 1
    mdd        = abs(max_drawdown(returns))
    return float(ann_return / mdd) if mdd > 0 else 0.0


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical VaR at the given confidence level.
    Returns the loss threshold (negative number) such that losses
    exceed this value only (1-confidence)% of the time.

    e.g. VaR(95%) = -0.03 means there is a 5% chance of losing more
    than 3% in a given week.
    """
    return float(np.percentile(returns, (1 - confidence) * 100))


def cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Conditional VaR (Expected Shortfall) — the average loss in the
    worst (1-confidence)% of weeks. More informative than VaR alone
    because it tells you how bad the bad weeks actually are.
    """
    var    = value_at_risk(returns, confidence)
    tail   = returns[returns <= var]
    return float(tail.mean()) if len(tail) > 0 else var


# ── Rolling metrics ───────────────────────────────────────────────────────────

def rolling_sharpe(returns: pd.Series, window: int = 52,
                   rf: float = None) -> pd.Series:
    """Rolling annualised Sharpe over a given window of weekly returns."""
    if rf is None: rf = _weekly_rf()
    excess = returns - rf
    roll_mean = excess.rolling(window).mean()
    roll_std  = excess.rolling(window).std()
    return (roll_mean / roll_std * np.sqrt(WEEKS_PER_YEAR)).dropna()


def rolling_sortino(returns: pd.Series, window: int = 52,
                    rf: float = None) -> pd.Series:
    """Rolling annualised Sortino over a given window of weekly returns."""
    if rf is None: rf = _weekly_rf()
    excess = returns - rf
    def _sortino_window(w):
        down = w[w < 0]
        std  = down.std()
        return (w.mean() / std * np.sqrt(WEEKS_PER_YEAR)) if std > 0 else np.nan
    return excess.rolling(window).apply(_sortino_window, raw=True).dropna()


def rolling_beta(portfolio_returns: pd.Series,
                 benchmark_returns: pd.Series,
                 window: int = 52) -> pd.Series:
    """Rolling beta of portfolio vs benchmark."""
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    p = aligned.iloc[:, 0]
    b = aligned.iloc[:, 1]
    def _beta(idx):
        pw = p.iloc[idx - window:idx]
        bw = b.iloc[idx - window:idx]
        var_b = np.var(bw)
        return np.cov(pw, bw)[0, 1] / var_b if var_b > 0 else np.nan
    result = pd.Series(
        [_beta(i) for i in range(window, len(p) + 1)],
        index=p.index[window - 1:]
    )
    return result.dropna()


# ── Summary table ─────────────────────────────────────────────────────────────

def compute_full_metrics(portfolio_returns: pd.Series,
                         benchmark_returns: pd.Series,
                         label: str = "Portfolio",
                         bench_label: str = "Benchmark") -> dict:
    """
    Compute all metrics for a portfolio and benchmark side by side.
    Returns a dict ready to display as a table.
    """
    beta, alpha = beta_and_alpha(portfolio_returns, benchmark_returns)
    ann_ret     = (1 + portfolio_returns.mean()) ** WEEKS_PER_YEAR - 1
    ann_vol     = portfolio_returns.std() * np.sqrt(WEEKS_PER_YEAR)
    mdd         = max_drawdown(portfolio_returns)
    var95       = value_at_risk(portfolio_returns, 0.95)
    cvar95      = cvar(portfolio_returns, 0.95)

    return {
        "Metric": [
            "Annualised Return", "Annualised Volatility",
            "Sharpe Ratio", "Sortino Ratio",
            "Beta", "Jensen's Alpha",
            "Max Drawdown", "Calmar Ratio",
            "VaR (95%, weekly)", "CVaR (95%, weekly)",
        ],
        label: [
            f"{ann_ret:+.2%}",
            f"{ann_vol:.2%}",
            f"{sharpe_ratio(portfolio_returns):.2f}",
            f"{sortino_ratio(portfolio_returns):.2f}",
            f"{beta:.2f}",
            f"{alpha:+.2%}",
            f"{mdd:.2%}",
            f"{calmar_ratio(portfolio_returns):.2f}",
            f"{var95:.2%}",
            f"{cvar95:.2%}",
        ],
        "_raw": {
            "ann_ret": ann_ret, "ann_vol": ann_vol,
            "sharpe": sharpe_ratio(portfolio_returns),
            "sortino": sortino_ratio(portfolio_returns),
            "beta": beta, "alpha": alpha,
            "mdd": mdd, "calmar": calmar_ratio(portfolio_returns),
            "var95": var95, "cvar95": cvar95,
        }
    }
