"""
tracker/prices.py  —  Live price fetching via Yahoo Finance
"""

from typing import Dict, Optional
import pandas as pd
import yfinance as yf


def _close_from_download(raw: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """
    Safely extract a (date x ticker) Close DataFrame from yf.download() output.
    Handles both the old flat format and the new MultiIndex format (yfinance >= 0.2.40).
    """
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw.xs("Close", axis=1, level=0)
    else:
        close = raw["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers[0].upper())
    close.columns = [str(c).upper() for c in close.columns]
    return close


class PriceFetcher:
    def __init__(self):
        self._cache: Dict[str, float] = {}

    def get_price(self, ticker: str) -> Optional[float]:
        ticker = ticker.upper()
        if ticker in self._cache:
            return self._cache[ticker]
        try:
            t     = yf.Ticker(ticker)
            price = t.fast_info.get("lastPrice") or t.fast_info.get("regularMarketPrice")
            if price is None:
                hist  = t.history(period="2d")
                price = float(hist["Close"].iloc[-1]) if not hist.empty else None
            if price is not None:
                self._cache[ticker] = float(price)
                return float(price)
        except Exception as e:
            print(f"[Warning] Could not fetch {ticker}: {e}")
        return None

    def get_prices(self, tickers: list) -> Dict[str, Optional[float]]:
        tickers  = [t.upper() for t in tickers]
        to_fetch = [t for t in tickers if t not in self._cache]

        if to_fetch:
            try:
                raw = yf.download(to_fetch, period="2d", progress=False, auto_adjust=True)
                if not raw.empty:
                    close = _close_from_download(raw, to_fetch)
                    for ticker in to_fetch:
                        if ticker in close.columns:
                            series = close[ticker].dropna()
                            if not series.empty:
                                self._cache[ticker] = float(series.iloc[-1])
            except Exception as e:
                print(f"[Warning] Batch fetch failed: {e}")

        # Fallback: fetch individually anything still missing
        for ticker in to_fetch:
            if ticker not in self._cache:
                self.get_price(ticker)

        return {t: self._cache.get(t) for t in tickers}

    def clear_cache(self):
        self._cache.clear()
