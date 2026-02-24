"""
tracker/prices.py  —  Live price fetching via Yahoo Finance

Key concepts:
  - Caching    : avoid redundant API calls within a session
  - Batch fetch: yf.download() retrieves multiple tickers in one request
  - Fallback   : if batch fails, retry each ticker individually
"""

from typing import Dict, Optional
import pandas as pd
import yfinance as yf


class PriceFetcher:
    def __init__(self):
        self._cache: Dict[str, float] = {}

    def get_price(self, ticker: str) -> Optional[float]:
        """Fetch a single price, using cache if available."""
        ticker = ticker.upper()
        if ticker in self._cache:
            return self._cache[ticker]
        try:
            t     = yf.Ticker(ticker)
            price = (t.fast_info.get("lastPrice")
                     or t.fast_info.get("regularMarketPrice"))
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
        """Fetch prices for multiple tickers, batch where possible."""
        tickers  = [t.upper() for t in tickers]
        to_fetch = [t for t in tickers if t not in self._cache]

        if to_fetch:
            try:
                raw = yf.download(to_fetch, period="2d", progress=False, auto_adjust=True)
                if not raw.empty:
                    close = raw["Close"]
                    # yfinance returns a Series for single ticker, DataFrame for multiple
                    if isinstance(close, pd.Series):
                        price = float(close.dropna().iloc[-1])
                        self._cache[to_fetch[0]] = price
                    else:
                        for ticker in to_fetch:
                            try:
                                self._cache[ticker] = float(close[ticker].dropna().iloc[-1])
                            except Exception:
                                pass
            except Exception as e:
                print(f"[Warning] Batch fetch failed: {e}")

        # Individual fallback for anything still missing
        for ticker in to_fetch:
            if ticker not in self._cache:
                self.get_price(ticker)

        return {t: self._cache.get(t) for t in tickers}

    def clear_cache(self):
        self._cache.clear()
