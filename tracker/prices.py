"""
tracker/prices.py  —  Live price fetching via Yahoo Finance

Price cache is now stored in the SQLite database (price_cache table)
instead of price_cache.json. Fallback behaviour is unchanged:
if Yahoo returns nothing, the last known price from the DB is used.
"""

from typing import Dict, Optional
import pandas as pd
import yfinance as yf


def _close_from_download(raw: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """
    Extract a (date × ticker) Close DataFrame from yf.download() output.
    Handles both old flat format and new MultiIndex format (yfinance >= 0.2.40).
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
    def __init__(self, db=None):
        self._db    = db
        self._cache: Dict[str, float] = {}
        self._fresh: set = set()   # tickers with a live price this session
        if self._db is not None:
            self._cache.update(self._db.get_price_cache())

    def _store(self, ticker: str, price: float) -> None:
        """Save a fresh price to memory, DB, and mark as live."""
        self._cache[ticker] = price
        self._fresh.add(ticker)
        if self._db is not None:
            self._db.set_price(ticker, price)

    def is_stale(self, ticker: str) -> bool:
        """True if price comes from DB cache, not a live fetch this session."""
        return ticker.upper() not in self._fresh

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
                self._store(ticker, float(price))
                return float(price)
        except Exception as e:
            print(f"[Warning] Could not fetch {ticker}: {e}")
        # Fall back to cached price (already in _cache from __init__)
        return self._cache.get(ticker)

    def get_prices(self, tickers: list) -> Dict[str, Optional[float]]:
        tickers  = [t.upper() for t in tickers]
        to_fetch = [t for t in tickers if t not in self._cache]

        if to_fetch:
            try:
                raw = yf.download(to_fetch, period="2d", progress=False, auto_adjust=True)
                if not raw.empty:
                    close = _close_from_download(raw, to_fetch)
                    fresh = {}
                    for ticker in to_fetch:
                        if ticker in close.columns:
                            series = close[ticker].dropna()
                            if not series.empty:
                                fresh[ticker] = float(series.iloc[-1])
                    # Batch-write to DB in one transaction
                    if fresh and self._db is not None:
                        self._db.set_prices(fresh)
                    self._cache.update(fresh)
            except Exception as e:
                print(f"[Warning] Batch fetch failed: {e}")

        # Individual fallback for anything still missing
        for ticker in to_fetch:
            if ticker not in self._cache:
                self.get_price(ticker)

        return {t: self._cache.get(t) for t in tickers}

    def clear_cache(self):
        """Clear in-memory cache only — DB cache is always preserved."""
        self._cache.clear()
        # Reload last-known prices from DB so fallback still works
        if self._db is not None:
            self._cache.update(self._db.get_price_cache())
