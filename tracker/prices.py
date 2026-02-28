"""
tracker/prices.py  —  Live price fetching + price cache
"""

from typing import Dict, Optional
import pandas as pd
import yfinance as yf


def _close_from_download(raw: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """
    Extract a clean (date × ticker) Close price DataFrame from yf.download() output.

    yfinance's output format has changed across versions:
      - Old (< 0.2.40)  : flat columns, "Close" is a column or a Series
      - New (>= 0.2.40) : MultiIndex columns — (field, ticker) or (ticker, field)
                          depending on how many tickers were requested

    This function handles all known variants defensively.
    """
    cols = raw.columns

    if isinstance(cols, pd.MultiIndex):
        # Determine which level holds the field names ("Close", "Open", etc.)
        # and which holds the ticker names
        level0_vals = set(cols.get_level_values(0))
        level1_vals = set(cols.get_level_values(1))

        if "Close" in level0_vals:
            # Format: (field, ticker)  — common for multi-ticker downloads
            close = raw["Close"]
        elif "Close" in level1_vals:
            # Format: (ticker, field)  — seen in some yfinance versions
            close = raw.xs("Close", axis=1, level=1)
        else:
            # Fall back: try the first level that has numeric-looking data
            # by taking the last column of each group
            raise KeyError(
                f"Could not find 'Close' in MultiIndex columns. "
                f"Level 0: {sorted(level0_vals)[:5]}, Level 1: {sorted(level1_vals)[:5]}"
            )

        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers[0].upper())

    else:
        # Flat column format (single ticker or old yfinance)
        if "Close" in cols:
            close = raw[["Close"]].copy()
            close.columns = [tickers[0].upper()]
        elif "close" in [c.lower() for c in cols]:
            col = next(c for c in cols if c.lower() == "close")
            close = raw[[col]].copy()
            close.columns = [tickers[0].upper()]
        else:
            # Last resort: assume the data IS the close prices
            close = raw.copy()

    # Normalise column names to uppercase tickers
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
                    if fresh and self._db is not None:
                        self._db.set_prices(fresh)
                    self._cache.update(fresh)
            except Exception as e:
                print(f"[Warning] Batch fetch failed: {e}")

        for ticker in to_fetch:
            if ticker not in self._cache:
                self.get_price(ticker)

        return {t: self._cache.get(t) for t in tickers}

    def clear_cache(self):
        """Clear in-memory cache only — DB cache is always preserved."""
        self._cache.clear()
        self._fresh.clear()
        if self._db is not None:
            self._cache.update(self._db.get_price_cache())
