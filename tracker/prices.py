"""
tracker/prices.py  —  Live price fetching via Yahoo Finance

Rate limiting strategy:
  - Prices are cached in memory for the session (avoids repeat calls)
  - Last known good prices are persisted to price_cache.json on disk
  - If Yahoo returns nothing, we fall back to the last known price
  - Disk cache only updates when a genuinely fresh price is received
"""

import json, os
from typing import Dict, Optional
import pandas as pd
import yfinance as yf

CACHE_FILE = "price_cache.json"


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


def _load_disk_cache() -> Dict[str, float]:
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_disk_cache(cache: Dict[str, float]) -> None:
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


class PriceFetcher:
    def __init__(self):
        self._cache: Dict[str, float] = {}           # in-memory for this session
        self._disk:  Dict[str, float] = _load_disk_cache()  # last known good prices

    def _store(self, ticker: str, price: float) -> None:
        """Save a fresh price to both memory and disk."""
        self._cache[ticker] = price
        self._disk[ticker]  = price
        _save_disk_cache(self._disk)

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

        # Fall back to last known good price from disk
        fallback = self._disk.get(ticker)
        if fallback is not None:
            print(f"[Info] Using cached price for {ticker}: {fallback}")
            self._cache[ticker] = fallback
        return fallback

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
                                self._store(ticker, float(series.iloc[-1]))
            except Exception as e:
                print(f"[Warning] Batch fetch failed: {e}")

        # Individual fallback for anything still missing (uses disk cache inside)
        for ticker in to_fetch:
            if ticker not in self._cache:
                self.get_price(ticker)

        return {t: self._cache.get(t) for t in tickers}

    def clear_cache(self):
        """Clear in-memory cache only — disk cache is always preserved."""
        self._cache.clear()
