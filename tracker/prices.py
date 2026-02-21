"""
tracker/prices.py
==================
Fetches live prices from Yahoo Finance via the `yfinance` library.

Key concepts introduced here:
  - External libraries  : using code written by others (yfinance)
  - Caching             : storing results temporarily to avoid repeat API calls
  - Exception handling  : gracefully dealing with network errors
  - Type hints          : Dict[str, float] makes code self-documenting
"""

from typing import Dict, Optional
import yfinance as yf


class PriceFetcher:
    """
    Fetches current market prices using Yahoo Finance.

    We cache prices in a simple dict for the lifetime of the session so that
    multiple references to the same ticker don't trigger extra API calls.
    """

    def __init__(self):
        # Simple in-memory cache: {"AAPL": 175.23, "BTC-USD": 62000.0, ...}
        self._cache: Dict[str, float] = {}

    def get_price(self, ticker: str) -> Optional[float]:
        """
        Return the latest price for a ticker.
        Returns None if the price cannot be fetched.
        """
        ticker = ticker.upper()

        # Return cached price if we already fetched it this session
        if ticker in self._cache:
            return self._cache[ticker]

        try:
            # yf.Ticker wraps the Yahoo Finance API for a single symbol
            data = yf.Ticker(ticker)

            # `fast_info` is a lightweight way to get the last price
            price = data.fast_info.get("lastPrice") or data.fast_info.get("regularMarketPrice")

            if price is None:
                # Fallback: pull recent history and use the last close
                hist = data.history(period="2d")
                if not hist.empty:
                    price = float(hist["Close"].iloc[-1])

            if price is not None:
                self._cache[ticker] = float(price)
                return float(price)

        except Exception as e:
            # Network errors, invalid tickers, etc. — we handle gracefully
            print(f"  [Warning] Could not fetch price for {ticker}: {e}")

        return None

    def get_prices(self, tickers: list) -> Dict[str, Optional[float]]:
        """
        Fetch prices for multiple tickers at once.
        yfinance supports batch downloading which is faster than one-by-one.
        """
        # Filter to only tickers not already cached
        to_fetch = [t.upper() for t in tickers if t.upper() not in self._cache]

        if to_fetch:
            try:
                # yf.download fetches multiple tickers in a single request
                raw = yf.download(to_fetch, period="2d", progress=False, auto_adjust=True)

                if not raw.empty:
                    close = raw["Close"] if len(to_fetch) > 1 else raw[["Close"]]
                    # Get the last available price for each ticker
                    for ticker in to_fetch:
                        col = ticker if len(to_fetch) > 1 else "Close"
                        try:
                            if len(to_fetch) > 1:
                                price = float(close[ticker].dropna().iloc[-1])
                            else:
                                price = float(raw["Close"].dropna().iloc[-1])
                            self._cache[ticker] = price
                        except Exception:
                            pass  # Will fall through to individual fetch

            except Exception as e:
                print(f"  [Warning] Batch fetch failed, trying individually: {e}")

        # For any still missing, try one-by-one
        for ticker in to_fetch:
            if ticker not in self._cache:
                self.get_price(ticker)

        return {t.upper(): self._cache.get(t.upper()) for t in tickers}

    def clear_cache(self) -> None:
        """Force a refresh of all prices on next fetch."""
        self._cache.clear()
