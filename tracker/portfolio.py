"""
tracker/portfolio.py
====================
Core data model. Handles holdings storage and persistence to JSON.

Key concepts introduced here:
  - dataclasses  : a clean way to define data-holding objects
  - JSON         : a simple text format for saving/loading data
  - datetime     : working with dates and times
  - List / Dict  : Python type hints for readability
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data Classes
# A @dataclass automatically creates __init__, __repr__ etc. for you.
# ---------------------------------------------------------------------------

@dataclass
class Transaction:
    """Represents a single buy or sell transaction."""
    date: str           # ISO format: "2024-01-15"
    action: str         # "buy" or "sell"
    quantity: float     # Number of shares / coins
    price: float        # Price paid per unit

    @property
    def total_cost(self) -> float:
        """Total value of this transaction."""
        return self.quantity * self.price


@dataclass
class Holding:
    """
    Represents one position in the portfolio.
    A position is built up from one or more transactions.
    """
    ticker: str                         # e.g. "AAPL", "BTC-USD", "SPY"
    name: str                           # e.g. "Apple Inc."
    asset_type: str                     # "stock", "crypto", or "etf"
    transactions: List[Transaction] = field(default_factory=list)

    # --- Computed properties (calculated on the fly, not stored) ---

    @property
    def quantity(self) -> float:
        """Net shares/coins owned (buys minus sells)."""
        total = 0.0
        for t in self.transactions:
            if t.action == "buy":
                total += t.quantity
            elif t.action == "sell":
                total -= t.quantity
        return total

    @property
    def average_cost(self) -> float:
        """
        Average cost basis per unit using a running weighted average.
        This is the standard method brokers use.
        """
        total_cost = 0.0
        total_qty = 0.0
        for t in self.transactions:
            if t.action == "buy":
                total_cost += t.quantity * t.price
                total_qty += t.quantity
            elif t.action == "sell" and total_qty > 0:
                # Reduce cost basis proportionally on sells
                sell_ratio = t.quantity / total_qty
                total_cost -= total_cost * sell_ratio
                total_qty -= t.quantity
        if total_qty == 0:
            return 0.0
        return total_cost / total_qty

    @property
    def total_invested(self) -> float:
        """Sum of all buy transactions."""
        return sum(t.total_cost for t in self.transactions if t.action == "buy")

    def current_value(self, current_price: float) -> float:
        """Market value at the given current price."""
        return self.quantity * current_price

    def unrealised_pnl(self, current_price: float) -> float:
        """Profit or loss compared to average cost."""
        return (current_price - self.average_cost) * self.quantity

    def pnl_percent(self, current_price: float) -> float:
        """Percentage gain/loss vs average cost."""
        if self.average_cost == 0:
            return 0.0
        return ((current_price - self.average_cost) / self.average_cost) * 100


class Portfolio:
    """
    The top-level portfolio object.
    Manages a collection of holdings and handles saving/loading from disk.
    """

    DATA_FILE = "portfolio_data.json"

    def __init__(self):
        # A dict mapping ticker -> Holding for fast lookup
        self.holdings: dict[str, Holding] = {}
        self.load()

    # -----------------------------------------------------------------------
    # CRUD operations
    # -----------------------------------------------------------------------

    def add_transaction(
        self,
        ticker: str,
        name: str,
        asset_type: str,
        action: str,
        quantity: float,
        price: float,
        date: Optional[str] = None,
    ) -> None:
        """Add a buy or sell transaction for a ticker."""
        if date is None:
            date = datetime.today().strftime("%Y-%m-%d")

        transaction = Transaction(date=date, action=action, quantity=quantity, price=price)

        ticker = ticker.upper()
        if ticker not in self.holdings:
            # First time we're seeing this ticker — create a new Holding
            self.holdings[ticker] = Holding(
                ticker=ticker,
                name=name,
                asset_type=asset_type.lower(),
                transactions=[],
            )

        self.holdings[ticker].transactions.append(transaction)
        self.save()

    def remove_holding(self, ticker: str) -> bool:
        """Delete all data for a ticker. Returns True if found."""
        ticker = ticker.upper()
        if ticker in self.holdings:
            del self.holdings[ticker]
            self.save()
            return True
        return False

    def get_holding(self, ticker: str) -> Optional[Holding]:
        return self.holdings.get(ticker.upper())

    def all_holdings(self) -> List[Holding]:
        """Return only positions where the net quantity is positive."""
        return [h for h in self.holdings.values() if h.quantity > 0]

    # -----------------------------------------------------------------------
    # Persistence — saving and loading from JSON
    # -----------------------------------------------------------------------

    def save(self) -> None:
        """
        Serialize the portfolio to JSON and write to disk.
        `asdict()` converts a dataclass (and nested dataclasses) to a plain dict.
        """
        data = {ticker: asdict(holding) for ticker, holding in self.holdings.items()}
        with open(self.DATA_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        """Load portfolio from JSON, or start fresh if no file exists."""
        if not os.path.exists(self.DATA_FILE):
            return  # Nothing to load — brand new portfolio

        with open(self.DATA_FILE, "r") as f:
            data = json.load(f)

        for ticker, holding_dict in data.items():
            # Re-create Transaction objects from the plain dicts
            transactions = [
                Transaction(**t) for t in holding_dict.get("transactions", [])
            ]
            self.holdings[ticker] = Holding(
                ticker=holding_dict["ticker"],
                name=holding_dict["name"],
                asset_type=holding_dict["asset_type"],
                transactions=transactions,
            )
