"""
tracker/portfolio.py  —  Core data model

Key concepts:
  - dataclasses   : clean data-holding objects with auto-generated __init__
  - JSON          : simple text format for saving/loading data
  - @property     : computed attributes that look like regular fields
"""

import json, os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional, Dict


@dataclass
class Transaction:
    date: str       # ISO format: "2024-01-15"
    action: str     # "buy" or "sell"
    quantity: float
    price: float

    @property
    def total_cost(self) -> float:
        return self.quantity * self.price


@dataclass
class Holding:
    ticker: str
    name: str
    asset_type: str         # "stock", "crypto", or "etf"
    transactions: List[Transaction] = field(default_factory=list)
    manual_price: Optional[float] = None   # ← NEW: user-set price override

    @property
    def quantity(self) -> float:
        total = 0.0
        for t in self.transactions:
            total += t.quantity if t.action == "buy" else -t.quantity
        return total

    @property
    def average_cost(self) -> float:
        total_cost = total_qty = 0.0
        for t in self.transactions:
            if t.action == "buy":
                total_cost += t.quantity * t.price
                total_qty  += t.quantity
            elif t.action == "sell" and total_qty > 0:
                total_cost -= total_cost * (t.quantity / total_qty)
                total_qty  -= t.quantity
        return total_cost / total_qty if total_qty else 0.0

    @property
    def total_invested(self) -> float:
        return sum(t.total_cost for t in self.transactions if t.action == "buy")

    def current_value(self, price: float) -> float:
        return self.quantity * price

    def unrealised_pnl(self, price: float) -> float:
        return (price - self.average_cost) * self.quantity

    def pnl_percent(self, price: float) -> float:
        return ((price - self.average_cost) / self.average_cost * 100
                if self.average_cost else 0.0)


@dataclass
class Snapshot:
    """
    A point-in-time record of total portfolio value.
    Saved manually by the user to track value over time.

    Key concept: @dataclass gives us __init__, __repr__ etc. for free,
    and asdict() lets us serialise it to JSON with one call.
    """
    date:            str    # "2024-01-15"
    total_value:     float
    total_invested:  float
    note:            str = ""

    @property
    def pnl(self) -> float:
        return self.total_value - self.total_invested

    @property
    def pnl_pct(self) -> float:
        return (self.pnl / self.total_invested * 100) if self.total_invested else 0.0


class Portfolio:
    DATA_FILE      = "portfolio_data.json"
    SNAPSHOT_FILE  = "portfolio_snapshots.json"

    def __init__(self):
        self.holdings:  Dict[str, Holding]  = {}
        self.snapshots: List[Snapshot]      = []
        self.load()

    # ── Holdings CRUD ─────────────────────────────────────────────────────────

    def add_transaction(self, ticker: str, name: str, asset_type: str,
                        action: str, quantity: float, price: float,
                        date: Optional[str] = None) -> None:
        if date is None:
            date = datetime.today().strftime("%Y-%m-%d")
        ticker = ticker.upper()
        if ticker not in self.holdings:
            self.holdings[ticker] = Holding(ticker=ticker, name=name,
                                            asset_type=asset_type.lower())
        self.holdings[ticker].transactions.append(
            Transaction(date=date, action=action, quantity=quantity, price=price))
        self.save()

    def remove_holding(self, ticker: str) -> bool:
        ticker = ticker.upper()
        if ticker in self.holdings:
            del self.holdings[ticker]
            self.save()
            return True
        return False

    def set_manual_price(self, ticker: str, price: Optional[float]) -> None:
        """Set or clear a manual price override for a ticker."""
        ticker = ticker.upper()
        if ticker in self.holdings:
            self.holdings[ticker].manual_price = price
            self.save()

    def get_holding(self, ticker: str) -> Optional[Holding]:
        return self.holdings.get(ticker.upper())

    def all_holdings(self) -> List[Holding]:
        return [h for h in self.holdings.values() if h.quantity > 0]

    # ── Snapshots ─────────────────────────────────────────────────────────────

    def add_snapshot(self, total_value: float, total_invested: float,
                     note: str = "") -> Snapshot:
        """Record current portfolio value as a snapshot."""
        snap = Snapshot(
            date=datetime.today().strftime("%Y-%m-%d"),
            total_value=round(total_value, 2),
            total_invested=round(total_invested, 2),
            note=note,
        )
        self.snapshots.append(snap)
        self.save_snapshots()
        return snap

    def delete_snapshot(self, index: int) -> None:
        if 0 <= index < len(self.snapshots):
            self.snapshots.pop(index)
            self.save_snapshots()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        data = {ticker: asdict(h) for ticker, h in self.holdings.items()}
        with open(self.DATA_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        if os.path.exists(self.DATA_FILE):
            with open(self.DATA_FILE) as f:
                data = json.load(f)
            for ticker, d in data.items():
                transactions = [Transaction(**t) for t in d.get("transactions", [])]
                self.holdings[ticker] = Holding(
                    ticker=d["ticker"], name=d["name"], asset_type=d["asset_type"],
                    transactions=transactions,
                    manual_price=d.get("manual_price"),   # backward-compatible
                )
        self.load_snapshots()

    def save_snapshots(self) -> None:
        with open(self.SNAPSHOT_FILE, "w") as f:
            json.dump([asdict(s) for s in self.snapshots], f, indent=2)

    def load_snapshots(self) -> None:
        if os.path.exists(self.SNAPSHOT_FILE):
            with open(self.SNAPSHOT_FILE) as f:
                self.snapshots = [Snapshot(**s) for s in json.load(f)]
