"""
tracker/portfolio.py  —  Core data model

The Holding / Transaction / Snapshot dataclasses are unchanged so every
other module (app.py, benchmark.py, quant.py, analysis.py) keeps working.

Portfolio now delegates all persistence to tracker.db.Database instead of
reading/writing JSON directly. JSON files are still written as backups after
every change.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from tracker.db import Database


# ── Dataclasses (unchanged — rest of app depends on these) ────────────────────

@dataclass
class Transaction:
    date:     str    # "YYYY-MM-DD"
    action:   str    # "buy" or "sell"
    quantity: float
    price:    float

    @property
    def total_cost(self) -> float:
        return self.quantity * self.price


@dataclass
class Holding:
    ticker:       str
    name:         str
    asset_type:   str
    transactions: List[Transaction] = field(default_factory=list)
    manual_price: Optional[float]   = None

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
    date:           str
    total_value:    float
    total_invested: float
    note:           str = ""

    @property
    def pnl(self) -> float:
        return self.total_value - self.total_invested

    @property
    def pnl_pct(self) -> float:
        return (self.pnl / self.total_invested * 100) if self.total_invested else 0.0


# ── Portfolio (now backed by SQLite) ─────────────────────────────────────────

class Portfolio:
    def __init__(self, db: Database):
        self._db       = db
        self.holdings:  Dict[str, Holding] = {}
        self.snapshots: List[Snapshot]     = []
        self._load()

    def _load(self) -> None:
        self.holdings  = self._db.get_all_holdings()
        self.snapshots = self._db.get_snapshots()

    def _reload(self) -> None:
        """Re-read from DB after any write — keeps in-memory state in sync."""
        self._load()

    # ── Holdings CRUD ─────────────────────────────────────────────────────────

    def add_transaction(self, ticker: str, name: str, asset_type: str,
                        action: str, quantity: float, price: float,
                        date: Optional[str] = None) -> None:
        ticker = ticker.upper()
        if date is None:
            date = datetime.today().strftime("%Y-%m-%d")
        # Ensure holding row exists first
        self._db.upsert_holding(ticker, name, asset_type.lower(),
                                self.holdings[ticker].manual_price
                                if ticker in self.holdings else None)
        self._db.add_transaction(ticker, date, action, quantity, price)
        self._db.export_json_backup()
        self._reload()

    def remove_holding(self, ticker: str) -> bool:
        ticker = ticker.upper()
        if ticker in self.holdings:
            self._db.delete_holding(ticker)
            self._db.export_json_backup()
            self._reload()
            return True
        return False

    def set_manual_price(self, ticker: str, price: Optional[float]) -> None:
        ticker = ticker.upper()
        if ticker in self.holdings:
            self._db.set_manual_price(ticker, price)
            self._db.export_json_backup()
            self._reload()

    def replace_transactions(self, ticker: str,
                              transactions: List[Transaction]) -> None:
        """Replace all transactions for a ticker atomically (used by editor)."""
        ticker = ticker.upper()
        self._db.replace_transactions(ticker, transactions)
        self._db.export_json_backup()
        self._reload()

    def get_holding(self, ticker: str) -> Optional[Holding]:
        return self.holdings.get(ticker.upper())

    def all_holdings(self) -> List[Holding]:
        return [h for h in self.holdings.values() if h.quantity > 0]

    # ── Snapshots ─────────────────────────────────────────────────────────────

    def add_snapshot(self, total_value: float, total_invested: float,
                     note: str = "") -> Snapshot:
        snap = self._db.add_snapshot(
            date=datetime.today().strftime("%Y-%m-%d"),
            total_value=round(total_value, 2),
            total_invested=round(total_invested, 2),
            note=note,
        )
        self._db.export_json_backup()
        self._reload()
        return snap

    def delete_snapshot(self, index: int) -> None:
        self._db.delete_snapshot(index)
        self._db.export_json_backup()
        self._reload()
