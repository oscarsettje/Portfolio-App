"""
tracker/models.py  —  Pure dataclasses, no dependencies on other tracker modules.

Keeping models here breaks the circular import:
  db.py       imports from models.py
  portfolio.py imports from models.py and db.py
  app.py      imports from models.py and portfolio.py
"""

from dataclasses import dataclass, field
from typing import List, Optional


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
