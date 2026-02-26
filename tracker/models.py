"""
tracker/models.py  —  Pure dataclasses, no dependencies on other tracker modules.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Transaction:
    date:       str    # "YYYY-MM-DD"
    action:     str    # "buy" or "sell"
    quantity:   float
    price:      float
    commission: float = 0.0   # broker fee in €

    @property
    def total_cost(self) -> float:
        """Gross cost including commission (for buys: cost basis; for sells: proceeds)."""
        return self.quantity * self.price

    @property
    def net_cost(self) -> float:
        """For buys: total paid including commission. For sells: proceeds minus commission."""
        return self.quantity * self.price + self.commission


@dataclass
class Dividend:
    ticker:          str
    date:            str    # "YYYY-MM-DD"
    amount:          float  # gross amount received in €
    withholding_tax: float = 0.0   # tax already withheld at source in €

    @property
    def net_amount(self) -> float:
        return self.amount - self.withholding_tax


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
        """
        Average cost per share including commissions (FIFO-aware for sells).
        Commission is spread across the shares bought, raising the cost basis.
        """
        total_cost = total_qty = 0.0
        for t in self.transactions:
            if t.action == "buy":
                total_cost += t.quantity * t.price + t.commission
                total_qty  += t.quantity
            elif t.action == "sell" and total_qty > 0:
                total_cost -= total_cost * (t.quantity / total_qty)
                total_qty  -= t.quantity
        return total_cost / total_qty if total_qty else 0.0

    @property
    def total_invested(self) -> float:
        """Total cash outflow for buys including commissions."""
        return sum(t.net_cost for t in self.transactions if t.action == "buy")

    @property
    def total_commissions(self) -> float:
        return sum(t.commission for t in self.transactions)

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
