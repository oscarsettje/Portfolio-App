"""
tracker/portfolio.py  —  Portfolio class (persistence layer)

Dataclasses (Holding, Transaction, Snapshot) have moved to tracker.models
to avoid a circular import with tracker.db.

Import map (no cycles):
  models.py   ← no tracker imports
  db.py       ← models.py
  portfolio.py← db.py, models.py
  app.py      ← portfolio.py, models.py, db.py
"""

from datetime import datetime
from typing import Dict, List, Optional

from tracker.db import Database
from tracker.models import Holding, Snapshot, Transaction


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
        self._load()

    # ── Holdings CRUD ─────────────────────────────────────────────────────────

    def add_transaction(self, ticker: str, name: str, asset_type: str,
                        action: str, quantity: float, price: float,
                        date: Optional[str] = None) -> None:
        ticker = ticker.upper()
        if date is None:
            date = datetime.today().strftime("%Y-%m-%d")
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
