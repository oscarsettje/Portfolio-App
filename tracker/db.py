"""
tracker/db.py  —  SQLite database layer

Design principles:
  - Single file database (portfolio.db) — easy to back up, no server needed
  - The Holding / Transaction / Snapshot dataclasses are unchanged so the
    rest of the app keeps working without modification
  - Every write is also mirrored to JSON (portfolio_data.json +
    portfolio_snapshots.json) as a human-readable backup
  - Thread-safe via check_same_thread=False (Streamlit runs in threads)
  - All SQL uses parameterised queries — no string formatting, no injection

Schema
──────
  holdings     : one row per ticker (name, asset_type, manual_price)
  transactions : one row per buy/sell event, FK → holdings.ticker
  snapshots    : one row per manual portfolio snapshot
  price_cache  : last known price per ticker (replaces price_cache.json)

Why SQLite over JSON?
  - Atomic writes — no corruption if the app crashes mid-save
  - Queries — "show me all sells in 2024" is a one-liner
  - Indexes — lookups are O(log n) not O(n)
  - Concurrent reads — Streamlit's rerun model is safer with SQLite
  - Future-proof — easy to add new tables without breaking old data
"""

import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional

from tracker.models import Holding, Snapshot, Transaction

DB_FILE            = "portfolio.db"
JSON_DATA_FILE     = "portfolio_data.json"
JSON_SNAPSHOT_FILE = "portfolio_snapshots.json"

# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS holdings (
    ticker       TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    asset_type   TEXT NOT NULL,
    manual_price REAL
);

CREATE TABLE IF NOT EXISTS transactions (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker   TEXT    NOT NULL REFERENCES holdings(ticker) ON DELETE CASCADE,
    date     TEXT    NOT NULL,
    action   TEXT    NOT NULL CHECK(action IN ('buy','sell')),
    quantity REAL    NOT NULL CHECK(quantity > 0),
    price    REAL    NOT NULL CHECK(price > 0)
);

CREATE INDEX IF NOT EXISTS idx_transactions_ticker ON transactions(ticker);
CREATE INDEX IF NOT EXISTS idx_transactions_date   ON transactions(date);

CREATE TABLE IF NOT EXISTS snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT    NOT NULL,
    total_value     REAL    NOT NULL,
    total_invested  REAL    NOT NULL,
    note            TEXT    DEFAULT ''
);

CREATE TABLE IF NOT EXISTS price_cache (
    ticker     TEXT PRIMARY KEY,
    price      REAL NOT NULL,
    updated_at TEXT NOT NULL
);
"""

# ── Connection management ─────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row   # rows behave like dicts
    conn.executescript(_SCHEMA)
    return conn


@contextmanager
def _tx(conn: sqlite3.Connection):
    """Context manager that commits on success, rolls back on error."""
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


# ── Database class ────────────────────────────────────────────────────────────

class Database:
    """
    All reads and writes go through this class.
    The Portfolio class in portfolio.py delegates to this instead of JSON files.
    """

    def __init__(self):
        self.conn = _connect()

    # ── Holdings ──────────────────────────────────────────────────────────────

    def get_all_holdings(self) -> Dict[str, Holding]:
        """Load every holding and its transactions from the database."""
        rows = self.conn.execute("SELECT * FROM holdings").fetchall()
        holdings = {}
        for row in rows:
            ticker = row["ticker"]
            txn_rows = self.conn.execute(
                "SELECT * FROM transactions WHERE ticker = ? ORDER BY date, id",
                (ticker,)
            ).fetchall()
            transactions = [
                Transaction(date=t["date"], action=t["action"],
                            quantity=t["quantity"], price=t["price"])
                for t in txn_rows
            ]
            holdings[ticker] = Holding(
                ticker=row["ticker"],
                name=row["name"],
                asset_type=row["asset_type"],
                manual_price=row["manual_price"],
                transactions=transactions,
            )
        return holdings

    def upsert_holding(self, ticker: str, name: str, asset_type: str,
                       manual_price: Optional[float] = None) -> None:
        """Insert or update a holding row (no transactions)."""
        with _tx(self.conn):
            self.conn.execute("""
                INSERT INTO holdings (ticker, name, asset_type, manual_price)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    name         = excluded.name,
                    asset_type   = excluded.asset_type,
                    manual_price = excluded.manual_price
            """, (ticker, name, asset_type, manual_price))

    def set_manual_price(self, ticker: str, price: Optional[float]) -> None:
        with _tx(self.conn):
            self.conn.execute(
                "UPDATE holdings SET manual_price = ? WHERE ticker = ?",
                (price, ticker))

    def delete_holding(self, ticker: str) -> None:
        """Delete holding and all its transactions (CASCADE handles txns)."""
        with _tx(self.conn):
            self.conn.execute("DELETE FROM holdings WHERE ticker = ?", (ticker,))

    # ── Transactions ──────────────────────────────────────────────────────────

    def add_transaction(self, ticker: str, date: str, action: str,
                        quantity: float, price: float) -> int:
        """Insert a transaction, return its new id."""
        with _tx(self.conn):
            cur = self.conn.execute("""
                INSERT INTO transactions (ticker, date, action, quantity, price)
                VALUES (?, ?, ?, ?, ?)
            """, (ticker, date, action, quantity, price))
        return cur.lastrowid

    def replace_transactions(self, ticker: str,
                              transactions: List[Transaction]) -> None:
        """
        Replace all transactions for a ticker in one atomic operation.
        Used when the user edits the transaction table in the Holdings page.
        """
        with _tx(self.conn):
            self.conn.execute("DELETE FROM transactions WHERE ticker = ?", (ticker,))
            self.conn.executemany("""
                INSERT INTO transactions (ticker, date, action, quantity, price)
                VALUES (?, ?, ?, ?, ?)
            """, [(ticker, t.date, t.action, t.quantity, t.price)
                  for t in transactions])

    def get_transactions(self, ticker: str) -> List[Transaction]:
        rows = self.conn.execute(
            "SELECT * FROM transactions WHERE ticker = ? ORDER BY date, id",
            (ticker,)
        ).fetchall()
        return [Transaction(date=r["date"], action=r["action"],
                            quantity=r["quantity"], price=r["price"])
                for r in rows]

    # ── Snapshots ─────────────────────────────────────────────────────────────

    def get_snapshots(self) -> List[Snapshot]:
        rows = self.conn.execute(
            "SELECT * FROM snapshots ORDER BY date, id"
        ).fetchall()
        return [Snapshot(date=r["date"], total_value=r["total_value"],
                         total_invested=r["total_invested"], note=r["note"] or "")
                for r in rows]

    def add_snapshot(self, date: str, total_value: float,
                     total_invested: float, note: str = "") -> Snapshot:
        with _tx(self.conn):
            self.conn.execute("""
                INSERT INTO snapshots (date, total_value, total_invested, note)
                VALUES (?, ?, ?, ?)
            """, (date, total_value, total_invested, note))
        return Snapshot(date=date, total_value=total_value,
                        total_invested=total_invested, note=note)

    def delete_snapshot(self, index: int) -> None:
        """Delete snapshot by its position in the ordered list."""
        rows = self.conn.execute(
            "SELECT id FROM snapshots ORDER BY date, id"
        ).fetchall()
        if 0 <= index < len(rows):
            with _tx(self.conn):
                self.conn.execute("DELETE FROM snapshots WHERE id = ?",
                                  (rows[index]["id"],))

    # ── Price cache ───────────────────────────────────────────────────────────

    def get_price_cache(self) -> Dict[str, float]:
        rows = self.conn.execute("SELECT ticker, price FROM price_cache").fetchall()
        return {r["ticker"]: r["price"] for r in rows}

    def set_price(self, ticker: str, price: float) -> None:
        with _tx(self.conn):
            self.conn.execute("""
                INSERT INTO price_cache (ticker, price, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    price      = excluded.price,
                    updated_at = excluded.updated_at
            """, (ticker, price, datetime.now().isoformat()))

    def set_prices(self, prices: Dict[str, float]) -> None:
        now = datetime.now().isoformat()
        with _tx(self.conn):
            self.conn.executemany("""
                INSERT INTO price_cache (ticker, price, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    price      = excluded.price,
                    updated_at = excluded.updated_at
            """, [(t, p, now) for t, p in prices.items()])

    def get_price_updated_at(self, ticker: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT updated_at FROM price_cache WHERE ticker = ?", (ticker,)
        ).fetchone()
        return row["updated_at"] if row else None

    # ── JSON backup ───────────────────────────────────────────────────────────

    def export_json_backup(self) -> None:
        """
        Write current database state to JSON files.
        Called automatically after every write so the backups stay in sync.
        """
        holdings = self.get_all_holdings()
        data = {ticker: asdict(h) for ticker, h in holdings.items()}
        try:
            with open(JSON_DATA_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[Warning] JSON backup failed: {e}")

        snapshots = self.get_snapshots()
        try:
            with open(JSON_SNAPSHOT_FILE, "w") as f:
                json.dump([asdict(s) for s in snapshots], f, indent=2)
        except Exception as e:
            print(f"[Warning] Snapshot JSON backup failed: {e}")

    # ── Stats / queries ───────────────────────────────────────────────────────

    def transaction_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]

    def holding_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM holdings").fetchone()[0]

    def transactions_for_year(self, year: int) -> List[dict]:
        """All transactions in a given year — useful for tax reporting later."""
        rows = self.conn.execute("""
            SELECT t.*, h.name, h.asset_type
            FROM transactions t
            JOIN holdings h ON t.ticker = h.ticker
            WHERE strftime('%Y', t.date) = ?
            ORDER BY t.date, t.id
        """, (str(year),)).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self.conn.close()
