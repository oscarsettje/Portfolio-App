"""
tracker/db.py  —  SQLite database layer

Schema
──────
  users        : user accounts (id, username, created_at)
  holdings     : one row per (user_id, ticker)
  transactions : one row per buy/sell, FK → holdings
  dividends    : dividend payments per (user_id, ticker)
  snapshots    : manual portfolio value snapshots per user
  price_cache  : last known price per ticker (shared across users)
"""

import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional

from tracker.models import Dividend, Holding, Snapshot, Transaction

DB_FILE            = "portfolio.db"
JSON_DATA_FILE     = "portfolio_data.json"
JSON_SNAPSHOT_FILE = "portfolio_snapshots.json"

DEFAULT_USER = "Default"   # existing data gets migrated to this user

_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    username   TEXT    NOT NULL UNIQUE,
    created_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS holdings (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    ticker       TEXT    NOT NULL,
    name         TEXT    NOT NULL,
    asset_type   TEXT    NOT NULL,
    manual_price REAL,
    UNIQUE(user_id, ticker)
);

CREATE TABLE IF NOT EXISTS transactions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    ticker     TEXT    NOT NULL,
    date       TEXT    NOT NULL,
    action     TEXT    NOT NULL CHECK(action IN ('buy','sell')),
    quantity   REAL    NOT NULL CHECK(quantity > 0),
    price      REAL    NOT NULL CHECK(price > 0),
    commission REAL    NOT NULL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_transactions_user   ON transactions(user_id);
CREATE INDEX IF NOT EXISTS idx_transactions_ticker ON transactions(user_id, ticker);
CREATE INDEX IF NOT EXISTS idx_transactions_date   ON transactions(date);

CREATE TABLE IF NOT EXISTS dividends (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    ticker          TEXT    NOT NULL,
    date            TEXT    NOT NULL,
    amount          REAL    NOT NULL CHECK(amount > 0),
    withholding_tax REAL    NOT NULL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_dividends_user   ON dividends(user_id);
CREATE INDEX IF NOT EXISTS idx_dividends_ticker ON dividends(user_id, ticker);
CREATE INDEX IF NOT EXISTS idx_dividends_date   ON dividends(date);

CREATE TABLE IF NOT EXISTS snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    date            TEXT    NOT NULL,
    total_value     REAL    NOT NULL,
    total_invested  REAL    NOT NULL,
    note            TEXT    DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_snapshots_user ON snapshots(user_id);

CREATE TABLE IF NOT EXISTS price_cache (
    ticker     TEXT PRIMARY KEY,
    price      REAL NOT NULL,
    updated_at TEXT NOT NULL
);
"""

# Migrations applied safely on every startup
_MIGRATIONS = [
    # v1: add commission to old single-user schema
    "ALTER TABLE transactions ADD COLUMN commission REAL NOT NULL DEFAULT 0.0",
    # v2: add users table (no-op if already exists — handled by IF NOT EXISTS in schema)
    # v3: add user_id to holdings (for migration from single-user schema)
    "ALTER TABLE holdings ADD COLUMN user_id INTEGER",
    # v4: add user_id to transactions
    "ALTER TABLE transactions ADD COLUMN user_id INTEGER",
    # v5: add user_id to dividends
    "ALTER TABLE dividends ADD COLUMN user_id INTEGER",
    # v6: add user_id to snapshots
    "ALTER TABLE snapshots ADD COLUMN user_id INTEGER",
]


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    for migration in _MIGRATIONS:
        try:
            conn.execute(migration)
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists or table doesn't exist yet
    return conn


def _migrate_to_multiuser(conn: sqlite3.Connection) -> None:
    """
    One-time migration: assign all existing data to the DEFAULT_USER.
    Runs only if there are rows with user_id = NULL.
    Safe to call repeatedly — does nothing after the first run.
    """
    needs_migration = conn.execute(
        "SELECT 1 FROM holdings WHERE user_id IS NULL LIMIT 1"
    ).fetchone()
    if not needs_migration:
        return

    # Ensure default user exists
    conn.execute("""
        INSERT OR IGNORE INTO users (username, created_at)
        VALUES (?, ?)
    """, (DEFAULT_USER, datetime.now().isoformat()))
    conn.commit()

    uid = conn.execute(
        "SELECT id FROM users WHERE username = ?", (DEFAULT_USER,)
    ).fetchone()["id"]

    # Assign all NULL user_ids to the default user
    for table in ("holdings", "transactions", "dividends", "snapshots"):
        try:
            conn.execute(f"UPDATE {table} SET user_id = ? WHERE user_id IS NULL", (uid,))
        except Exception:
            pass
    conn.commit()


@contextmanager
def _tx(conn: sqlite3.Connection):
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


class Database:
    def __init__(self):
        try:
            self.conn = _connect()
            _migrate_to_multiuser(self.conn)
        except Exception as e:
            raise RuntimeError(
                f"Could not open portfolio database: {e}\n"
                f"Make sure portfolio.db is not locked by another program "
                f"and that you have write permission to the folder."
            ) from e

    # ── User management ───────────────────────────────────────────────────────

    def get_all_users(self) -> List[dict]:
        rows = self.conn.execute(
            "SELECT id, username, created_at FROM users ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_user_id(self, username: str) -> Optional[int]:
        row = self.conn.execute(
            "SELECT id FROM users WHERE username = ?", (username,)
        ).fetchone()
        return row["id"] if row else None

    def create_user(self, username: str) -> int:
        with _tx(self.conn):
            cur = self.conn.execute(
                "INSERT INTO users (username, created_at) VALUES (?, ?)",
                (username, datetime.now().isoformat())
            )
        return cur.lastrowid

    def get_or_create_user(self, username: str) -> int:
        uid = self.get_user_id(username)
        if uid is None:
            uid = self.create_user(username)
        return uid

    def delete_user(self, user_id: int) -> None:
        """Delete a user and all their data (CASCADE handles related rows)."""
        with _tx(self.conn):
            self.conn.execute("DELETE FROM users WHERE id = ?", (user_id,))

    def rename_user(self, user_id: int, new_username: str) -> None:
        with _tx(self.conn):
            self.conn.execute(
                "UPDATE users SET username = ? WHERE id = ?",
                (new_username, user_id)
            )

    # ── Holdings ──────────────────────────────────────────────────────────────

    def get_all_holdings(self, user_id: int) -> Dict[str, Holding]:
        rows = self.conn.execute(
            "SELECT * FROM holdings WHERE user_id = ?", (user_id,)
        ).fetchall()
        holdings = {}
        for row in rows:
            ticker = row["ticker"]
            txn_rows = self.conn.execute(
                "SELECT * FROM transactions WHERE user_id = ? AND ticker = ? ORDER BY date, id",
                (user_id, ticker)
            ).fetchall()
            transactions = [
                Transaction(date=t["date"], action=t["action"],
                            quantity=t["quantity"], price=t["price"],
                            commission=t["commission"] if "commission" in t.keys() else 0.0)
                for t in txn_rows
            ]
            holdings[ticker] = Holding(
                ticker=row["ticker"], name=row["name"],
                asset_type=row["asset_type"], manual_price=row["manual_price"],
                transactions=transactions,
            )
        return holdings

    def upsert_holding(self, user_id: int, ticker: str, name: str,
                       asset_type: str, manual_price: Optional[float] = None) -> None:
        with _tx(self.conn):
            self.conn.execute("""
                INSERT INTO holdings (user_id, ticker, name, asset_type, manual_price)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, ticker) DO UPDATE SET
                    name         = excluded.name,
                    asset_type   = excluded.asset_type,
                    manual_price = excluded.manual_price
            """, (user_id, ticker, name, asset_type, manual_price))

    def set_manual_price(self, user_id: int, ticker: str,
                         price: Optional[float]) -> None:
        with _tx(self.conn):
            self.conn.execute(
                "UPDATE holdings SET manual_price = ? WHERE user_id = ? AND ticker = ?",
                (price, user_id, ticker))

    def delete_holding(self, user_id: int, ticker: str) -> None:
        with _tx(self.conn):
            self.conn.execute(
                "DELETE FROM holdings WHERE user_id = ? AND ticker = ?",
                (user_id, ticker))

    # ── Transactions ──────────────────────────────────────────────────────────

    def add_transaction(self, user_id: int, ticker: str, date: str, action: str,
                        quantity: float, price: float,
                        commission: float = 0.0) -> int:
        with _tx(self.conn):
            cur = self.conn.execute("""
                INSERT INTO transactions (user_id, ticker, date, action, quantity, price, commission)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, ticker, date, action, quantity, price, commission))
        return cur.lastrowid

    def replace_transactions(self, user_id: int, ticker: str,
                              transactions: List[Transaction]) -> None:
        with _tx(self.conn):
            self.conn.execute(
                "DELETE FROM transactions WHERE user_id = ? AND ticker = ?",
                (user_id, ticker))
            self.conn.executemany("""
                INSERT INTO transactions (user_id, ticker, date, action, quantity, price, commission)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [(user_id, ticker, t.date, t.action, t.quantity, t.price, t.commission)
                  for t in transactions])

    # ── Dividends ─────────────────────────────────────────────────────────────

    def add_dividend(self, user_id: int, ticker: str, date: str,
                     amount: float, withholding_tax: float = 0.0) -> int:
        with _tx(self.conn):
            cur = self.conn.execute("""
                INSERT INTO dividends (user_id, ticker, date, amount, withholding_tax)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, ticker, date, amount, withholding_tax))
        return cur.lastrowid

    def get_dividends(self, user_id: int,
                      ticker: Optional[str] = None) -> List[Dividend]:
        if ticker:
            rows = self.conn.execute(
                "SELECT * FROM dividends WHERE user_id = ? AND ticker = ? ORDER BY date, id",
                (user_id, ticker)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM dividends WHERE user_id = ? ORDER BY date, id",
                (user_id,)
            ).fetchall()
        return [Dividend(ticker=r["ticker"], date=r["date"],
                         amount=r["amount"], withholding_tax=r["withholding_tax"])
                for r in rows]

    def delete_dividend(self, user_id: int, div_id: int) -> None:
        with _tx(self.conn):
            self.conn.execute(
                "DELETE FROM dividends WHERE id = ? AND user_id = ?",
                (div_id, user_id))

    def get_dividends_with_ids(self, user_id: int) -> List[dict]:
        rows = self.conn.execute(
            "SELECT * FROM dividends WHERE user_id = ? ORDER BY date DESC, id DESC",
            (user_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Snapshots ─────────────────────────────────────────────────────────────

    def get_snapshots(self, user_id: int) -> List[Snapshot]:
        rows = self.conn.execute(
            "SELECT * FROM snapshots WHERE user_id = ? ORDER BY date, id",
            (user_id,)
        ).fetchall()
        return [Snapshot(date=r["date"], total_value=r["total_value"],
                         total_invested=r["total_invested"], note=r["note"] or "")
                for r in rows]

    def add_snapshot(self, user_id: int, date: str, total_value: float,
                     total_invested: float, note: str = "") -> Snapshot:
        with _tx(self.conn):
            self.conn.execute("""
                INSERT INTO snapshots (user_id, date, total_value, total_invested, note)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, date, total_value, total_invested, note))
        return Snapshot(date=date, total_value=total_value,
                        total_invested=total_invested, note=note)

    def delete_snapshot(self, user_id: int, index: int) -> None:
        rows = self.conn.execute(
            "SELECT id FROM snapshots WHERE user_id = ? ORDER BY date, id",
            (user_id,)
        ).fetchall()
        if 0 <= index < len(rows):
            with _tx(self.conn):
                self.conn.execute("DELETE FROM snapshots WHERE id = ?",
                                  (rows[index]["id"],))

    # ── Price cache (shared across users) ─────────────────────────────────────

    def get_price_cache(self) -> Dict[str, float]:
        rows = self.conn.execute("SELECT ticker, price FROM price_cache").fetchall()
        return {r["ticker"]: r["price"] for r in rows}

    def set_price(self, ticker: str, price: float) -> None:
        with _tx(self.conn):
            self.conn.execute("""
                INSERT INTO price_cache (ticker, price, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    price = excluded.price, updated_at = excluded.updated_at
            """, (ticker, price, datetime.now().isoformat()))

    def set_prices(self, prices: Dict[str, float]) -> None:
        now = datetime.now().isoformat()
        with _tx(self.conn):
            self.conn.executemany("""
                INSERT INTO price_cache (ticker, price, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    price = excluded.price, updated_at = excluded.updated_at
            """, [(t, p, now) for t, p in prices.items()])

    def get_price_updated_at(self, ticker: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT updated_at FROM price_cache WHERE ticker = ?", (ticker,)
        ).fetchone()
        return row["updated_at"] if row else None

    # ── JSON backup ───────────────────────────────────────────────────────────

    def export_json_backup(self, user_id: int, username: str) -> None:
        """Write per-user JSON backup files."""
        safe_name = username.replace(" ", "_").lower()
        data_file = f"portfolio_data_{safe_name}.json"
        snap_file = f"portfolio_snapshots_{safe_name}.json"
        try:
            holdings = self.get_all_holdings(user_id)
            with open(data_file, "w") as f:
                json.dump({t: asdict(h) for t, h in holdings.items()}, f, indent=2)
        except Exception as e:
            print(f"[Warning] JSON backup failed for {username}: {e}")
        try:
            with open(snap_file, "w") as f:
                json.dump([asdict(s) for s in self.get_snapshots(user_id)], f, indent=2)
        except Exception as e:
            print(f"[Warning] Snapshot backup failed for {username}: {e}")

    # ── Utility ───────────────────────────────────────────────────────────────

    def transaction_count(self, user_id: int) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE user_id = ?", (user_id,)
        ).fetchone()[0]

    def holding_count(self, user_id: int) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM holdings WHERE user_id = ?", (user_id,)
        ).fetchone()[0]

    def close(self) -> None:
        self.conn.close()
