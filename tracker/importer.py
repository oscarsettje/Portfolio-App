"""
tracker/importer.py  —  Portfolio Performance CSV importer

Handles the German-locale "Alle Buchungen" CSV export from Portfolio Performance.

Imported row types:
  Kauf     → buy transaction
  Verkauf  → sell transaction
  Dividende→ dividend record

Skipped row types:
  Einlieferung  — fractional share deliveries (savings plan artefacts)
  Einlage       — cash deposit
  Entnahme      — cash withdrawal
  Steuern       — tax booking
  Zinsen        — interest
  Gebühren      — standalone fee booking
  (any row with no Symbol)
"""

import csv
import io
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

# ── Row types ──────────────────────────────────────────────────────────────────
IMPORT_AS_BUY      = {"kauf"}
IMPORT_AS_SELL     = {"verkauf"}
IMPORT_AS_DIVIDEND = {"dividende"}
SKIP_TYPES = {
    "einlieferung", "auslieferung",
    "einlage", "entnahme",
    "steuern", "zinsen",
    "gebühren", "gebührenerstattung",
    "steuererstattung",
}


@dataclass
class ImportRow:
    row_type:       str          # 'buy', 'sell', or 'dividend'
    symbol:         str
    name:           str
    date:           str          # YYYY-MM-DD
    quantity:       float
    price:          float        # per unit
    amount:         float        # total gross
    commission:     float
    withholding:    float        # tax withheld (dividends only)
    raw_typ:        str          # original German type label


@dataclass
class ImportResult:
    imported_buys:      int = 0
    imported_sells:     int = 0
    imported_dividends: int = 0
    skipped_no_symbol:  int = 0
    skipped_type:       int = 0
    skipped_duplicate:  int = 0
    errors:             List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    @property
    def total_imported(self):
        return self.imported_buys + self.imported_sells + self.imported_dividends


# ── Number parsing ─────────────────────────────────────────────────────────────

def _parse_de_number(s: str) -> Optional[float]:
    """
    Parse a German-locale number string.
    '1.234,56' → 1234.56
    '1,229346' → 1.229346
    '140'      → 140.0
    ''         → None
    """
    s = s.strip()
    if not s:
        return None
    # Remove thousand-separator dots only when a comma follows later
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _parse_date(s: str) -> Optional[str]:
    """DD.MM.YYYY [HH:MM] → YYYY-MM-DD"""
    s = s.strip()
    for fmt in ("%d.%m.%Y %H:%M", "%d.%m.%Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


# ── CSV parsing ────────────────────────────────────────────────────────────────

def _detect_delimiter(raw: str) -> str:
    first_line = raw.split("\n")[0]
    return ";" if first_line.count(";") > first_line.count("\t") else "\t"


# Map of normalised column names → possible raw header variants
_COL_ALIASES = {
    "datum":      ["datum"],
    "typ":        ["typ"],
    "wertpapier": ["wertpapier"],
    "stueck":     ["stück", "stuck", "st\u00fcck", "stu\u0308ck", "st\xc3\xbcck",
                   "stã¼ck", "stã\x83\xbcck", "stück"],
    "kurs":       ["kurs"],
    "betrag":     ["betrag"],
    "gebuehren":  ["gebühren", "gebuhren", "geb\u00fchren", "geb\xc3\xbchren",
                   "gebã¼hren", "gebühren"],
    "steuern":    ["steuern"],
    "symbol":     ["symbol"],
    "notiz":      ["notiz"],
}


def _map_headers(raw_headers: List[str]) -> dict:
    """Return {normalised_key: column_index} for all recognised columns."""
    mapping = {}
    for idx, h in enumerate(raw_headers):
        h_lower = h.strip().lower()
        for key, aliases in _COL_ALIASES.items():
            if h_lower in aliases:
                mapping[key] = idx
                break
    return mapping


def parse_pp_csv(file_content: bytes) -> Tuple[List[ImportRow], List[str]]:
    """
    Parse a Portfolio Performance 'Alle Buchungen' CSV export.
    Returns (rows_to_import, warning_messages).
    """
    # Try encodings in order
    text = None
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            text = file_content.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        return [], ["Could not decode file — try exporting from PP with UTF-8 encoding."]

    delim = _detect_delimiter(text)
    reader = csv.reader(io.StringIO(text), delimiter=delim)
    rows = list(reader)

    if not rows:
        return [], ["File appears to be empty."]

    col = _map_headers(rows[0])
    warnings = []

    # Require at minimum: datum, typ, symbol
    for required in ("datum", "typ", "symbol"):
        if required not in col:
            warnings.append(
                f"Could not find required column '{required}' in CSV headers. "
                f"Headers found: {rows[0]}"
            )
    if warnings:
        return [], warnings

    parsed: List[ImportRow] = []

    for line_no, row in enumerate(rows[1:], start=2):
        if not row or all(c.strip() == "" for c in row):
            continue

        def get(key, default=""):
            idx = col.get(key)
            if idx is None or idx >= len(row):
                return default
            return row[idx].strip()

        typ      = get("typ").lower()
        symbol   = get("symbol").strip().upper()
        name     = get("wertpapier")
        date_raw = get("datum")

        # Skip rows with no symbol (cash flows etc.)
        if not symbol:
            continue

        # Skip non-security types
        if typ in SKIP_TYPES:
            continue

        # Only process known types
        if typ not in (IMPORT_AS_BUY | IMPORT_AS_SELL | IMPORT_AS_DIVIDEND):
            continue

        date = _parse_date(date_raw)
        if not date:
            warnings.append(f"Line {line_no}: Could not parse date '{date_raw}' — skipped.")
            continue

        quantity   = _parse_de_number(get("stueck"))   or 0.0
        kurs       = _parse_de_number(get("kurs"))     or 0.0
        betrag     = _parse_de_number(get("betrag"))   or 0.0
        commission = _parse_de_number(get("gebuehren")) or 0.0
        steuern    = _parse_de_number(get("steuern"))  or 0.0

        if typ in IMPORT_AS_BUY | IMPORT_AS_SELL:
            if quantity <= 0:
                warnings.append(f"Line {line_no}: {symbol} {typ} has quantity=0 — skipped.")
                continue
            # Use Kurs (price per unit) if available, otherwise derive from Betrag/Stück
            price = kurs if kurs > 0 else (betrag / quantity if quantity else 0)
            if price <= 0:
                warnings.append(f"Line {line_no}: {symbol} {typ} has price=0 — skipped.")
                continue
            row_type = "buy" if typ in IMPORT_AS_BUY else "sell"
            parsed.append(ImportRow(
                row_type=row_type, symbol=symbol, name=name,
                date=date, quantity=quantity, price=price,
                amount=betrag, commission=commission,
                withholding=0.0, raw_typ=typ,
            ))

        elif typ in IMPORT_AS_DIVIDEND:
            amount = betrag
            if amount <= 0:
                warnings.append(f"Line {line_no}: {symbol} dividend has amount=0 — skipped.")
                continue
            parsed.append(ImportRow(
                row_type="dividend", symbol=symbol, name=name,
                date=date, quantity=quantity, price=0.0,
                amount=amount, commission=0.0,
                withholding=steuern, raw_typ=typ,
            ))

    return parsed, warnings


# ── Duplicate detection ────────────────────────────────────────────────────────

def _existing_txn_keys(portfolio) -> set:
    """Set of (ticker, date, action, round(qty,4)) for all existing transactions."""
    keys = set()
    for h in portfolio.holdings.values():
        for t in h.transactions:
            keys.add((h.ticker, t.date, t.action, round(t.quantity, 4)))
    return keys


def _existing_div_keys(portfolio) -> set:
    """Set of (ticker, date, round(amount,2)) for existing dividends."""
    return {
        (d.ticker, d.date, round(d.amount, 2))
        for d in portfolio.all_dividends()
    }


# ── Import execution ───────────────────────────────────────────────────────────

def execute_import(rows: List[ImportRow], portfolio,
                   asset_type_map: dict) -> ImportResult:
    """
    Write parsed rows into the portfolio.
    asset_type_map: {symbol: asset_type} — user can override per-ticker asset type.
    Default asset type is inferred from symbol (ends in -USD/-EUR → crypto, else stock).
    """
    result = ImportResult()
    txn_keys = _existing_txn_keys(portfolio)
    div_keys  = _existing_div_keys(portfolio)

    for row in rows:
        symbol = row.symbol
        name   = row.name or symbol

        # Infer asset type if not provided
        atype = asset_type_map.get(symbol, _infer_asset_type(symbol))

        if row.row_type in ("buy", "sell"):
            key = (symbol, row.date, row.row_type, round(row.quantity, 4))
            if key in txn_keys:
                result.skipped_duplicate += 1
                continue
            try:
                portfolio.add_transaction(
                    ticker=symbol, name=name, asset_type=atype,
                    action=row.row_type, quantity=row.quantity,
                    price=row.price, date=row.date, commission=row.commission,
                )
                txn_keys.add(key)
                if row.row_type == "buy":
                    result.imported_buys += 1
                else:
                    result.imported_sells += 1
            except Exception as e:
                result.errors.append(f"{symbol} {row.date}: {e}")

        elif row.row_type == "dividend":
            key = (symbol, row.date, round(row.amount, 2))
            if key in div_keys:
                result.skipped_duplicate += 1
                continue
            try:
                # Ensure holding exists (may have dividends without a buy in this import)
                if symbol not in portfolio.holdings:
                    portfolio._db.upsert_holding(
                        portfolio.user_id, symbol, name, atype
                    )
                    portfolio._reload()
                portfolio.add_dividend(
                    ticker=symbol, date=row.date,
                    amount=row.amount, withholding_tax=row.withholding,
                )
                div_keys.add(key)
                result.imported_dividends += 1
            except Exception as e:
                result.errors.append(f"{symbol} dividend {row.date}: {e}")

    return result


def _infer_asset_type(symbol: str) -> str:
    s = symbol.upper()
    if s.endswith("-USD") or s.endswith("-EUR") or s.endswith("-GBP"):
        return "crypto"
    return "stock"
