"""
tracker/validation.py  —  Input validation rules

All validators return a list of error strings (empty = valid).
Keeping rules here means app.py just calls validate_*() and shows
the results — no business logic scattered through the UI layer.
"""

import re
from datetime import date, datetime
from typing import Dict, List, Optional

from tracker.models import Holding, Transaction

# Tickers that look obviously wrong
_BAD_TICKER_CHARS = re.compile(r'[^A-Z0-9.\-]')
_MAX_TICKER_LEN   = 12
_MIN_TICKER_LEN   = 1

# Sanity bounds — not hard limits, just "almost certainly a typo" guards
_MAX_PRICE        = 1_000_000.0   # €1M per share — covers Berkshire A etc.
_MAX_QUANTITY     = 1_000_000_000 # 1 billion units
_MAX_COMMISSION   = 1_000.0       # €1,000 broker fee


def validate_ticker(ticker: str) -> List[str]:
    errors = []
    t = ticker.strip().upper()
    if not t:
        errors.append("Ticker symbol cannot be empty.")
        return errors
    if len(t) < _MIN_TICKER_LEN:
        errors.append(f"Ticker is too short (minimum {_MIN_TICKER_LEN} character).")
    if len(t) > _MAX_TICKER_LEN:
        errors.append(f"Ticker '{t}' is too long (max {_MAX_TICKER_LEN} characters). "
                      f"Check the format — e.g. 'AAPL', 'SIE.DE', 'BTC-USD'.")
    if _BAD_TICKER_CHARS.search(t):
        errors.append(f"Ticker '{t}' contains invalid characters. "
                      f"Only letters, numbers, dots and hyphens are allowed.")
    if t.startswith('.') or t.endswith('.') or t.startswith('-') or t.endswith('-'):
        errors.append(f"Ticker '{t}' cannot start or end with '.' or '-'.")
    return errors


def validate_transaction(
        action: str,
        quantity: float,
        price: float,
        txn_date: date,
        commission: float,
        holding: Optional[Holding] = None,   # existing holding (None for first buy)
        existing_txns: Optional[List[Transaction]] = None,  # for sell checks
) -> List[str]:
    errors = []

    # Date checks
    today = date.today()
    if txn_date > today:
        errors.append(f"Date {txn_date} is in the future. "
                      f"Transactions can only be recorded for today or earlier.")

    # Price checks
    if price <= 0:
        errors.append("Price must be greater than zero.")
    elif price > _MAX_PRICE:
        errors.append(f"Price €{price:,.2f} seems unusually high. "
                      f"Please double-check — if correct, this is fine to proceed.")

    # Quantity checks
    if quantity <= 0:
        errors.append("Quantity must be greater than zero.")
    elif quantity > _MAX_QUANTITY:
        errors.append(f"Quantity {quantity:,.0f} seems extremely large. "
                      f"Please double-check.")

    # Commission checks
    if commission < 0:
        errors.append("Commission cannot be negative.")
    elif commission > _MAX_COMMISSION:
        errors.append(f"Commission €{commission:,.2f} seems unusually high. "
                      f"Please double-check.")
    elif commission > price * quantity * 0.1:
        # Commission > 10% of trade value is almost certainly a data entry error
        trade_val = price * quantity
        errors.append(f"Commission €{commission:,.2f} is more than 10% of the trade "
                      f"value €{trade_val:,.2f}. Please double-check.")

    # Sell-specific checks
    if action.lower() == "sell":
        if holding is None and (existing_txns is None or not existing_txns):
            errors.append("Cannot sell a holding you don't own yet.")
        else:
            # Calculate current quantity from existing transactions
            txns = existing_txns or (holding.transactions if holding else [])
            current_qty = sum(
                t.quantity if t.action == "buy" else -t.quantity
                for t in txns
            )
            if quantity > current_qty + 1e-9:
                errors.append(
                    f"Cannot sell {quantity:,.4f} units — you only hold "
                    f"{current_qty:,.4f}. Selling more than you own is not "
                    f"supported (no short selling)."
                )
            if current_qty <= 0:
                errors.append("You have no units of this holding to sell.")

    return errors


def validate_transaction_list(
        ticker: str,
        transactions: List[Transaction],
) -> List[str]:
    """
    Validate a complete list of transactions for a holding (used by the editor).
    Checks chronological sell quantities cumulatively.
    """
    errors = []
    running_qty = 0.0
    today = date.today()

    for i, t in enumerate(sorted(transactions, key=lambda x: x.date), 1):
        # Date format
        try:
            txn_date = datetime.strptime(t.date, "%Y-%m-%d").date()
        except ValueError:
            errors.append(f"Row {i}: '{t.date}' is not a valid date (expected YYYY-MM-DD).")
            continue

        if txn_date > today:
            errors.append(f"Row {i} ({t.date}): date is in the future.")

        # Action
        if t.action.lower() not in ("buy", "sell"):
            errors.append(f"Row {i}: action must be 'buy' or 'sell', got '{t.action}'.")
            continue

        # Quantity / price
        if t.quantity <= 0:
            errors.append(f"Row {i} ({t.date}): quantity must be > 0.")
        if t.price <= 0:
            errors.append(f"Row {i} ({t.date}): price must be > 0.")
        if t.commission < 0:
            errors.append(f"Row {i} ({t.date}): commission cannot be negative.")

        # Running quantity check
        if t.action.lower() == "buy":
            running_qty += t.quantity
        else:
            running_qty -= t.quantity
            if running_qty < -1e-9:
                errors.append(
                    f"Row {i} ({t.date}): sell of {t.quantity:,.4f} units would "
                    f"exceed holdings at that point. Check the order of your transactions."
                )

    return errors


def validate_dividend(
        amount: float,
        withholding_tax: float,
        div_date: date,
) -> List[str]:
    errors = []
    today = date.today()

    if amount <= 0:
        errors.append("Dividend amount must be greater than zero.")
    if withholding_tax < 0:
        errors.append("Withholding tax cannot be negative.")
    if withholding_tax > amount:
        errors.append(
            f"Withholding tax €{withholding_tax:,.2f} cannot exceed the gross "
            f"dividend amount €{amount:,.2f}."
        )
    if withholding_tax > amount * 0.5:
        errors.append(
            f"Withholding tax is {withholding_tax/amount*100:.0f}% of the gross "
            f"dividend — this seems very high. Please double-check."
        )
    if div_date > today:
        errors.append(f"Dividend date {div_date} is in the future.")

    return errors


def validate_name(name: str) -> List[str]:
    errors = []
    n = name.strip()
    if not n:
        errors.append("Please enter a name for this holding.")
    elif len(n) < 2:
        errors.append("Name is too short — please enter the full holding name.")
    elif len(n) > 100:
        errors.append("Name is too long (max 100 characters).")
    return errors
