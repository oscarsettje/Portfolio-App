"""
tracker/planner.py  —  Rebalancing, savings plan, and investment goal calculations
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple
import math


# ── Rebalancing ────────────────────────────────────────────────────────────────

@dataclass
class RebalanceRow:
    ticker:      str
    name:        str
    current_val: float
    current_pct: float
    target_pct:  float
    drift_pct:   float      # current_pct - target_pct
    trade_value: float      # positive = buy, negative = sell
    trade_qty:   float      # units to trade (informational)
    price:       float


def compute_rebalance(
    holdings,           # List[Holding]
    prices: Dict[str, Optional[float]],
    targets: Dict[str, float],  # {ticker: target_pct}
    commission: float = 0.0,
    new_cash: float = 0.0,
    drift_threshold: float = 5.0,
) -> Tuple[List[RebalanceRow], float, float]:
    """
    Compute rebalancing trades needed to bring portfolio to target allocations.

    Strategy:
      1. Add any new cash to the total portfolio value
      2. For each holding, compute target value = (target_pct/100) * total
      3. Trade value = target_value - current_value
      4. Adjust for commission: only suggest trades where |trade_value| > commission
      5. Only flag holdings outside drift_threshold

    Returns (rows, total_value, allocated_pct)
    """
    priced = {h.ticker: h for h in holdings if prices.get(h.ticker)}
    if not priced:
        return [], 0.0, 0.0

    current_values = {
        t: h.quantity * prices[t]
        for t, h in priced.items()
        if h.quantity > 0
    }
    total_current = sum(current_values.values())
    total = total_current + new_cash
    if total <= 0:
        return [], 0.0, 0.0

    allocated_pct = sum(targets.get(t, 0) for t in current_values)
    rows: List[RebalanceRow] = []

    # Include all tickers that have either a current value or a target
    all_tickers = set(current_values.keys()) | set(targets.keys())

    for ticker in sorted(all_tickers):
        h = priced.get(ticker)
        if h is None:
            continue
        price = prices.get(ticker) or 0
        if price <= 0:
            continue

        current_val = current_values.get(ticker, 0.0)
        current_pct = (current_val / total * 100) if total > 0 else 0.0
        target_pct  = targets.get(ticker, 0.0)
        target_val  = (target_pct / 100) * total
        drift       = current_pct - target_pct
        trade_val   = target_val - current_val

        # Don't suggest tiny trades that don't cover commission
        if abs(trade_val) <= commission:
            trade_val = 0.0

        trade_qty = trade_val / price if price > 0 else 0.0

        rows.append(RebalanceRow(
            ticker=ticker,
            name=h.name,
            current_val=current_val,
            current_pct=current_pct,
            target_pct=target_pct,
            drift_pct=drift,
            trade_value=trade_val,
            trade_qty=trade_qty,
            price=price,
        ))

    # Sort by abs(drift) descending
    rows.sort(key=lambda r: abs(r.drift_pct), reverse=True)
    return rows, total, allocated_pct


# ── Savings plan ───────────────────────────────────────────────────────────────

@dataclass
class SavingsPlanRow:
    ticker:        str
    name:          str
    planned:       float   # monthly target €
    actual_mtd:    float   # bought this calendar month so far €
    remaining:     float   # planned - actual (if > 0 still to invest)
    on_track:      bool


def compute_savings_plan_status(
    holdings,
    prices: Dict[str, Optional[float]],
    plans: Dict[str, float],   # {ticker: monthly_amount}
) -> List[SavingsPlanRow]:
    """
    Compare planned monthly contributions against actual transactions this month.
    """
    holding_map = {h.ticker: h for h in holdings}
    today = date.today()
    month_start = today.replace(day=1).isoformat()

    rows = []
    for ticker, planned in plans.items():
        h = holding_map.get(ticker)
        name = h.name if h else ticker

        # Sum buys this calendar month
        actual_mtd = 0.0
        if h:
            for t in h.transactions:
                if t.action == "buy" and t.date >= month_start:
                    actual_mtd += t.quantity * t.price

        remaining = max(0.0, planned - actual_mtd)
        on_track  = actual_mtd >= planned * 0.95   # 5% tolerance

        rows.append(SavingsPlanRow(
            ticker=ticker,
            name=name,
            planned=planned,
            actual_mtd=actual_mtd,
            remaining=remaining,
            on_track=on_track,
        ))

    rows.sort(key=lambda r: r.ticker)
    return rows


def project_portfolio(
    current_value: float,
    monthly_contribution: float,
    annual_return: float,       # e.g. 0.07 for 7%
    years: int,
) -> List[Tuple[date, float]]:
    """
    Project portfolio value year by year using future value of annuity formula.
    Returns list of (date, value) tuples — one per month for smoothness.
    """
    monthly_rate = (1 + annual_return) ** (1 / 12) - 1
    points = []
    value = current_value
    today = date.today()

    for month in range(years * 12 + 1):
        yr  = today.year  + (today.month - 1 + month) // 12
        mo  = (today.month - 1 + month) % 12 + 1
        dt  = date(yr, mo, 1)
        points.append((dt, value))
        value = value * (1 + monthly_rate) + monthly_contribution

    return points


# ── Investment goal ────────────────────────────────────────────────────────────

@dataclass
class GoalStatus:
    target_value:       float
    current_value:      float
    progress_pct:       float
    monthly_savings:    float       # actual monthly savings rate from transactions
    assumed_return:     float

    # Projections
    months_to_goal:     Optional[int]   # at current savings + return rate
    required_monthly:   Optional[float] # to hit target by target_date

    target_date:        Optional[date]
    projected_date:     Optional[date]  # when goal will be reached

    # Scenarios: {label: [(date, value), ...]}
    scenarios:          Dict[str, List[Tuple[date, float]]]


def compute_goal_status(
    current_value:  float,
    monthly_savings: float,
    target_value:   float,
    assumed_return: float,
    target_date:    Optional[date] = None,
    horizon_years:  int = 30,
) -> GoalStatus:
    progress_pct = min((current_value / target_value * 100)
                       if target_value > 0 else 0, 100)

    monthly_rate = (1 + assumed_return) ** (1 / 12) - 1

    # How many months to reach target at current savings + assumed return?
    months_to_goal  = None
    projected_date  = None
    if monthly_savings > 0 or current_value > 0:
        v = current_value
        for m in range(1, horizon_years * 12 + 1):
            v = v * (1 + monthly_rate) + monthly_savings
            if v >= target_value:
                months_to_goal = m
                today = date.today()
                yr = today.year  + (today.month - 1 + m) // 12
                mo = (today.month - 1 + m) % 12 + 1
                projected_date = date(yr, mo, 1)
                break

    # Required monthly savings to hit target by target_date
    required_monthly = None
    if target_date:
        today = date.today()
        months_left = max(1,
            (target_date.year - today.year) * 12 +
            (target_date.month - today.month))
        # FV = PV*(1+r)^n + PMT * [((1+r)^n - 1) / r]
        factor = (1 + monthly_rate) ** months_left
        fv_current = current_value * factor
        remaining  = target_value - fv_current
        if monthly_rate > 0:
            required_monthly = remaining * monthly_rate / (factor - 1)
        else:
            required_monthly = remaining / months_left
        required_monthly = max(0.0, required_monthly)

    # Build scenario projections
    scenarios: Dict[str, List[Tuple[date, float]]] = {}
    scenario_defs = [
        ("Pessimistic (4%)",  0.04),
        (f"Base ({assumed_return*100:.0f}%)", assumed_return),
        ("Optimistic (10%)", 0.10),
    ]
    for label, rate in scenario_defs:
        scenarios[label] = project_portfolio(
            current_value, monthly_savings, rate, horizon_years)

    return GoalStatus(
        target_value=target_value,
        current_value=current_value,
        progress_pct=progress_pct,
        monthly_savings=monthly_savings,
        assumed_return=assumed_return,
        months_to_goal=months_to_goal,
        required_monthly=required_monthly,
        target_date=target_date,
        projected_date=projected_date,
        scenarios=scenarios,
    )


def estimate_monthly_savings(holdings) -> float:
    """
    Estimate the user's monthly savings rate from the last 3 months of buys.
    Returns average monthly net investment.
    """
    today = date.today()
    cutoff = date(today.year - (1 if today.month <= 3 else 0),
                  (today.month - 3) % 12 or 12, 1).isoformat()
    total = 0.0
    for h in holdings:
        for t in h.transactions:
            if t.date >= cutoff and t.action == "buy":
                total += t.quantity * t.price
    return total / 3
