"""
tracker/tax.py  —  German Abgeltungsteuer calculations

Rules implemented:
  - Abgeltungsteuer  : 25% flat on realised gains and dividends
  - Solidaritätszuschlag : 5.5% on top → effective rate 26.375%
  - Sparerpauschbetrag   : €1,000 annual allowance (single filer)
  - FIFO cost basis      : required under German law (§ 20 EStG)
  - Commission included in cost basis (raises purchase price, lowers taxable gain)
  - Withholding tax on dividends credited against Abgeltungsteuer owed

Note: Kirchensteuer (8–9%) is NOT included — it varies by person and Bundesland.
Note: Loss offsetting across years (Verlustverrechnungstopf) is simplified here —
      losses reduce the taxable base within the same year only. Carry-forward is
      noted but not auto-applied (requires your broker's records).

All amounts in EUR.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from tracker.models import Dividend, Holding, Transaction

# ── German tax constants ───────────────────────────────────────────────────────
ABGELTUNGSTEUER     = 0.25
SOLI_RATE           = 0.055          # on Abgeltungsteuer amount
EFFECTIVE_RATE      = ABGELTUNGSTEUER * (1 + SOLI_RATE)   # = 0.26375
SPARERPAUSCHBETRAG  = 1_000.0        # € per year, single filer


@dataclass
class FifoLot:
    """A tax lot: shares acquired at a specific date and cost basis."""
    date:       str
    quantity:   float
    unit_cost:  float   # price per share + commission per share


@dataclass
class RealisedGain:
    """One sell event matched against FIFO lots."""
    ticker:        str
    sell_date:     str
    quantity:      float
    proceeds:      float     # sell price × qty (before commission)
    commission:    float     # sell commission
    cost_basis:    float     # FIFO cost of the matched lots
    gain:          float     # proceeds - commission - cost_basis
    lots_matched:  List[dict] = field(default_factory=list)

    @property
    def is_gain(self) -> bool:
        return self.gain > 0


@dataclass
class YearSummary:
    year:               int
    realised_gains:     float    # sum of gains from sells
    realised_losses:    float    # sum of losses from sells (positive number)
    dividend_income:    float    # gross dividends
    withholding_paid:   float    # withholding tax already deducted at source
    total_commissions:  float
    net_taxable:        float    # gains + dividends - losses (before allowance)
    after_allowance:    float    # net_taxable minus Sparerpauschbetrag (floor 0)
    abgeltungsteuer:    float    # 25% on after_allowance
    soli:               float    # 5.5% on abgeltungsteuer
    withholding_credit: float    # withholding_paid credited against tax owed
    tax_owed:           float    # abgeltungsteuer + soli - withholding_credit (floor 0)
    allowance_used:     float    # how much of the €1,000 was consumed
    allowance_remaining:float    # leftover allowance


# ── FIFO engine ───────────────────────────────────────────────────────────────

def build_fifo_lots(transactions: List[Transaction]) -> Tuple[List[RealisedGain], List[FifoLot]]:
    """
    Process all transactions chronologically using FIFO.
    Returns:
      - List of RealisedGain events (one per sell)
      - Remaining open lots (current holdings)
    """
    lots: List[FifoLot] = []
    gains: List[RealisedGain] = []

    for t in sorted(transactions, key=lambda x: x.date):
        if t.action == "buy":
            # Commission spread across shares → raises unit cost basis
            unit_cost = t.price + (t.commission / t.quantity if t.quantity else 0)
            lots.append(FifoLot(date=t.date, quantity=t.quantity, unit_cost=unit_cost))

        elif t.action == "sell":
            remaining_sell_qty = t.quantity
            cost_basis         = 0.0
            lots_matched       = []

            # Consume lots oldest-first (FIFO)
            while remaining_sell_qty > 1e-9 and lots:
                lot = lots[0]
                matched = min(lot.quantity, remaining_sell_qty)
                cost_basis         += matched * lot.unit_cost
                lots_matched.append({"date": lot.date, "qty": matched,
                                     "unit_cost": lot.unit_cost})
                lot.quantity       -= matched
                remaining_sell_qty -= matched
                if lot.quantity < 1e-9:
                    lots.pop(0)

            proceeds = t.quantity * t.price
            gain     = proceeds - t.commission - cost_basis
            gains.append(RealisedGain(
                ticker="", sell_date=t.date, quantity=t.quantity,
                proceeds=proceeds, commission=t.commission,
                cost_basis=cost_basis, gain=gain,
                lots_matched=lots_matched,
            ))

    return gains, lots


def compute_realised_gains(holdings: Dict[str, Holding]) -> List[RealisedGain]:
    """Run FIFO for every holding and tag each gain with the ticker."""
    all_gains = []
    for ticker, holding in holdings.items():
        gains, _ = build_fifo_lots(holding.transactions)
        for g in gains:
            g.ticker = ticker
        all_gains.extend(gains)
    return sorted(all_gains, key=lambda g: g.sell_date)


def year_summary(year: int,
                 holdings: Dict[str, Holding],
                 dividends: List[Dividend],
                 sparerpauschbetrag: float = SPARERPAUSCHBETRAG) -> YearSummary:
    """
    Compute the full German tax picture for a given year.
    """
    y = str(year)

    # ── Realised gains / losses for this year ──
    gains = [g for g in compute_realised_gains(holdings)
             if g.sell_date.startswith(y)]
    realised_gains   = sum(g.gain for g in gains if g.gain > 0)
    realised_losses  = abs(sum(g.gain for g in gains if g.gain < 0))

    # ── Dividends for this year ──
    year_divs        = [d for d in dividends if d.date.startswith(y)]
    dividend_income  = sum(d.amount for d in year_divs)
    withholding_paid = sum(d.withholding_tax for d in year_divs)

    # ── Commissions ──
    total_commissions = sum(
        t.commission
        for h in holdings.values()
        for t in h.transactions
        if t.date.startswith(y)
    )

    # ── Net taxable base ──
    # Gains + dividends - losses (losses offset gains in same year)
    net_taxable     = max(0.0, realised_gains - realised_losses) + dividend_income
    after_allowance = max(0.0, net_taxable - sparerpauschbetrag)
    allowance_used  = min(sparerpauschbetrag, net_taxable)
    allowance_rem   = sparerpauschbetrag - allowance_used

    # ── Tax calculation ──
    abgelt          = round(after_allowance * ABGELTUNGSTEUER, 2)
    soli            = round(abgelt * SOLI_RATE, 2)
    # Withholding tax already paid is credited — can't go below 0
    wht_credit      = min(withholding_paid, abgelt + soli)
    tax_owed        = max(0.0, round(abgelt + soli - wht_credit, 2))

    return YearSummary(
        year=year,
        realised_gains=round(realised_gains, 2),
        realised_losses=round(realised_losses, 2),
        dividend_income=round(dividend_income, 2),
        withholding_paid=round(withholding_paid, 2),
        total_commissions=round(total_commissions, 2),
        net_taxable=round(net_taxable, 2),
        after_allowance=round(after_allowance, 2),
        abgeltungsteuer=abgelt,
        soli=soli,
        withholding_credit=round(wht_credit, 2),
        tax_owed=tax_owed,
        allowance_used=round(allowance_used, 2),
        allowance_remaining=round(allowance_rem, 2),
    )


def all_active_years(holdings: Dict[str, Holding],
                     dividends: List[Dividend]) -> List[int]:
    """Return all calendar years that have at least one transaction or dividend."""
    years = set()
    for h in holdings.values():
        for t in h.transactions:
            years.add(int(t.date[:4]))
    for d in dividends:
        years.add(int(d.date[:4]))
    return sorted(years)
