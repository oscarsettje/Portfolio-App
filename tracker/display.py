"""
tracker/display.py
==================
Renders portfolio data in the terminal using the `rich` library.

Key concepts introduced here:
  - f-strings         : formatted string literals for readable output
  - rich library      : professional terminal tables and styling
  - Conditional logic : colouring values green/red based on sign
  - Separation of concerns : display logic lives here, not in business logic
"""

from typing import Dict, List, Optional

from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel

from tracker.portfolio import Holding


console = Console()

# ── Palette ─────────────────────────────────────────────────────────────────
GAIN   = "green"
LOSS   = "red"
MUTED  = "grey62"
ACCENT = "steel_blue1"
HEAD   = "bold white"


# ── Formatters ───────────────────────────────────────────────────────────────

def _colour(value: float, text: str) -> str:
    if value > 0:  return f"[{GAIN}]{text}[/{GAIN}]"
    if value < 0:  return f"[{LOSS}]{text}[/{LOSS}]"
    return f"[{MUTED}]{text}[/{MUTED}]"

def _cur(value: float) -> str:
    return f"€{value:,.2f}"

def _pct(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}%"

def _arrow(value: float) -> str:
    if value > 0:  return f"[{GAIN}]▲[/{GAIN}]"
    if value < 0:  return f"[{LOSS}]▼[/{LOSS}]"
    return f"[{MUTED}]─[/{MUTED}]"


# ── Portfolio summary ────────────────────────────────────────────────────────

def print_portfolio_summary(
    holdings: List[Holding],
    prices: Dict[str, Optional[float]],
) -> None:
    if not holdings:
        console.print(f"\n  [{MUTED}]No holdings yet. Press 2 to add your first position.[/{MUTED}]\n")
        return

    table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style=f"bold {ACCENT}",
        show_edge=False,
        pad_edge=True,
        row_styles=["", "on grey7"],
    )

    table.add_column("",         width=2)
    table.add_column("Ticker",   style=HEAD, min_width=8)
    table.add_column("Name",     style=MUTED, min_width=18)
    table.add_column("Type",     style=MUTED, min_width=6)
    table.add_column("Qty",      justify="right", min_width=10)
    table.add_column("Avg Cost", justify="right", min_width=11, style=MUTED)
    table.add_column("Price",    justify="right", min_width=11)
    table.add_column("Value",    justify="right", min_width=13, style=HEAD)
    table.add_column("P&L",      justify="right", min_width=13)
    table.add_column("P&L %",    justify="right", min_width=9)

    total_value, total_pnl, total_invested = 0.0, 0.0, 0.0

    for h in holdings:
        price = prices.get(h.ticker)

        if price is None:
            na = f"[{MUTED}]—[/{MUTED}]"
            table.add_row(
                f"[{MUTED}]?[/{MUTED}]", h.ticker, h.name,
                h.asset_type.upper(),
                f"{h.quantity:,.4f}", _cur(h.average_cost),
                na, na, na, na,
            )
            continue

        value   = h.current_value(price)
        pnl     = h.unrealised_pnl(price)
        pnl_pct = h.pnl_percent(price)

        total_value    += value
        total_pnl      += pnl
        total_invested += h.total_invested

        table.add_row(
            _arrow(pnl),
            h.ticker,
            h.name,
            h.asset_type.upper(),
            f"{h.quantity:,.4f}",
            _cur(h.average_cost),
            _cur(price),
            _cur(value),
            _colour(pnl, _cur(pnl)),
            _colour(pnl_pct, _pct(pnl_pct)),
        )

    console.print()
    console.print(table)
    _print_totals(total_invested, total_value, total_pnl)


def _print_totals(invested: float, value: float, pnl: float) -> None:
    overall_pct = ((value - invested) / invested * 100) if invested else 0
    parts = [
        f"[{MUTED}]Invested[/{MUTED}]  [white]{_cur(invested)}[/white]",
        f"[{MUTED}]Value[/{MUTED}]  [bold white]{_cur(value)}[/bold white]",
        f"[{MUTED}]P&L[/{MUTED}]  {_colour(pnl, _cur(pnl))}  {_colour(overall_pct, _pct(overall_pct))}",
    ]
    console.print("  " + "     ".join(parts) + "\n")


# ── Allocation breakdown ─────────────────────────────────────────────────────

def print_allocation_breakdown(
    holdings: List[Holding],
    prices: Dict[str, Optional[float]],
) -> None:
    type_totals: Dict[str, float] = {}
    total = 0.0

    for h in holdings:
        price = prices.get(h.ticker)
        if price is None:
            continue
        value = h.current_value(price)
        type_totals[h.asset_type] = type_totals.get(h.asset_type, 0) + value
        total += value

    if total == 0:
        return

    table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style=f"bold {ACCENT}",
        show_edge=False,
        pad_edge=True,
    )
    table.add_column("Type",  min_width=8)
    table.add_column("Value", justify="right", min_width=13)
    table.add_column("",      min_width=36)

    BAR_WIDTH = 28
    for asset_type, value in sorted(type_totals.items(), key=lambda x: -x[1]):
        pct  = value / total * 100
        fill = round(pct / 100 * BAR_WIDTH)
        bar  = (
            f"[{ACCENT}]{'█' * fill}[/{ACCENT}]"
            f"[{MUTED}]{'░' * (BAR_WIDTH - fill)}[/{MUTED}]"
            f"  [{MUTED}]{pct:.1f}%[/{MUTED}]"
        )
        table.add_row(asset_type.upper(), _cur(value), bar)

    console.print(table)


# ── Holding detail ────────────────────────────────────────────────────────────

def print_holding_detail(holding: Holding, current_price: Optional[float]) -> None:
    console.print()

    title = f"[bold white]{holding.ticker}[/bold white]  [{MUTED}]{holding.name}[/{MUTED}]"
    lines = [
        f"[{MUTED}]Type[/{MUTED}]           {holding.asset_type.upper()}",
        f"[{MUTED}]Net Quantity[/{MUTED}]   [white]{holding.quantity:,.6f}[/white]",
        f"[{MUTED}]Avg Cost[/{MUTED}]       [white]{_cur(holding.average_cost)}[/white]",
        f"[{MUTED}]Total Invested[/{MUTED}]  [white]{_cur(holding.total_invested)}[/white]",
    ]

    if current_price is not None:
        pnl = holding.unrealised_pnl(current_price)
        pct = holding.pnl_percent(current_price)
        lines += [
            f"[{MUTED}]Current Price[/{MUTED}]  [white]{_cur(current_price)}[/white]",
            f"[{MUTED}]Market Value[/{MUTED}]   [white]{_cur(holding.current_value(current_price))}[/white]",
            f"[{MUTED}]P&L[/{MUTED}]            {_colour(pnl, _cur(pnl))}  {_colour(pct, _pct(pct))}",
        ]

    console.print(Panel("\n".join(lines), title=title, border_style=ACCENT, padding=(1, 2)))

    t_table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style=f"bold {ACCENT}",
        show_edge=False,
        pad_edge=True,
    )
    t_table.add_column("Date",     style=MUTED)
    t_table.add_column("Action",   min_width=6)
    t_table.add_column("Quantity", justify="right")
    t_table.add_column("Price",    justify="right")
    t_table.add_column("Total",    justify="right", style=HEAD)

    for t in holding.transactions:
        action_str = f"[{GAIN}]BUY[/{GAIN}]" if t.action == "buy" else f"[{LOSS}]SELL[/{LOSS}]"
        t_table.add_row(
            t.date, action_str,
            f"{t.quantity:,.4f}",
            _cur(t.price),
            _cur(t.total_cost),
        )

    console.print(t_table)
    console.print()
