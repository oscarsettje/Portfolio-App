"""
tracker/exporter.py
====================
Exports portfolio data to Excel (.xlsx) and CSV formats.

Key concepts introduced here:
  - openpyxl         : writing Excel files with formatting and formulas
  - csv module       : Python's built-in CSV writer
  - datetime         : stamping exports with date/time
  - Context managers : the `with open(...)` pattern for safe file handling
"""

import csv
from datetime import datetime
from typing import Dict, List, Optional

import openpyxl
from openpyxl.styles import (
    Font, PatternFill, Alignment, Border, Side
)
from openpyxl.utils import get_column_letter

from tracker.portfolio import Holding


# ---------------------------------------------------------------------------
# Colour constants for Excel
# ---------------------------------------------------------------------------
HEADER_BG  = "1A237E"   # Dark blue header
HEADER_FG  = "FFFFFF"   # White text
SUBHEAD_BG = "283593"
POS_FG     = "1B5E20"   # Dark green for gains
NEG_FG     = "B71C1C"   # Dark red for losses
ALT_ROW    = "E8EAF6"   # Light lavender for alternating rows


def _make_border(style="thin"):
    s = Side(style=style, color="BDBDBD")
    return Border(left=s, right=s, top=s, bottom=s)


def export_to_excel(
    holdings: List[Holding],
    prices: Dict[str, Optional[float]],
    filename: Optional[str] = None,
) -> str:
    """
    Export the portfolio to a professional Excel workbook.
    Returns the filename used.
    """
    if filename is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"portfolio_{stamp}.xlsx"

    wb = openpyxl.Workbook()

    _write_summary_sheet(wb, holdings, prices)
    _write_transactions_sheet(wb, holdings)

    # Remove the default empty sheet created by openpyxl
    if "Sheet" in wb.sheetnames:
        del wb["Sheet"]

    wb.save(filename)
    return filename


# ---------------------------------------------------------------------------
# Summary sheet
# ---------------------------------------------------------------------------

def _write_summary_sheet(wb, holdings: List[Holding], prices: Dict[str, Optional[float]]):
    ws = wb.create_sheet("Summary")

    # ---- Title area ----
    ws.merge_cells("A1:I1")
    title_cell = ws["A1"]
    title_cell.value = "📊 Portfolio Summary"
    title_cell.font = Font(name="Arial", size=16, bold=True, color=HEADER_FG)
    title_cell.fill = PatternFill("solid", fgColor=HEADER_BG)
    title_cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 30

    ws.merge_cells("A2:I2")
    date_cell = ws["A2"]
    date_cell.value = f"Generated: {datetime.now().strftime('%d %b %Y  %H:%M')}"
    date_cell.font = Font(name="Arial", size=10, italic=True, color=HEADER_FG)
    date_cell.fill = PatternFill("solid", fgColor=SUBHEAD_BG)
    date_cell.alignment = Alignment(horizontal="center")

    # ---- Column headers ----
    headers = [
        "Ticker", "Name", "Type", "Quantity",
        "Avg Cost ($)", "Current Price ($)", "Market Value ($)",
        "Unrealised P&L ($)", "P&L %"
    ]
    header_row = 4
    for col_idx, header in enumerate(headers, start=1):
        cell = ws.cell(row=header_row, column=col_idx, value=header)
        cell.font = Font(name="Arial", size=10, bold=True, color=HEADER_FG)
        cell.fill = PatternFill("solid", fgColor=HEADER_BG)
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = _make_border()

    ws.row_dimensions[header_row].height = 20

    # ---- Data rows ----
    data_start_row = header_row + 1
    total_value = 0.0
    total_invested = 0.0

    for row_offset, h in enumerate(holdings):
        row = data_start_row + row_offset
        price = prices.get(h.ticker)

        # Alternating row background
        row_fill = PatternFill("solid", fgColor=ALT_ROW) if row_offset % 2 == 0 else PatternFill("solid", fgColor="FFFFFF")

        values = [
            h.ticker,
            h.name,
            h.asset_type.upper(),
            h.quantity,
            h.average_cost,
            price if price is not None else "N/A",
            h.current_value(price) if price else "N/A",
            h.unrealised_pnl(price) if price else "N/A",
            h.pnl_percent(price) / 100 if price else "N/A",  # Excel % format
        ]

        for col_idx, value in enumerate(values, start=1):
            cell = ws.cell(row=row, column=col_idx, value=value)
            cell.font = Font(name="Arial", size=10)
            cell.fill = row_fill
            cell.border = _make_border()
            cell.alignment = Alignment(horizontal="right" if col_idx > 3 else "left")

            # Number formatting
            if col_idx == 4:
                cell.number_format = "#,##0.0000"
            elif col_idx in (5, 6, 7):
                cell.number_format = "$#,##0.00"
            elif col_idx == 8 and price:
                cell.number_format = '$#,##0.00;[Red]($#,##0.00)'
                pnl = h.unrealised_pnl(price)
                cell.font = Font(name="Arial", size=10,
                                 color=(POS_FG if pnl >= 0 else NEG_FG))
            elif col_idx == 9 and price:
                cell.number_format = "0.00%;[Red]-0.00%"
                pct = h.pnl_percent(price)
                cell.font = Font(name="Arial", size=10,
                                 color=(POS_FG if pct >= 0 else NEG_FG))

        if price:
            total_value += h.current_value(price)
        total_invested += h.total_invested

    # ---- Totals row ----
    total_row = data_start_row + len(holdings)
    ws.cell(row=total_row, column=1, value="TOTAL").font = Font(name="Arial", bold=True)
    ws.cell(row=total_row, column=7, value=total_value).number_format = "$#,##0.00"
    ws.cell(row=total_row, column=7).font = Font(name="Arial", bold=True)

    pnl_total = total_value - total_invested
    pnl_cell = ws.cell(row=total_row, column=8, value=pnl_total)
    pnl_cell.number_format = '$#,##0.00;[Red]($#,##0.00)'
    pnl_cell.font = Font(name="Arial", bold=True,
                         color=(POS_FG if pnl_total >= 0 else NEG_FG))

    if total_invested:
        pct_cell = ws.cell(row=total_row, column=9, value=pnl_total / total_invested)
        pct_cell.number_format = "0.00%"
        pct_cell.font = Font(name="Arial", bold=True,
                             color=(POS_FG if pnl_total >= 0 else NEG_FG))

    for col in range(1, 10):
        ws.cell(row=total_row, column=col).fill = PatternFill("solid", fgColor=SUBHEAD_BG)
        ws.cell(row=total_row, column=col).border = _make_border()

    # ---- Column widths ----
    col_widths = [10, 22, 8, 12, 14, 16, 16, 18, 10]
    for i, width in enumerate(col_widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = width

    ws.freeze_panes = "A5"


# ---------------------------------------------------------------------------
# Transactions sheet
# ---------------------------------------------------------------------------

def _write_transactions_sheet(wb, holdings: List[Holding]):
    ws = wb.create_sheet("Transactions")

    ws.merge_cells("A1:F1")
    ws["A1"].value = "Transaction History"
    ws["A1"].font = Font(name="Arial", size=14, bold=True, color=HEADER_FG)
    ws["A1"].fill = PatternFill("solid", fgColor=HEADER_BG)
    ws["A1"].alignment = Alignment(horizontal="center")

    headers = ["Ticker", "Date", "Action", "Quantity", "Price ($)", "Total ($)"]
    for col_idx, h in enumerate(headers, start=1):
        cell = ws.cell(row=3, column=col_idx, value=h)
        cell.font = Font(name="Arial", bold=True, color=HEADER_FG)
        cell.fill = PatternFill("solid", fgColor=HEADER_BG)
        cell.alignment = Alignment(horizontal="center")
        cell.border = _make_border()

    row = 4
    for holding in holdings:
        for t in holding.transactions:
            ws.cell(row=row, column=1, value=holding.ticker)
            ws.cell(row=row, column=2, value=t.date)
            ws.cell(row=row, column=3, value=t.action.upper())
            ws.cell(row=row, column=4, value=t.quantity).number_format = "#,##0.0000"
            ws.cell(row=row, column=5, value=t.price).number_format = "$#,##0.00"
            ws.cell(row=row, column=6, value=t.total_cost).number_format = "$#,##0.00"

            # Colour buy/sell rows
            action_color = "E8F5E9" if t.action == "buy" else "FFEBEE"
            fill = PatternFill("solid", fgColor=action_color)
            for col in range(1, 7):
                ws.cell(row=row, column=col).fill = fill
                ws.cell(row=row, column=col).border = _make_border()
                ws.cell(row=row, column=col).font = Font(name="Arial", size=10)

            row += 1

    col_widths = [10, 12, 8, 14, 14, 14]
    for i, width in enumerate(col_widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = width

    ws.freeze_panes = "A4"


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_to_csv(
    holdings: List[Holding],
    prices: Dict[str, Optional[float]],
    filename: Optional[str] = None,
) -> str:
    """Export a simple flat CSV of the current portfolio snapshot."""
    if filename is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"portfolio_{stamp}.csv"

    fieldnames = [
        "ticker", "name", "type", "quantity", "avg_cost",
        "current_price", "market_value", "unrealised_pnl", "pnl_percent"
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for h in holdings:
            price = prices.get(h.ticker)
            writer.writerow({
                "ticker":         h.ticker,
                "name":           h.name,
                "type":           h.asset_type,
                "quantity":       round(h.quantity, 6),
                "avg_cost":       round(h.average_cost, 4),
                "current_price":  round(price, 4) if price else "N/A",
                "market_value":   round(h.current_value(price), 2) if price else "N/A",
                "unrealised_pnl": round(h.unrealised_pnl(price), 2) if price else "N/A",
                "pnl_percent":    round(h.pnl_percent(price), 2) if price else "N/A",
            })

    return filename
