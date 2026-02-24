"""
tracker/exporter.py  —  Excel and CSV export
"""

import csv, os, tempfile
from datetime import datetime
from typing import Dict, List, Optional

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from tracker.portfolio import Holding

# ── Colour constants ──────────────────────────────────────────────────────────
HEADER_BG  = "1A237E"
HEADER_FG  = "FFFFFF"
SUBHEAD_BG = "283593"
POS_FG     = "1B5E20"
NEG_FG     = "B71C1C"
ALT_ROW    = "E8EAF6"

def _border():
    s = Side(style="thin", color="BDBDBD")
    return Border(left=s, right=s, top=s, bottom=s)

def _header_font(bold=True, size=10):
    return Font(name="Arial", size=size, bold=bold, color=HEADER_FG)

def _header_fill(bg=HEADER_BG):
    return PatternFill("solid", fgColor=bg)

def _style(cell, value=None, font=None, fill=None, fmt=None, align="left"):
    if value is not None: cell.value = value
    if font:  cell.font = font
    if fill:  cell.fill = fill
    if fmt:   cell.number_format = fmt
    cell.border    = _border()
    cell.alignment = Alignment(horizontal=align)
    return cell

# ── Public API ────────────────────────────────────────────────────────────────
def export_to_excel(holdings: List[Holding],
                    prices: Dict[str, Optional[float]],
                    filename: Optional[str] = None) -> str:
    if filename is None:
        filename = f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    wb = openpyxl.Workbook()
    _summary_sheet(wb, holdings, prices)
    _transactions_sheet(wb, holdings)
    if "Sheet" in wb.sheetnames:
        del wb["Sheet"]
    wb.save(filename)
    return filename

def export_to_csv(holdings: List[Holding],
                  prices: Dict[str, Optional[float]],
                  filename: Optional[str] = None) -> str:
    if filename is None:
        filename = f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fields = ["ticker","name","type","quantity","avg_cost",
              "current_price","market_value","unrealised_pnl","pnl_percent"]
    with open(filename, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for h in holdings:
            p = prices.get(h.ticker)
            w.writerow({"ticker": h.ticker, "name": h.name, "type": h.asset_type,
                        "quantity":       round(h.quantity, 6),
                        "avg_cost":       round(h.average_cost, 4),
                        "current_price":  round(p, 4) if p else "N/A",
                        "market_value":   round(h.current_value(p), 2) if p else "N/A",
                        "unrealised_pnl": round(h.unrealised_pnl(p), 2) if p else "N/A",
                        "pnl_percent":    round(h.pnl_percent(p), 2) if p else "N/A"})
    return filename

# ── Summary sheet ─────────────────────────────────────────────────────────────
def _summary_sheet(wb, holdings, prices):
    ws = wb.create_sheet("Summary")

    # Title rows
    for row, text, size in [(1,"📊 Portfolio Summary",16),(2,f"Generated: {datetime.now().strftime('%d %b %Y  %H:%M')}",10)]:
        ws.merge_cells(f"A{row}:I{row}")
        c = ws[f"A{row}"]
        c.value = text
        c.font  = Font(name="Arial", size=size, bold=(row==1), italic=(row==2), color=HEADER_FG)
        c.fill  = _header_fill(HEADER_BG if row==1 else SUBHEAD_BG)
        c.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 30

    # Column headers
    headers = ["Ticker","Name","Type","Quantity",
               "Avg Cost (€)","Current Price (€)","Market Value (€)","Unrealised P&L (€)","P&L %"]
    for col, h in enumerate(headers, 1):
        _style(ws.cell(4, col, h), font=_header_font(), fill=_header_fill(), align="center")
    ws.row_dimensions[4].height = 20

    # Data rows
    total_value = total_invested = 0.0
    for i, h in enumerate(holdings):
        row   = 5 + i
        p     = prices.get(h.ticker)
        fill  = PatternFill("solid", fgColor=ALT_ROW if i%2==0 else "FFFFFF")
        base_font = Font(name="Arial", size=10)

        vals = [h.ticker, h.name, h.asset_type.upper(), h.quantity,
                h.average_cost,
                p if p is not None else "N/A",
                h.current_value(p) if p else "N/A",
                h.unrealised_pnl(p) if p else "N/A",
                h.pnl_percent(p)/100 if p else "N/A"]
        fmts = [None,None,None,"#,##0.0000","€#,##0.00","€#,##0.00","€#,##0.00",
                '€#,##0.00;[Red](€#,##0.00)',"0.00%;[Red]-0.00%"]

        for col, (val, fmt) in enumerate(zip(vals, fmts), 1):
            cell = ws.cell(row, col, val)
            cell.font   = base_font
            cell.fill   = fill
            cell.border = _border()
            cell.alignment = Alignment(horizontal="right" if col>3 else "left")
            if fmt: cell.number_format = fmt
            # Colour P&L cells
            if col == 8 and p:
                cell.font = Font(name="Arial", size=10, color=POS_FG if h.unrealised_pnl(p)>=0 else NEG_FG)
            if col == 9 and p:
                cell.font = Font(name="Arial", size=10, color=POS_FG if h.pnl_percent(p)>=0 else NEG_FG)

        if p: total_value += h.current_value(p)
        total_invested += h.total_invested

    # Totals row
    tr  = 5 + len(holdings)
    pnl = total_value - total_invested
    sub_fill = _header_fill(SUBHEAD_BG)
    bold     = Font(name="Arial", bold=True)

    for col in range(1, 10):
        ws.cell(tr, col).fill   = sub_fill
        ws.cell(tr, col).border = _border()

    _style(ws.cell(tr, 1, "TOTAL"), font=Font(name="Arial", bold=True, color=HEADER_FG))
    _style(ws.cell(tr, 7, total_value), font=Font(name="Arial",bold=True,color=HEADER_FG),
           fmt="€#,##0.00", align="right")
    _style(ws.cell(tr, 8, pnl), font=Font(name="Arial",bold=True,color=POS_FG if pnl>=0 else NEG_FG),
           fmt='€#,##0.00;[Red](€#,##0.00)', align="right")
    if total_invested:
        _style(ws.cell(tr, 9, pnl/total_invested),
               font=Font(name="Arial",bold=True,color=POS_FG if pnl>=0 else NEG_FG),
               fmt="0.00%", align="right")

    for i, w in enumerate([10,22,8,12,14,16,16,18,10], 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A5"

# ── Transactions sheet ────────────────────────────────────────────────────────
def _transactions_sheet(wb, holdings):
    ws = wb.create_sheet("Transactions")
    ws.merge_cells("A1:F1")
    _style(ws["A1"], "Transaction History",
           font=Font(name="Arial",size=14,bold=True,color=HEADER_FG),
           fill=_header_fill(), align="center")

    for col, h in enumerate(["Ticker","Date","Action","Quantity","Price (€)","Total (€)"], 1):
        _style(ws.cell(3, col, h), font=_header_font(), fill=_header_fill(), align="center")

    row = 4
    for h in holdings:
        for t in h.transactions:
            fill = PatternFill("solid", fgColor="E8F5E9" if t.action=="buy" else "FFEBEE")
            vals = [h.ticker, t.date, t.action.upper(), t.quantity, t.price, t.total_cost]
            fmts = [None, None, None, "#,##0.0000", "€#,##0.00", "€#,##0.00"]
            for col, (val, fmt) in enumerate(zip(vals, fmts), 1):
                cell = ws.cell(row, col, val)
                cell.font   = Font(name="Arial", size=10)
                cell.fill   = fill
                cell.border = _border()
                if fmt: cell.number_format = fmt
            row += 1

    for i, w in enumerate([10,12,8,14,14,14], 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A4"
