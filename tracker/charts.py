"""
tracker/charts.py
=================
Generates charts using matplotlib — clean, minimal style.

Key concepts introduced here:
  - matplotlib        : Python's most popular charting library
  - Figure/Axes API   : the "object oriented" way to build plots
  - rcParams          : global style settings applied once
  - tight_layout()    : automatically adjusts spacing
"""

from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from tracker.portfolio import Holding


# ── Global style ──────────────────────────────────────────────────────────────
# Setting rcParams once applies to every chart in this session.
BG      = "#0f0f0f"
SURFACE = "#1a1a1a"
BORDER  = "#2a2a2a"
TEXT    = "#cccccc"
MUTED   = "#666666"
GAIN    = "#4caf7d"
LOSS    = "#e05c5c"
BLUE    = "#5b9bd5"
PALETTE = ["#5b9bd5", "#4caf7d", "#e8a838", "#b07fd4", "#e05c5c", "#4db6ac", "#f06292"]

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    SURFACE,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   MUTED,
    "axes.titlecolor":   TEXT,
    "axes.titlesize":    13,
    "axes.titlepad":     16,
    "axes.grid":         True,
    "grid.color":        BORDER,
    "grid.linewidth":    0.6,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.facecolor":  SURFACE,
    "legend.edgecolor":  BORDER,
    "legend.labelcolor": TEXT,
    "legend.fontsize":   9,
    "text.color":        TEXT,
    "font.family":       "sans-serif",
    "figure.dpi":        120,
})


def _savefig(fig: plt.Figure, filename: str) -> None:
    plt.tight_layout()
    fig.savefig(filename, bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"  Saved: {filename}")


# ── Donut allocation chart ────────────────────────────────────────────────────

def show_pie_chart(holdings: List[Holding], prices: Dict[str, Optional[float]]) -> None:
    labels, values = [], []
    for h in holdings:
        price = prices.get(h.ticker)
        if price is None:
            continue
        labels.append(h.ticker)
        values.append(h.current_value(price))

    if not values:
        print("No price data available.")
        return

    colours = PALETTE[:len(values)]
    total   = sum(values)

    fig, ax = plt.subplots(figsize=(7, 6))

    wedges, _, autotexts = ax.pie(
        values,
        labels=None,
        colors=colours,
        autopct=lambda p: f"{p:.1f}%" if p > 4 else "",
        startangle=90,
        wedgeprops={"width": 0.55, "edgecolor": BG, "linewidth": 2},
        pctdistance=0.78,
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color(TEXT)

    # Centre label
    ax.text(0, 0.08, "Total", ha="center", va="center", fontsize=9, color=MUTED)
    ax.text(0, -0.08, f"€{total:,.0f}", ha="center", va="center",
            fontsize=14, fontweight="bold", color=TEXT)

    # Legend outside chart
    ax.legend(
        wedges, [f"{l}  €{v:,.0f}" for l, v in zip(labels, values)],
        loc="lower center", bbox_to_anchor=(0.5, -0.08),
        ncol=min(3, len(labels)), frameon=True,
    )

    ax.set_title("Allocation")
    _savefig(fig, "portfolio_allocation.png")


# ── P&L bar chart ─────────────────────────────────────────────────────────────

def show_pnl_bar_chart(holdings: List[Holding], prices: Dict[str, Optional[float]]) -> None:
    tickers, pnls, pcts = [], [], []
    for h in holdings:
        price = prices.get(h.ticker)
        if price is None:
            continue
        tickers.append(h.ticker)
        pnls.append(h.unrealised_pnl(price))
        pcts.append(h.pnl_percent(price))

    if not tickers:
        print("No price data available.")
        return

    colours = [GAIN if p >= 0 else LOSS for p in pnls]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, max(3.5, len(tickers) * 0.65 + 1.5)))

    # ── Left: absolute P&L
    bars = ax1.barh(tickers, pnls, color=colours, height=0.5, zorder=3)
    ax1.axvline(0, color=BORDER, linewidth=1, zorder=2)
    ax1.set_title("Unrealised P&L  (€)")
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    for bar, pnl in zip(bars, pnls):
        pad = max(abs(p) for p in pnls) * 0.015
        x   = bar.get_width() + pad if pnl >= 0 else bar.get_width() - pad
        ha  = "left" if pnl >= 0 else "right"
        ax1.text(x, bar.get_y() + bar.get_height() / 2,
                 f"€{pnl:+,.0f}", va="center", ha=ha, fontsize=8, color=TEXT)

    # ── Right: percentage P&L
    bars2 = ax2.barh(tickers, pcts, color=colours, height=0.5, zorder=3)
    ax2.axvline(0, color=BORDER, linewidth=1, zorder=2)
    ax2.set_title("Unrealised P&L  (%)")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.1f}%"))
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    for bar, pct in zip(bars2, pcts):
        pad = max(abs(p) for p in pcts) * 0.015
        x   = bar.get_width() + pad if pct >= 0 else bar.get_width() - pad
        ha  = "left" if pct >= 0 else "right"
        ax2.text(x, bar.get_y() + bar.get_height() / 2,
                 f"{pct:+.1f}%", va="center", ha=ha, fontsize=8, color=TEXT)

    _savefig(fig, "portfolio_pnl.png")


# ── Invested vs value chart ───────────────────────────────────────────────────

def show_value_bar_chart(holdings: List[Holding], prices: Dict[str, Optional[float]]) -> None:
    tickers, values, invested = [], [], []
    for h in holdings:
        price = prices.get(h.ticker)
        if price is None:
            continue
        tickers.append(h.ticker)
        values.append(h.current_value(price))
        invested.append(h.total_invested)

    if not tickers:
        print("No price data available.")
        return

    x     = np.arange(len(tickers))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(6, len(tickers) * 1.6), 5))

    b1 = ax.bar(x - width / 2, invested, width, label="Invested",
                color=BLUE, alpha=0.75, zorder=3)
    b2 = ax.bar(x + width / 2, values,   width, label="Current Value",
                color=[GAIN if v >= i else LOSS for v, i in zip(values, invested)],
                alpha=0.9, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"€{y:,.0f}"))
    ax.set_title("Invested vs Current Value")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()

    # Value labels on top of bars
    for bar in list(b1) + list(b2):
        h_val = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h_val + max(values) * 0.01,
                f"€{h_val:,.0f}", ha="center", va="bottom", fontsize=7.5, color=MUTED)

    _savefig(fig, "portfolio_value.png")


def show_all_charts(holdings: List[Holding], prices: Dict[str, Optional[float]]) -> None:
    show_pie_chart(holdings, prices)
    show_pnl_bar_chart(holdings, prices)
    show_value_bar_chart(holdings, prices)
