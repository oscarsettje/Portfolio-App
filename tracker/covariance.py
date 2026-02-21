"""
tracker/covariance.py
======================
Computes and visualises the covariance (and correlation) matrix
across portfolio holdings using weekly returns over 3 years.

Key concepts introduced here:
  - pandas DataFrame    : 2D table of data, perfect for time series
  - pct_change()        : computing returns from price series
  - .cov() / .corr()    : built-in pandas matrix calculations
  - numpy               : numerical operations on arrays
  - seaborn-style heatmap via matplotlib : annotated colour grids
  - resample()          : resampling daily data to weekly frequency
"""

from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import yfinance as yf

from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

# 3 years of weekly data
PERIOD = "3y"
INTERVAL = "1wk"


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_weekly_returns(tickers: List[str]) -> Optional[pd.DataFrame]:
    """
    Download 3 years of weekly closing prices for all tickers
    and return a DataFrame of weekly percentage returns.

    Returns None if data could not be fetched.

    Concepts:
      - yf.download()  : bulk historical data download
      - .pct_change()  : converts prices to % returns row-by-row
      - .dropna()      : removes rows where any ticker has missing data
    """
    console.print(f"[dim]Downloading {PERIOD} of weekly price history for {len(tickers)} ticker(s)...[/dim]")

    try:
        raw = yf.download(
            tickers,
            period=PERIOD,
            interval=INTERVAL,
            progress=False,
            auto_adjust=True,
        )
    except Exception as e:
        console.print(f"[red]Failed to download data: {e}[/red]")
        return None

    if raw.empty:
        console.print("[red]No historical data returned.[/red]")
        return None

    # Extract closing prices — structure differs for single vs multiple tickers
    if len(tickers) == 1:
        prices = raw[["Close"]].copy()
        prices.columns = [tickers[0].upper()]
    else:
        prices = raw["Close"].copy()
        prices.columns = [c.upper() for c in prices.columns]

    # Weekly returns: (price_this_week - price_last_week) / price_last_week
    returns = prices.pct_change().dropna()

    if returns.empty:
        console.print("[red]Not enough data to compute returns.[/red]")
        return None

    return returns


# ---------------------------------------------------------------------------
# Matrix computation
# ---------------------------------------------------------------------------

def compute_matrices(returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute the covariance matrix and correlation matrix from weekly returns.

    Covariance  : measures how two assets move together (in return² units)
    Correlation : normalised version — always between -1 and +1, easier to read

    Returns (covariance_matrix, correlation_matrix)
    """
    # pandas does all the heavy lifting here
    cov_matrix  = returns.cov()
    corr_matrix = returns.corr()
    return cov_matrix, corr_matrix


# ---------------------------------------------------------------------------
# Terminal display
# ---------------------------------------------------------------------------

def print_covariance_matrix(cov_matrix: pd.DataFrame, returns: pd.DataFrame) -> None:
    """Print the covariance matrix and summary stats as a rich terminal table."""

    tickers = list(cov_matrix.columns)
    n = len(tickers)

    # ---- Covariance matrix table ----
    table = Table(
        title=f"📐  Covariance Matrix  ({PERIOD} weekly returns)",
        box=box.ROUNDED,
        header_style="bold cyan",
        border_style="cyan",
    )
    table.add_column("", style="bold cyan", min_width=10)  # Row label column
    for t in tickers:
        table.add_column(t, justify="right", min_width=12)

    for row_ticker in tickers:
        row_values = []
        for col_ticker in tickers:
            val = cov_matrix.loc[row_ticker, col_ticker]
            # Diagonal (variance) shown differently
            if row_ticker == col_ticker:
                row_values.append(f"[bold yellow]{val:.6f}[/bold yellow]")
            else:
                colour = "green" if val > 0 else "red"
                row_values.append(f"[{colour}]{val:.6f}[/{colour}]")
        table.add_row(row_ticker, *row_values)

    console.print(table)

    # ---- Correlation matrix table ----
    corr_matrix = returns.corr()
    corr_table = Table(
        title="🔗  Correlation Matrix  (−1 = inverse  |  0 = none  |  +1 = perfect)",
        box=box.ROUNDED,
        header_style="bold cyan",
        border_style="cyan",
    )
    corr_table.add_column("", style="bold cyan", min_width=10)
    for t in tickers:
        corr_table.add_column(t, justify="right", min_width=10)

    for row_ticker in tickers:
        row_values = []
        for col_ticker in tickers:
            val = corr_matrix.loc[row_ticker, col_ticker]
            if row_ticker == col_ticker:
                row_values.append(f"[bold yellow]{val:.4f}[/bold yellow]")
            elif val > 0.7:
                row_values.append(f"[bold green]{val:.4f}[/bold green]")
            elif val > 0.3:
                row_values.append(f"[green]{val:.4f}[/green]")
            elif val < -0.3:
                row_values.append(f"[red]{val:.4f}[/red]")
            else:
                row_values.append(f"{val:.4f}")
        corr_table.add_row(row_ticker, *row_values)

    console.print(corr_table)

    # ---- Per-ticker stats ----
    stats_table = Table(
        title="📊  Individual Return Statistics  (weekly)",
        box=box.SIMPLE_HEAD,
        header_style="bold cyan",
    )
    stats_table.add_column("Ticker", style="bold white")
    stats_table.add_column("Avg Weekly Return", justify="right")
    stats_table.add_column("Weekly Volatility (σ)", justify="right")
    stats_table.add_column("Ann. Volatility (σ×√52)", justify="right")
    stats_table.add_column("Weeks of Data", justify="right")

    for ticker in tickers:
        col = returns[ticker].dropna()
        avg    = col.mean()
        weekly_std = col.std()
        annual_std = weekly_std * np.sqrt(52)   # Annualise weekly vol

        avg_str = f"[green]+{avg:.4%}[/green]" if avg >= 0 else f"[red]{avg:.4%}[/red]"
        stats_table.add_row(
            ticker,
            avg_str,
            f"{weekly_std:.4%}",
            f"{annual_std:.2%}",
            str(len(col)),
        )

    console.print(stats_table)


# ---------------------------------------------------------------------------
# Heatmap chart
# ---------------------------------------------------------------------------

def show_heatmap(cov_matrix: pd.DataFrame, corr_matrix: pd.DataFrame) -> None:
    """
    Render a side-by-side heatmap of the covariance and correlation matrices.

    Concepts:
      - imshow()     : displays a 2D array as a colour image
      - colorbar()   : adds a colour scale legend
      - annotate()   : writes the numeric value inside each cell
    """

    tickers = list(cov_matrix.columns)
    n = len(tickers)

    fig, axes = plt.subplots(1, 2, figsize=(max(10, n * 2.5 + 4), max(5, n * 1.8 + 2)))
    fig.patch.set_facecolor("#1a1a2e")

    _draw_heatmap(
        ax=axes[0],
        matrix=cov_matrix,
        tickers=tickers,
        title="Covariance Matrix\n(weekly returns)",
        fmt=".6f",
        cmap="RdYlGn",
        centre=0,
    )

    _draw_heatmap(
        ax=axes[1],
        matrix=corr_matrix,
        tickers=tickers,
        title="Correlation Matrix\n(−1 to +1)",
        fmt=".3f",
        cmap="RdYlGn",
        centre=0,
        vmin=-1,
        vmax=1,
    )

    fig.suptitle(
        f"Portfolio Risk Analysis  ·  3-Year Weekly Returns",
        color="white", fontsize=14, fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    fname = "portfolio_covariance.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    console.print(f"[green]  Chart saved: {fname}[/green]")


def _draw_heatmap(ax, matrix: pd.DataFrame, tickers: List[str],
                  title: str, fmt: str, cmap: str, centre: float,
                  vmin: Optional[float] = None, vmax: Optional[float] = None) -> None:
    """Helper that draws a single annotated heatmap on a given Axes object."""

    data = matrix.values.astype(float)
    n = len(tickers)

    # Determine colour scale bounds
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()

    # Create a diverging normaliser centred on `centre`
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=centre, vmax=vmax)

    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    # Colour bar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)

    # Axis labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tickers, color="white", fontsize=10, rotation=45, ha="right")
    ax.set_yticklabels(tickers, color="white", fontsize=10)
    ax.set_facecolor("#1a1a2e")
    ax.set_title(title, color="white", fontsize=11, pad=10)

    # Annotate each cell with its numeric value
    for i in range(n):
        for j in range(n):
            val = data[i, j]
            # Pick text colour based on background brightness
            bg_norm = (val - vmin) / (vmax - vmin) if vmax != vmin else 0.5
            text_colour = "black" if 0.3 < bg_norm < 0.7 else "white"
            ax.text(
                j, i,
                format(val, fmt),
                ha="center", va="center",
                fontsize=max(7, 10 - n),  # Shrink font for large matrices
                color=text_colour,
                fontweight="bold" if i == j else "normal",
            )

    # Draw grid lines between cells
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="#1a1a2e", linewidth=2)
    ax.tick_params(which="minor", size=0)


# ---------------------------------------------------------------------------
# Main entry point called from CLI
# ---------------------------------------------------------------------------

def run_covariance_analysis(tickers: List[str]) -> None:
    """Orchestrates the full covariance analysis flow."""

    if len(tickers) < 2:
        console.print("[red]Please select at least 2 tickers for a covariance matrix.[/red]")
        return

    returns = fetch_weekly_returns(tickers)
    if returns is None:
        return

    # Drop any tickers that came back with all NaN (e.g. invalid symbols)
    returns = returns.dropna(axis=1, how="all")
    valid_tickers = list(returns.columns)

    if len(valid_tickers) < 2:
        console.print("[red]Not enough valid tickers with data to compute a matrix.[/red]")
        return

    if len(valid_tickers) < len(tickers):
        dropped = set(t.upper() for t in tickers) - set(valid_tickers)
        console.print(f"[yellow]Warning: no data found for {', '.join(dropped)} — excluded.[/yellow]")

    cov_matrix, corr_matrix = compute_matrices(returns)

    console.print()
    print_covariance_matrix(cov_matrix, returns)
    console.print()
    show_heatmap(cov_matrix, corr_matrix)
