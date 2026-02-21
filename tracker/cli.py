"""
tracker/cli.py
==============
The interactive command-line interface.

Key concepts introduced here:
  - while loops       : keeping a program running until the user exits
  - input()           : reading text from the user
  - try/except        : catching bad input (e.g. user types letters for a number)
  - Modular design    : this file orchestrates the other modules
"""

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

from tracker.portfolio import Portfolio
from tracker.prices import PriceFetcher
from tracker import display, charts, exporter, covariance

console = Console()


class CLI:
    """Main command-line interface class."""

    ASSET_TYPES = {"1": "stock", "2": "crypto", "3": "etf"}

    def __init__(self):
        self.portfolio = Portfolio()
        self.fetcher = PriceFetcher()

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _get_prices(self):
        """Fetch prices for all holdings in the portfolio."""
        tickers = [h.ticker for h in self.portfolio.all_holdings()]
        if not tickers:
            return {}
        console.print("[dim]Fetching live prices...[/dim]")
        return self.fetcher.get_prices(tickers)

    def _prompt_float(self, prompt: str) -> float:
        """Keep asking until the user enters a valid positive number."""
        while True:
            raw = Prompt.ask(prompt)
            try:
                value = float(raw)
                if value <= 0:
                    console.print("[red]Please enter a positive number.[/red]")
                    continue
                return value
            except ValueError:
                console.print("[red]That doesn't look like a number. Try again.[/red]")

    def _prompt_date(self) -> str:
        """Ask for a date, defaulting to today."""
        from datetime import date
        default = date.today().strftime("%Y-%m-%d")
        raw = Prompt.ask(f"Date (YYYY-MM-DD)", default=default)
        return raw

    # -----------------------------------------------------------------------
    # Menu actions
    # -----------------------------------------------------------------------

    def view_portfolio(self):
        holdings = self.portfolio.all_holdings()
        prices = self._get_prices()
        console.print()
        display.print_portfolio_summary(holdings, prices)
        console.print()
        display.print_allocation_breakdown(holdings, prices)

    def add_transaction(self, action: str = "buy"):
        """Guided flow to add a buy or sell transaction."""
        console.print(f"\n[steel_blue1]── Add {action.upper()} Transaction ──[/steel_blue1]")

        ticker = Prompt.ask("Ticker symbol (e.g. AAPL, BTC-USD, SPY)").upper()

        # If it's a new holding we need more info
        existing = self.portfolio.get_holding(ticker)
        if existing:
            name = existing.name
            asset_type = existing.asset_type
        else:
            name = Prompt.ask("Full name (e.g. Apple Inc.)")
            console.print("  Asset type:  1 = Stock   2 = Crypto   3 = ETF")
            type_choice = Prompt.ask("Choose", choices=["1", "2", "3"])
            asset_type = self.ASSET_TYPES[type_choice]

        quantity = self._prompt_float("Quantity")
        price = self._prompt_float("Price per unit ($)")
        date = self._prompt_date()

        self.portfolio.add_transaction(
            ticker=ticker,
            name=name,
            asset_type=asset_type,
            action=action,
            quantity=quantity,
            price=price,
            date=date,
        )
        console.print(f"[green]✓ {action.upper()} recorded for {ticker}[/green]")

    def view_holding_detail(self):
        holdings = self.portfolio.all_holdings()
        if not holdings:
            console.print("[yellow]No holdings found.[/yellow]")
            return

        console.print("\nAvailable tickers: " +
                      ", ".join(f"[cyan]{h.ticker}[/cyan]" for h in holdings))
        ticker = Prompt.ask("Enter ticker").upper()
        holding = self.portfolio.get_holding(ticker)

        if not holding:
            console.print(f"[red]Ticker '{ticker}' not found.[/red]")
            return

        price = self.fetcher.get_price(ticker)
        display.print_holding_detail(holding, price)

    def remove_holding(self):
        ticker = Prompt.ask("Enter ticker to remove").upper()
        if Confirm.ask(f"[red]Delete ALL data for {ticker}? This cannot be undone.[/red]"):
            if self.portfolio.remove_holding(ticker):
                console.print(f"[green]✓ {ticker} removed.[/green]")
            else:
                console.print(f"[red]'{ticker}' not found.[/red]")

    def show_charts(self):
        holdings = self.portfolio.all_holdings()
        if not holdings:
            console.print("[yellow]No holdings to chart.[/yellow]")
            return

        prices = self._get_prices()
        console.print("\n[steel_blue1]── Charts ──[/steel_blue1]")
        console.print("  1. Allocation (donut)")
        console.print("  2. Unrealised P&L (bar)")
        console.print("  3. Invested vs Current Value (bar)")
        console.print("  4. All charts")
        choice = Prompt.ask("Choose", choices=["1", "2", "3", "4"])

        if choice == "1":
            charts.show_pie_chart(holdings, prices)
        elif choice == "2":
            charts.show_pnl_bar_chart(holdings, prices)
        elif choice == "3":
            charts.show_value_bar_chart(holdings, prices)
        elif choice == "4":
            charts.show_all_charts(holdings, prices)

    def export_data(self):
        holdings = self.portfolio.all_holdings()
        if not holdings:
            console.print("[yellow]No holdings to export.[/yellow]")
            return

        prices = self._get_prices()
        console.print("\n  1. Export to Excel (.xlsx)")
        console.print("  2. Export to CSV")
        console.print("  3. Both")
        choice = Prompt.ask("Choose", choices=["1", "2", "3"])

        if choice in ("1", "3"):
            fname = exporter.export_to_excel(holdings, prices)
            console.print(f"[green]✓ Excel saved: {fname}[/green]")
        if choice in ("2", "3"):
            fname = exporter.export_to_csv(holdings, prices)
            console.print(f"[green]✓ CSV saved:   {fname}[/green]")

    def show_covariance(self):
        """Let the user pick tickers and run the covariance analysis."""
        holdings = self.portfolio.all_holdings()

        if len(holdings) < 2:
            console.print("[yellow]You need at least 2 holdings to compute a covariance matrix.[/yellow]")
            return

        available = [h.ticker for h in holdings]
        console.print("\n[steel_blue1]── Covariance & Correlation Matrix ──[/steel_blue1]")
        console.print("Available tickers: " + "  ".join(f"[cyan]{t}[/cyan]" for t in available))
        console.print("[dim]Enter tickers separated by spaces, or press Enter to use all.[/dim]")

        raw = Prompt.ask("Tickers", default=" ".join(available))
        selected = [t.strip().upper() for t in raw.split() if t.strip()]

        # Validate selection against portfolio
        invalid = [t for t in selected if t not in available]
        if invalid:
            console.print(f"[yellow]Note: {', '.join(invalid)} not in your portfolio — will still attempt to fetch.[/yellow]")

        if len(selected) < 2:
            console.print("[red]Please enter at least 2 tickers.[/red]")
            return

        covariance.run_covariance_analysis(selected)

    def refresh_prices(self):
        self.fetcher.clear_cache()
        console.print("[green]✓ Price cache cleared. Prices will refresh on next view.[/green]")

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    MENU = """
[grey39]┌─────────────────────────────────┐[/grey39]
[grey39]│[/grey39]  [steel_blue1]Portfolio Tracker[/steel_blue1]               [grey39]│[/grey39]
[grey39]├─────────────────────────────────┤[/grey39]
[grey39]│[/grey39]  [white]1[/white]  [grey62]View portfolio[/grey62]              [grey39]│[/grey39]
[grey39]│[/grey39]  [white]2[/white]  [grey62]Add BUY transaction[/grey62]         [grey39]│[/grey39]
[grey39]│[/grey39]  [white]3[/white]  [grey62]Add SELL transaction[/grey62]        [grey39]│[/grey39]
[grey39]│[/grey39]  [white]4[/white]  [grey62]View holding detail[/grey62]         [grey39]│[/grey39]
[grey39]│[/grey39]  [white]5[/white]  [grey62]Charts[/grey62]                      [grey39]│[/grey39]
[grey39]│[/grey39]  [white]6[/white]  [grey62]Export  (Excel / CSV)[/grey62]       [grey39]│[/grey39]
[grey39]│[/grey39]  [white]7[/white]  [grey62]Remove a holding[/grey62]            [grey39]│[/grey39]
[grey39]│[/grey39]  [white]8[/white]  [grey62]Refresh prices[/grey62]              [grey39]│[/grey39]
[grey39]│[/grey39]  [white]9[/white]  [grey62]Covariance matrix[/grey62]           [grey39]│[/grey39]
[grey39]│[/grey39]  [white]q[/white]  [grey62]Quit[/grey62]                        [grey39]│[/grey39]
[grey39]└─────────────────────────────────┘[/grey39]"""

    def run(self):
        console.print(Panel(
            "[bold white]Portfolio Tracker[/bold white]  [grey62]by Yahoo Finance · yfinance[/grey62]",
            border_style="grey39",
            padding=(0, 2),
        ))

        while True:
            console.print(self.MENU)
            choice = Prompt.ask("Choice", default="1").strip().lower()

            if choice == "q":
                console.print("[cyan]Goodbye! 👋[/cyan]")
                break
            elif choice == "1":
                self.view_portfolio()
            elif choice == "2":
                self.add_transaction("buy")
            elif choice == "3":
                self.add_transaction("sell")
            elif choice == "4":
                self.view_holding_detail()
            elif choice == "5":
                self.show_charts()
            elif choice == "6":
                self.export_data()
            elif choice == "7":
                self.remove_holding()
            elif choice == "8":
                self.refresh_prices()
            elif choice == "9":
                self.show_covariance()
            else:
                console.print("[red]Invalid choice. Please try again.[/red]")
