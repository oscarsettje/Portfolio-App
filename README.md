# Portfolio Tracker

A personal portfolio tracker built with Python and Streamlit. Tracks stocks, ETFs and crypto with live prices from Yahoo Finance, advanced analytics, German tax estimates, and multi-user support. Data is stored locally in a SQLite database with automatic JSON backups.

## Features

| Page | What it does |
|---|---|
| **Dashboard** | Portfolio value, P&L, best/worst holdings, allocation donut, value over time, dividend income chart, news feed |
| **Holdings** | Full position table, price history charts, editable transactions, manual price override, Excel/CSV export |
| **Add Transaction** | Record buys and sells with date, price, quantity and broker commission |
| **Benchmark** | Compare portfolio vs MSCI World, S&P 500, NASDAQ 100 and MSCI Emerging Markets |
| **Portfolio Analysis** | Correlation matrix, sector breakdown, diversification score, stress testing (preset and custom scenarios) |
| **Quant Metrics** | Sharpe, Sortino, Jensen's Alpha, Beta, VaR, CVaR, Calmar ratio, rolling performance charts |
| **Plan** | Rebalancing tool with drift analysis and trade suggestions, monthly savings plan tracker with projections, investment goal setting with scenario modelling |
| **Tax & Income** | German Abgeltungsteuer estimates (25% + 5.5% Soli), FIFO cost basis, Sparerpauschbetrag (€1,000), dividend tracking with withholding tax |
| **Snapshot History** | Manually record portfolio value over time to track growth |

## Setup

**Requirements:** Python 3.10+

```bash
# 1. Clone the repo
git clone https://github.com/oscarsettje/Portfolio-App.git
cd Portfolio-App

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

Opens at `http://localhost:8501`. The database (`portfolio.db`) is created automatically on first launch.

## First Launch

When you first open the app you'll see a login screen. Enter a name to create your account — no password needed. Each user has completely isolated data: their own holdings, transactions, dividends, and snapshots.

To add a second user, just open the app and create a new account from the login screen. Users can switch accounts at any time using the **Switch User** button at the bottom of the sidebar.

> **Existing users upgrading from v1:** Your data is automatically migrated to a user called "Default" on first launch. Nothing is lost.

## Ticker Formats

| Asset | Format | Examples |
|---|---|---|
| US stocks / ETFs | Plain ticker | `AAPL`, `MSFT`, `SPY`, `VTI` |
| German stocks | `TICKER.DE` | `SIE.DE`, `BMW.DE` |
| Dutch stocks | `TICKER.AS` | `ASML.AS` |
| French stocks | `TICKER.PA` | `MC.PA` |
| UK stocks | `TICKER.L` | `SHEL.L` |
| Crypto | `TICKER-USD` | `BTC-USD`, `ETH-USD` |
| ETFs (XETRA) | `TICKER.DE` | `IWDA.AS`, `VWCE.DE` |

If a ticker isn't supported by Yahoo Finance, use the **manual price override** in the Holdings page.

## Export

The Holdings page has two export options:

- **Excel (.xlsx)** — 5 sheets: Summary, Transactions (with commissions), Dividends (with per-year subtotals), FIFO Gains, and a German Tax Summary with live formulas
- **CSV** — flat summary of current positions

## Data & Backups

All data lives in `portfolio.db` (SQLite). After every write, the app also writes per-user JSON backups:

- `portfolio_data_<username>.json` — holdings and transactions
- `portfolio_snapshots_<username>.json` — snapshot history

Both JSON files and `portfolio.db` are gitignored by default. To back up your data, copy these files to a safe location.

## Project Structure

```
portfolio_tracker/
├── app.py                    # Streamlit UI — all pages, charts, login screen
├── requirements.txt
├── portfolio.db              # SQLite database (auto-created, gitignored)
└── tracker/
    ├── db.py                 # SQLite layer — multi-user schema, migrations
    ├── portfolio.py          # Portfolio state scoped to a single user
    ├── models.py             # Dataclasses: Holding, Transaction, Dividend, Snapshot
    ├── prices.py             # Live price fetching via yfinance + DB-backed cache
    ├── benchmark.py          # Portfolio vs index performance comparison
    ├── analysis.py           # Correlation, sector breakdown, stress testing
    ├── quant.py              # Sharpe, Sortino, Alpha, Beta, VaR, rolling metrics
    ├── tax.py                # German Abgeltungsteuer — FIFO, Sparerpauschbetrag
    ├── planner.py            # Rebalancing, savings plan, investment goal calculations
    ├── exporter.py           # Excel (5 sheets) and CSV export
    └── validation.py         # Input validation for tickers, transactions, dividends
```

## Database Schema

```
users         id, username, created_at
holdings      id, user_id, ticker, name, asset_type, manual_price
transactions  id, user_id, ticker, date, action, quantity, price, commission
dividends     id, user_id, ticker, date, amount, withholding_tax
snapshots     id, user_id, date, total_value, total_invested, note
price_cache   ticker, price, updated_at   ← shared across all users
```

## Rate Limiting

Yahoo Finance enforces rate limits. The app caches prices in the database so they persist across restarts. If you see a warning about cached prices, wait a few minutes before refreshing. The Benchmark, Quant, and Portfolio Analysis pages cache their downloaded data per session — use the **Re-fetch** button to force a fresh download.

## Dependencies

```
streamlit   yfinance   plotly   pandas   numpy   openpyxl   scipy
```
