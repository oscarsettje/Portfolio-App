# 📈 Portfolio Tracker

A personal portfolio tracker built with Python and Streamlit. Tracks stocks, ETFs and crypto with live prices, performance charts, and advanced portfolio analytics. Data is stored in a local SQLite database with automatic JSON backups.

## Features

| Page | What it does |
|---|---|
| **Dashboard** | Overview metrics, allocation chart, P&L bars, news feed |
| **Holdings** | Full position table, price history charts, editable transactions, Excel/CSV export |
| **Add Transaction** | Record buys and sells for any stock, ETF or crypto |
| **Benchmark** | Compare portfolio vs MSCI World, S&P 500, NASDAQ, EM indices |
| **Portfolio Analysis** | Diversification breakdown, correlation matrix, stress testing |
| **Quant Metrics** | Sharpe, Sortino, Jensen's Alpha, Beta, VaR, CVaR, Calmar, rolling charts |
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

## Project Structure

```
portfolio_tracker/
├── app.py                       # Streamlit UI — all pages and charts
├── requirements.txt
├── portfolio.db                 # SQLite database (auto-created, gitignored)
├── portfolio_data.json          # JSON backup — auto-synced after every write
├── portfolio_snapshots.json     # Snapshot backup — auto-synced
└── tracker/
    ├── db.py                    # SQLite layer — all reads/writes
    ├── portfolio.py             # Data model: Portfolio, Holding, Transaction, Snapshot
    ├── prices.py                # Live price fetching + DB-backed cache
    ├── benchmark.py             # Portfolio vs index performance
    ├── analysis.py              # Diversification, correlation, stress testing
    ├── quant.py                 # Sharpe, Sortino, Alpha, Beta, VaR, rolling metrics
    └── exporter.py              # Excel and CSV export
```

## Database Schema

```sql
holdings      ticker, name, asset_type, manual_price
transactions  id, ticker → holdings, date, action, buy/sell, quantity, price
snapshots     id, date, total_value, total_invested, note
price_cache   ticker, price, updated_at
```

## Data & Backups

All data lives in `portfolio.db` (SQLite). After every write, the app also updates:
- `portfolio_data.json` — holdings and transactions in human-readable form
- `portfolio_snapshots.json` — snapshot history

Both JSON files and `portfolio.db` are gitignored by default. To back up your data, copy these three files.

## Ticker Formats

| Asset | Format | Examples |
|---|---|---|
| US stocks / ETFs | Plain ticker | `AAPL`, `MSFT`, `SPY` |
| German stocks | `TICKER.DE` | `SIE.DE`, `BMW.DE` |
| Dutch stocks | `TICKER.AS` | `ASML.AS` |
| French stocks | `TICKER.PA` | `MC.PA` |
| UK stocks | `TICKER.L` | `SHEL.L` |
| Crypto | `TICKER-USD` | `BTC-USD`, `ETH-USD` |

If a ticker is not supported by Yahoo Finance, use the manual price override in Holdings.

## Dependencies

```
streamlit   yfinance   plotly   pandas   numpy   openpyxl
```
