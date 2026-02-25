# 📈 Portfolio Tracker

A personal portfolio tracker built with Python and Streamlit. Track stocks, ETFs and crypto with live prices, performance charts, and portfolio analysis.

## Features

| Page | What it does |
|---|---|
| **Dashboard** | Overview metrics, allocation chart, P&L bars, news feed |
| **Holdings** | Full position table, price history charts, editable transactions, Excel/CSV export |
| **Add Transaction** | Record buys and sells for any stock, ETF or crypto |
| **Benchmark** | Compare portfolio performance vs MSCI World, S&P 500, NASDAQ, EM indices |
| **Portfolio Analysis** | Diversification breakdown, correlation matrix, stress testing |
| **Snapshot History** | Manually record portfolio value over time to track growth |

## Setup

**Requirements:** Python 3.10+

```bash
# 1. Clone the repo
git clone https://github.com/oscarsettje/Portfolio-App.git
cd Portfolio-App

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Project Structure

```
portfolio_tracker/
├── app.py                      # Streamlit UI — all pages and charts
├── requirements.txt
├── portfolio_data.json         # Your holdings (auto-created, gitignored)
├── portfolio_snapshots.json    # Snapshot history (auto-created, gitignored)
└── tracker/
    ├── portfolio.py            # Data model: Portfolio, Holding, Transaction, Snapshot
    ├── prices.py               # Live price fetching via Yahoo Finance
    ├── benchmark.py            # Portfolio vs index performance calculation
    ├── analysis.py             # Diversification, correlation, stress testing
    └── exporter.py             # Excel and CSV export
```

## Ticker Formats

| Asset | Format | Examples |
|---|---|---|
| US stocks / ETFs | Plain ticker | `AAPL`, `MSFT`, `SPY`, `QQQ` |
| German stocks | `TICKER.DE` | `SIE.DE`, `BMW.DE` |
| Dutch stocks | `TICKER.AS` | `ASML.AS` |
| French stocks | `TICKER.PA` | `MC.PA`, `TTE.PA` |
| UK stocks | `TICKER.L` | `SHEL.L`, `VOD.L` |
| Crypto | `TICKER-USD` | `BTC-USD`, `ETH-USD`, `SOL-USD` |

If a ticker isn't supported by Yahoo Finance, use the **manual price override** in the Holdings page.

## Data Storage

All data is saved locally as JSON files. Nothing is sent to any server.

- `portfolio_data.json` — your transactions and manual price overrides
- `portfolio_snapshots.json` — your saved portfolio snapshots

Both files are gitignored by default.

## Dependencies

```
streamlit
yfinance
plotly
pandas
numpy
openpyxl
```
