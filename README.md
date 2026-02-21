# 📊 Portfolio Tracker

A clean, educational Python portfolio tracker for Stocks, Crypto, and ETFs —
with live prices, terminal tables, charts, and Excel/CSV export.

---

## Features

- **Live prices** via Yahoo Finance (`yfinance`) — no API key needed
- **Terminal dashboard** with colour-coded P&L (powered by `rich`)
- **3 chart types**: allocation donut, P&L bar, invested vs value
- **Excel export** with professional formatting, colour coding, and formulas
- **CSV export** for use in Google Sheets or other tools
- **Persistent storage** — your data is saved to `portfolio_data.json` automatically
- **Tracks**: Stocks, ETFs, and Crypto (anything Yahoo Finance supports)

---

## Project Structure

```
portfolio_tracker/
│
├── main.py                  ← Run this to start the app
├── requirements.txt         ← Python library dependencies
├── portfolio_data.json      ← Auto-created; stores your holdings
│
└── tracker/
    ├── __init__.py
    ├── portfolio.py         ← Data model: Holdings, Transactions (dataclasses, JSON)
    ├── prices.py            ← Live price fetching + caching (yfinance)
    ├── display.py           ← Terminal output (rich tables, colours)
    ├── charts.py            ← Charts (matplotlib)
    ├── exporter.py          ← Excel and CSV export (openpyxl, csv)
    └── cli.py               ← Interactive menu loop (orchestrates everything)
```

---

## Setup

### 1. Install Python 3.10+
Download from https://www.python.org/downloads/

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Activate on Mac/Linux:
source venv/bin/activate

# Activate on Windows:
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the tracker
```bash
python main.py
```

---

## Usage

The app presents a numbered menu:

```
Portfolio Tracker
─────────────────────────────────
  1  View portfolio
  2  Add BUY transaction
  3  Add SELL transaction
  4  View holding detail
  5  Show charts
  6  Export (Excel / CSV)
  7  Remove a holding
  8  Refresh live prices
  q  Quit
```

### Adding your first holding
1. Press `2` (Add BUY)
2. Enter the **ticker symbol** — e.g.:
   - `AAPL` (Apple stock)
   - `BTC-USD` (Bitcoin in USD)
   - `SPY` (S&P 500 ETF)
   - `ETH-USD` (Ethereum)
3. Enter the **full name**, **asset type**, **quantity**, **price paid**, and **date**

Your data is saved automatically.

### Ticker formats
| Asset        | Yahoo Finance ticker |
|--------------|----------------------|
| Apple        | `AAPL`               |
| Microsoft    | `MSFT`               |
| Bitcoin      | `BTC-USD`            |
| Ethereum     | `ETH-USD`            |
| S&P 500 ETF  | `SPY`                |
| Nasdaq ETF   | `QQQ`                |
| Gold ETF     | `GLD`                |

You can look up any ticker at https://finance.yahoo.com

---

## Python Concepts Used (Learning Reference)

| File             | Concepts                                               |
|------------------|--------------------------------------------------------|
| `portfolio.py`   | `@dataclass`, `@property`, JSON, type hints            |
| `prices.py`      | External libraries, caching, exception handling         |
| `display.py`     | f-strings, rich markup, functions, separation of concerns |
| `charts.py`      | matplotlib, subplots, list comprehensions               |
| `exporter.py`    | openpyxl, CSV, context managers (`with` statement)     |
| `cli.py`         | `while` loops, `input()`, `try/except`, class methods  |

---

## Extending the project (ideas for next steps)

- Add **price history charts** using `yfinance`'s `.history()` method
- Track **dividends** by adding a `dividend` transaction type
- Add a **target allocation** feature to show drift from your desired weights
- Build a **web UI** using Flask or Streamlit
- Connect to a **database** (SQLite) instead of JSON for larger portfolios
