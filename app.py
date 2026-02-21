"""
app.py
======
Streamlit web interface for the Portfolio Tracker.

Run with:  streamlit run app.py

Key concepts introduced here:
  - streamlit         : turns Python scripts into interactive web apps
  - st.session_state  : persists data between user interactions (like a page reload)
  - st.cache_data     : caches expensive function results (e.g. API calls)
  - Plotly            : interactive charts that work natively in Streamlit
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, datetime
from typing import Dict, List, Optional

from tracker.portfolio import Portfolio, Holding
from tracker.prices import PriceFetcher

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Portfolio Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme colours ─────────────────────────────────────────────────────────────
GAIN  = "#4caf7d"
LOSS  = "#e05c5c"
BLUE  = "#5b9bd5"
MUTED = "#888888"
BG    = "#0f0f0f"

PALETTE = ["#5b9bd5", "#4caf7d", "#e8a838", "#b07fd4",
           "#e05c5c", "#4db6ac", "#f06292", "#a1887f"]

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Tighten up metric cards */
  [data-testid="metric-container"] {
      background: #1a1a1a;
      border: 1px solid #2a2a2a;
      border-radius: 8px;
      padding: 14px 18px;
  }
  [data-testid="stMetricValue"]  { font-size: 1.35rem; }
  [data-testid="stMetricDelta"]  { font-size: 0.85rem; }

  /* Table styling */
  [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

  /* Sidebar */
  [data-testid="stSidebar"] { background: #111111; }

  /* Remove default top padding */
  .block-container { padding-top: 1.5rem; }

  /* Section divider */
  .section-title {
      font-size: 0.75rem;
      font-weight: 600;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: #555;
      margin: 1.5rem 0 0.5rem 0;
  }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────────
# st.session_state persists values across reruns (when the user clicks a button).
if "portfolio" not in st.session_state:
    st.session_state.portfolio = Portfolio()

if "fetcher" not in st.session_state:
    st.session_state.fetcher = PriceFetcher()

if "prices" not in st.session_state:
    st.session_state.prices = {}


def portfolio() -> Portfolio:
    return st.session_state.portfolio

def fetcher() -> PriceFetcher:
    return st.session_state.fetcher


# ── Price fetching (cached per session) ───────────────────────────────────────

def fetch_prices() -> Dict[str, Optional[float]]:
    """Fetch live prices for all holdings and store in session state."""
    tickers = [h.ticker for h in portfolio().all_holdings()]
    if not tickers:
        return {}
    with st.spinner("Fetching live prices…"):
        prices = fetcher().get_prices(tickers)
    st.session_state.prices = prices
    return prices

def get_prices() -> Dict[str, Optional[float]]:
    if not st.session_state.prices:
        return fetch_prices()
    return st.session_state.prices


# ── Formatters ────────────────────────────────────────────────────────────────

def fmt_cur(v: float) -> str:
    return f"€{v:,.2f}"

def fmt_pct(v: float) -> str:
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.2f}%"

def colour_val(v: float, text: str) -> str:
    """Wrap text in green/red HTML span."""
    c = GAIN if v > 0 else (LOSS if v < 0 else MUTED)
    return f'<span style="color:{c}">{text}</span>'


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("## 📈 Portfolio Tracker")
        st.divider()

        # Navigation
        page = st.radio(
            "Navigation",
            ["Dashboard", "Holdings", "Add Transaction", "Covariance Matrix"],
            label_visibility="collapsed",
        )

        st.divider()

        # Quick stats in sidebar
        holdings = portfolio().all_holdings()
        prices   = get_prices()

        total_value    = sum(h.current_value(prices[h.ticker]) for h in holdings if prices.get(h.ticker))
        total_invested = sum(h.total_invested for h in holdings)
        total_pnl      = total_value - total_invested

        st.metric("Portfolio Value", fmt_cur(total_value))
        st.metric(
            "Unrealised P&L",
            fmt_cur(total_pnl),
            delta=fmt_pct((total_pnl / total_invested * 100) if total_invested else 0),
        )

        st.divider()

        if st.button("🔄  Refresh Prices", use_container_width=True):
            fetcher().clear_cache()
            st.session_state.prices = {}
            st.rerun()

    return page


# ── Dashboard page ────────────────────────────────────────────────────────────

def render_dashboard():
    st.markdown("## Dashboard")

    holdings = portfolio().all_holdings()
    prices   = get_prices()

    if not holdings:
        st.info("Your portfolio is empty. Go to **Add Transaction** to get started.")
        return

    # ── Top metric cards ──
    total_value    = sum(h.current_value(prices[h.ticker]) for h in holdings if prices.get(h.ticker))
    total_invested = sum(h.total_invested for h in holdings)
    total_pnl      = total_value - total_invested
    overall_pct    = (total_pnl / total_invested * 100) if total_invested else 0
    n_holdings     = len(holdings)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Invested",  fmt_cur(total_invested))
    c2.metric("Portfolio Value", fmt_cur(total_value))
    c3.metric("Unrealised P&L",  fmt_cur(total_pnl), delta=fmt_pct(overall_pct))
    c4.metric("Holdings",        str(n_holdings))

    st.divider()

    # ── Charts row ──
    col_left, col_right = st.columns([1, 1])

    with col_left:
        _render_allocation_chart(holdings, prices)

    with col_right:
        _render_pnl_chart(holdings, prices)

    st.divider()
    _render_value_chart(holdings, prices)


def _render_allocation_chart(holdings, prices):
    labels, values, colours = [], [], []
    for i, h in enumerate(holdings):
        p = prices.get(h.ticker)
        if p is None:
            continue
        labels.append(h.ticker)
        values.append(h.current_value(p))
        colours.append(PALETTE[i % len(PALETTE)])

    if not values:
        return

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colours, line=dict(color="#0f0f0f", width=2)),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>€%{value:,.2f}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        title="Allocation",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font_color="#cccccc",
        showlegend=False,
        margin=dict(t=40, b=10, l=10, r=10),
        height=320,
        annotations=[dict(
            text=f"€{sum(values):,.0f}",
            x=0.5, y=0.5, font_size=16, showarrow=False,
            font_color="#cccccc",
        )],
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_pnl_chart(holdings, prices):
    tickers, pnls = [], []
    for h in holdings:
        p = prices.get(h.ticker)
        if p is None:
            continue
        tickers.append(h.ticker)
        pnls.append(h.unrealised_pnl(p))

    if not tickers:
        return

    colours = [GAIN if p >= 0 else LOSS for p in pnls]
    fig = go.Figure(go.Bar(
        x=pnls, y=tickers,
        orientation="h",
        marker_color=colours,
        hovertemplate="<b>%{y}</b><br>€%{x:,.2f}<extra></extra>",
    ))
    fig.update_layout(
        title="Unrealised P&L",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font_color="#cccccc",
        xaxis=dict(gridcolor="#2a2a2a", tickprefix="€", zeroline=True,
                   zerolinecolor="#444", zerolinewidth=1),
        yaxis=dict(gridcolor="#2a2a2a"),
        margin=dict(t=40, b=10, l=10, r=10),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_value_chart(holdings, prices):
    tickers, invested_vals, current_vals = [], [], []
    for h in holdings:
        p = prices.get(h.ticker)
        if p is None:
            continue
        tickers.append(h.ticker)
        invested_vals.append(h.total_invested)
        current_vals.append(h.current_value(p))

    if not tickers:
        return

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Invested", x=tickers, y=invested_vals,
        marker_color=BLUE, opacity=0.75,
        hovertemplate="<b>%{x}</b><br>Invested: €%{y:,.2f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Current Value", x=tickers, y=current_vals,
        marker_color=[GAIN if c >= i else LOSS for c, i in zip(current_vals, invested_vals)],
        opacity=0.9,
        hovertemplate="<b>%{x}</b><br>Value: €%{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(
        title="Invested vs Current Value",
        barmode="group",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font_color="#cccccc",
        xaxis=dict(gridcolor="#2a2a2a"),
        yaxis=dict(gridcolor="#2a2a2a", tickprefix="€"),
        legend=dict(bgcolor="#1a1a1a", bordercolor="#2a2a2a"),
        margin=dict(t=40, b=10, l=10, r=10),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Holdings page ─────────────────────────────────────────────────────────────

def render_holdings():
    st.markdown("## Holdings")

    holdings = portfolio().all_holdings()
    prices   = get_prices()

    if not holdings:
        st.info("No holdings yet.")
        return

    # Build a DataFrame for display
    rows = []
    for h in holdings:
        p = prices.get(h.ticker)
        rows.append({
            "Ticker":      h.ticker,
            "Name":        h.name,
            "Type":        h.asset_type.upper(),
            "Qty":         round(h.quantity, 4),
            "Avg Cost":    round(h.average_cost, 4),
            "Price":       round(p, 4) if p else None,
            "Value":       round(h.current_value(p), 2) if p else None,
            "P&L":         round(h.unrealised_pnl(p), 2) if p else None,
            "P&L %":       round(h.pnl_percent(p), 2) if p else None,
        })

    df = pd.DataFrame(rows)

    # Colour the P&L columns
    def colour_pnl(val):
        if val is None or pd.isna(val):
            return "color: #888"
        return f"color: {GAIN}" if val > 0 else f"color: {LOSS}"

    styled = (
        df.style
        .applymap(colour_pnl, subset=["P&L", "P&L %"])
        .format({
            "Avg Cost": "€{:.4f}",
            "Price":    "€{:.4f}",
            "Value":    "€{:,.2f}",
            "P&L":      "€{:+,.2f}",
            "P&L %":    "{:+.2f}%",
        }, na_rep="—")
    )

    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.divider()

    # ── Holding detail expanders ──
    st.markdown('<p class="section-title">Transaction History</p>', unsafe_allow_html=True)

    for h in holdings:
        with st.expander(f"{h.ticker}  —  {h.name}"):
            p = prices.get(h.ticker)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Avg Cost",      fmt_cur(h.average_cost))
            m2.metric("Current Price", fmt_cur(p) if p else "—")
            m3.metric("Market Value",  fmt_cur(h.current_value(p)) if p else "—")
            if p:
                pnl = h.unrealised_pnl(p)
                m4.metric("P&L", fmt_cur(pnl), delta=fmt_pct(h.pnl_percent(p)))

            t_rows = [
                {
                    "Date":     t.date,
                    "Action":   t.action.upper(),
                    "Quantity": t.quantity,
                    "Price":    t.price,
                    "Total":    round(t.total_cost, 2),
                }
                for t in h.transactions
            ]
            t_df = pd.DataFrame(t_rows)

            def colour_action(val):
                return f"color: {GAIN}" if val == "BUY" else f"color: {LOSS}"

            styled_t = (
                t_df.style
                .applymap(colour_action, subset=["Action"])
                .format({"Price": "€{:.4f}", "Total": "€{:,.2f}"})
            )
            st.dataframe(styled_t, use_container_width=True, hide_index=True)

            # Remove holding button
            if st.button(f"🗑  Remove {h.ticker}", key=f"remove_{h.ticker}"):
                portfolio().remove_holding(h.ticker)
                st.session_state.prices = {}
                st.rerun()


# ── Add transaction page ───────────────────────────────────────────────────────

def render_add_transaction():
    st.markdown("## Add Transaction")

    holdings  = portfolio().all_holdings()
    existing  = {h.ticker: h for h in holdings}

    with st.form("add_transaction_form", clear_on_submit=True):
        col1, col2 = st.columns(2)

        with col1:
            action = st.selectbox("Action", ["BUY", "SELL"])
            ticker = st.text_input("Ticker", placeholder="e.g. AAPL, BTC-USD, SIE.DE").upper()

            # Auto-fill name/type if ticker already exists
            if ticker in existing:
                name       = existing[ticker].name
                asset_type = existing[ticker].asset_type
                st.info(f"Adding to existing position: {name}")
            else:
                name       = st.text_input("Name", placeholder="e.g. Apple Inc.")
                asset_type = st.selectbox("Asset Type", ["stock", "crypto", "etf"])

        with col2:
            quantity = st.number_input("Quantity", min_value=0.0001, step=0.0001, format="%.4f")
            price    = st.number_input("Price per unit (€)", min_value=0.0001, step=0.01, format="%.4f")
            txn_date = st.date_input("Date", value=date.today())

        submitted = st.form_submit_button("Add Transaction", use_container_width=True, type="primary")

        if submitted:
            if not ticker:
                st.error("Please enter a ticker symbol.")
            elif ticker not in existing and not name:
                st.error("Please enter a name for the new holding.")
            else:
                portfolio().add_transaction(
                    ticker=ticker,
                    name=name if ticker not in existing else existing[ticker].name,
                    asset_type=asset_type if ticker not in existing else existing[ticker].asset_type,
                    action=action.lower(),
                    quantity=quantity,
                    price=price,
                    date=str(txn_date),
                )
                st.session_state.prices = {}  # Invalidate price cache
                st.success(f"✓ {action} recorded for {ticker}")
                st.rerun()

    # ── Ticker format help ──
    with st.expander("📖  Ticker format guide"):
        st.markdown("""
| Asset | Ticker |
|---|---|
| US Stocks / ETFs | `AAPL`, `MSFT`, `SPY`, `QQQ` |
| German stocks (Xetra) | `SIE.DE`, `BMW.DE` |
| Dutch stocks (Euronext) | `ASML.AS`, `HEIA.AS` |
| French stocks | `MC.PA`, `TTE.PA` |
| London Stock Exchange | `VOD.L`, `SHEL.L` |
| Bitcoin / Ethereum | `BTC-USD`, `ETH-USD` |
| Other crypto | `SOL-USD`, `ADA-USD` |

Search for any ticker at [finance.yahoo.com](https://finance.yahoo.com)
        """)


# ── Covariance matrix page ────────────────────────────────────────────────────

def render_covariance():
    st.markdown("## Covariance & Correlation Matrix")
    st.caption("Based on 3 years of weekly returns · sourced from Yahoo Finance")

    holdings = portfolio().all_holdings()

    if len(holdings) < 2:
        st.info("You need at least 2 holdings to compute a covariance matrix.")
        return

    all_tickers = [h.ticker for h in holdings]
    selected = st.multiselect(
        "Select tickers to analyse",
        options=all_tickers,
        default=all_tickers,
    )

    if len(selected) < 2:
        st.warning("Please select at least 2 tickers.")
        return

    if st.button("Run Analysis", type="primary"):
        import yfinance as yf
        import numpy as np

        with st.spinner("Downloading 3 years of weekly data…"):
            try:
                raw = yf.download(selected, period="3y", interval="1wk",
                                  progress=False, auto_adjust=True)
            except Exception as e:
                st.error(f"Failed to download data: {e}")
                return

        if raw.empty:
            st.error("No data returned. Check your ticker symbols.")
            return

        prices_df = raw["Close"] if len(selected) > 1 else raw[["Close"]]
        if len(selected) == 1:
            prices_df.columns = [selected[0].upper()]
        else:
            prices_df.columns = [c.upper() for c in prices_df.columns]

        returns    = prices_df.pct_change().dropna()
        cov_matrix = returns.cov()
        corr_matrix = returns.corr()

        # ── Stats row ──
        st.divider()
        st.markdown('<p class="section-title">Individual Statistics  (weekly)</p>', unsafe_allow_html=True)

        cols = st.columns(len(selected))
        for i, ticker in enumerate(returns.columns):
            col      = returns[ticker].dropna()
            avg      = col.mean()
            ann_vol  = col.std() * np.sqrt(52)
            delta_colour = "normal" if avg >= 0 else "inverse"
            cols[i].metric(
                ticker,
                f"Ann. Vol: {ann_vol:.1%}",
                delta=f"Avg weekly: {avg:+.3%}",
                delta_color=delta_colour,
            )

        st.divider()

        # ── Heatmaps side by side ──
        col_cov, col_corr = st.columns(2)

        with col_cov:
            st.markdown('<p class="section-title">Covariance Matrix</p>', unsafe_allow_html=True)
            _render_heatmap(cov_matrix, fmt=".5f", title="Covariance")

        with col_corr:
            st.markdown('<p class="section-title">Correlation Matrix</p>', unsafe_allow_html=True)
            _render_heatmap(corr_matrix, fmt=".3f", title="Correlation",
                            zmin=-1, zmax=1, zmid=0)

        st.divider()

        # ── Raw matrices as tables ──
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown('<p class="section-title">Covariance Table</p>', unsafe_allow_html=True)
            st.dataframe(cov_matrix.style.format("{:.6f}"), use_container_width=True)
        with col_t2:
            st.markdown('<p class="section-title">Correlation Table</p>', unsafe_allow_html=True)
            st.dataframe(corr_matrix.style.format("{:.4f}").background_gradient(
                cmap="RdYlGn", vmin=-1, vmax=1
            ), use_container_width=True)


def _render_heatmap(matrix: "pd.DataFrame", fmt: str, title: str,
                    zmin=None, zmax=None, zmid=None):
    import plotly.figure_factory as ff
    import numpy as np

    z      = matrix.values.tolist()
    labels = list(matrix.columns)
    text   = [[format(v, fmt) for v in row] for row in matrix.values]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        text=text,
        texttemplate="%{text}",
        colorscale="RdYlGn",
        zmin=zmin,
        zmax=zmax,
        zmid=zmid,
        showscale=True,
        hovertemplate="<b>%{y} × %{x}</b><br>%{text}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font_color="#cccccc",
        font_size=11,
        margin=dict(t=10, b=10, l=10, r=10),
        height=350,
        xaxis=dict(side="bottom"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Main router ───────────────────────────────────────────────────────────────

def main():
    page = render_sidebar()

    if page == "Dashboard":
        render_dashboard()
    elif page == "Holdings":
        render_holdings()
    elif page == "Add Transaction":
        render_add_transaction()
    elif page == "Covariance Matrix":
        render_covariance()


if __name__ == "__main__":
    main()
