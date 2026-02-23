"""
app.py
======
Streamlit web interface for the Portfolio Tracker.
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, datetime
from typing import Dict, List, Optional

from tracker.portfolio import Portfolio, Holding
from tracker.prices import PriceFetcher
from tracker.benchmark import (
    INDICES, build_portfolio_value_series, fetch_index_series,
    normalise, compute_drawdown, compute_stats, get_portfolio_start_date,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme ─────────────────────────────────────────────────────────────────────
GAIN    = "#4caf7d"
LOSS    = "#e05c5c"
BLUE    = "#5b9bd5"
MUTED   = "#888888"
BG      = "#0f0f0f"
PALETTE = ["#5b9bd5", "#4caf7d", "#e8a838", "#b07fd4",
           "#e05c5c", "#4db6ac", "#f06292", "#a1887f"]

# Benchmark line colours — distinct from portfolio palette
BENCH_COLOURS = {
    "MSCI World":         "#e8a838",
    "S&P 500":            "#b07fd4",
    "NASDAQ 100":         "#4db6ac",
    "MSCI Emerging Mkts": "#f06292",
}

st.markdown("""
<style>
  [data-testid="metric-container"] {
      background: #1a1a1a; border: 1px solid #2a2a2a;
      border-radius: 8px; padding: 14px 18px;
  }
  [data-testid="stMetricValue"]  { font-size: 1.35rem; }
  [data-testid="stMetricDelta"]  { font-size: 0.85rem; }
  [data-testid="stDataFrame"]    { border-radius: 8px; overflow: hidden; }
  [data-testid="stSidebar"]      { background: #111111; }
  .block-container               { padding-top: 1.5rem; }
  .section-title {
      font-size: 0.75rem; font-weight: 600; letter-spacing: 0.1em;
      text-transform: uppercase; color: #555; margin: 1.5rem 0 0.5rem 0;
  }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
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

# ── Prices ────────────────────────────────────────────────────────────────────
def fetch_prices() -> Dict[str, Optional[float]]:
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

def _chart_layout(title="", height=400) -> dict:
    """Shared Plotly layout for a consistent dark theme."""
    return dict(
        title=title,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font_color="#cccccc",
        height=height,
        xaxis=dict(gridcolor="#1e1e1e", showgrid=True),
        yaxis=dict(gridcolor="#1e1e1e", showgrid=True),
        legend=dict(bgcolor="#1a1a1a", bordercolor="#2a2a2a", borderwidth=1),
        margin=dict(t=50, b=20, l=10, r=10),
        hovermode="x unified",
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## 📈 Portfolio Tracker")
        st.divider()

        page = st.radio(
            "Navigation",
            ["Dashboard", "Holdings", "Add Transaction",
             "Benchmark", "Covariance Matrix"],
            label_visibility="collapsed",
        )

        st.divider()

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

# ── Dashboard ─────────────────────────────────────────────────────────────────
def render_dashboard():
    st.markdown("## Dashboard")
    holdings = portfolio().all_holdings()
    prices   = get_prices()

    if not holdings:
        st.info("Your portfolio is empty. Go to **Add Transaction** to get started.")
        return

    total_value    = sum(h.current_value(prices[h.ticker]) for h in holdings if prices.get(h.ticker))
    total_invested = sum(h.total_invested for h in holdings)
    total_pnl      = total_value - total_invested
    overall_pct    = (total_pnl / total_invested * 100) if total_invested else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Invested",  fmt_cur(total_invested))
    c2.metric("Portfolio Value", fmt_cur(total_value))
    c3.metric("Unrealised P&L",  fmt_cur(total_pnl), delta=fmt_pct(overall_pct))
    c4.metric("Holdings",        str(len(holdings)))

    st.divider()
    col_left, col_right = st.columns(2)
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
        if p is None: continue
        labels.append(h.ticker)
        values.append(h.current_value(p))
        colours.append(PALETTE[i % len(PALETTE)])
    if not values: return

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.55,
        marker=dict(colors=colours, line=dict(color=BG, width=2)),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>€%{value:,.2f}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(**_chart_layout("Allocation", 320))
    fig.update_layout(
        showlegend=False,
        annotations=[dict(text=f"€{sum(values):,.0f}", x=0.5, y=0.5,
                          font_size=16, showarrow=False, font_color="#cccccc")],
    )
    st.plotly_chart(fig, use_container_width=True)

def _render_pnl_chart(holdings, prices):
    tickers, pnls = [], []
    for h in holdings:
        p = prices.get(h.ticker)
        if p is None: continue
        tickers.append(h.ticker)
        pnls.append(h.unrealised_pnl(p))
    if not tickers: return

    fig = go.Figure(go.Bar(
        x=pnls, y=tickers, orientation="h",
        marker_color=[GAIN if p >= 0 else LOSS for p in pnls],
        hovertemplate="<b>%{y}</b><br>€%{x:,.2f}<extra></extra>",
    ))
    fig.update_layout(**_chart_layout("Unrealised P&L", 320))
    fig.update_layout(xaxis=dict(gridcolor="#1e1e1e", tickprefix="€",
                                 zeroline=True, zerolinecolor="#444"))
    st.plotly_chart(fig, use_container_width=True)

def _render_value_chart(holdings, prices):
    tickers, inv, cur = [], [], []
    for h in holdings:
        p = prices.get(h.ticker)
        if p is None: continue
        tickers.append(h.ticker)
        inv.append(h.total_invested)
        cur.append(h.current_value(p))
    if not tickers: return

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Invested", x=tickers, y=inv,
                         marker_color=BLUE, opacity=0.75,
                         hovertemplate="<b>%{x}</b><br>Invested: €%{y:,.2f}<extra></extra>"))
    fig.add_trace(go.Bar(name="Current Value", x=tickers, y=cur,
                         marker_color=[GAIN if c >= i else LOSS for c, i in zip(cur, inv)],
                         hovertemplate="<b>%{x}</b><br>Value: €%{y:,.2f}<extra></extra>"))
    fig.update_layout(**_chart_layout("Invested vs Current Value", 320))
    fig.update_layout(barmode="group", yaxis=dict(gridcolor="#1e1e1e", tickprefix="€"))
    st.plotly_chart(fig, use_container_width=True)

# ── Holdings ──────────────────────────────────────────────────────────────────
def render_holdings():
    st.markdown("## Holdings")
    holdings = portfolio().all_holdings()
    prices   = get_prices()

    if not holdings:
        st.info("No holdings yet.")
        return

    rows = []
    for h in holdings:
        p = prices.get(h.ticker)
        rows.append({
            "Ticker":   h.ticker, "Name": h.name, "Type": h.asset_type.upper(),
            "Qty":      round(h.quantity, 4), "Avg Cost": round(h.average_cost, 4),
            "Price":    round(p, 4) if p else None,
            "Value":    round(h.current_value(p), 2) if p else None,
            "P&L":      round(h.unrealised_pnl(p), 2) if p else None,
            "P&L %":    round(h.pnl_percent(p), 2) if p else None,
        })

    df = pd.DataFrame(rows)

    def colour_pnl(val):
        if val is None or (isinstance(val, float) and pd.isna(val)): return "color: #888"
        return f"color: {GAIN}" if val > 0 else f"color: {LOSS}"

    styled = (df.style
              .applymap(colour_pnl, subset=["P&L", "P&L %"])
              .format({"Avg Cost": "€{:.4f}", "Price": "€{:.4f}",
                       "Value": "€{:,.2f}", "P&L": "€{:+,.2f}",
                       "P&L %": "{:+.2f}%"}, na_rep="—"))
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.divider()

    st.markdown('<p class="section-title">Transaction History</p>', unsafe_allow_html=True)
    for h in holdings:
        with st.expander(f"{h.ticker}  —  {h.name}"):
            p = prices.get(h.ticker)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Avg Cost", fmt_cur(h.average_cost))
            m2.metric("Current Price", fmt_cur(p) if p else "—")
            m3.metric("Market Value", fmt_cur(h.current_value(p)) if p else "—")
            if p:
                m4.metric("P&L", fmt_cur(h.unrealised_pnl(p)), delta=fmt_pct(h.pnl_percent(p)))

            t_rows = [{"Date": t.date, "Action": t.action.upper(),
                       "Quantity": t.quantity, "Price": t.price,
                       "Total": round(t.total_cost, 2)} for t in h.transactions]
            t_df = pd.DataFrame(t_rows)
            def colour_action(val):
                return f"color: {GAIN}" if val == "BUY" else f"color: {LOSS}"
            st.dataframe(
                t_df.style.applymap(colour_action, subset=["Action"])
                    .format({"Price": "€{:.4f}", "Total": "€{:,.2f}"}),
                use_container_width=True, hide_index=True,
            )
            if st.button(f"🗑  Remove {h.ticker}", key=f"remove_{h.ticker}"):
                portfolio().remove_holding(h.ticker)
                st.session_state.prices = {}
                st.rerun()

# ── Add Transaction ───────────────────────────────────────────────────────────
def render_add_transaction():
    st.markdown("## Add Transaction")
    holdings = portfolio().all_holdings()
    existing = {h.ticker: h for h in holdings}

    with st.form("add_transaction_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            action = st.selectbox("Action", ["BUY", "SELL"])
            ticker = st.text_input("Ticker", placeholder="e.g. AAPL, BTC-USD, SIE.DE").upper()
            if ticker in existing:
                name       = existing[ticker].name
                asset_type = existing[ticker].asset_type
                st.info(f"Adding to existing position: {name}")
            else:
                name       = st.text_input("Name", placeholder="e.g. Apple Inc.")
                asset_type = st.selectbox("Asset Type", ["stock", "crypto", "etf"])
        with col2:
            quantity = st.number_input("Quantity",        min_value=0.0001, step=0.0001, format="%.4f")
            price    = st.number_input("Price per unit (€)", min_value=0.0001, step=0.01,   format="%.4f")
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
                    action=action.lower(), quantity=quantity,
                    price=price, date=str(txn_date),
                )
                st.session_state.prices = {}
                st.success(f"✓ {action} recorded for {ticker}")
                st.rerun()

    with st.expander("📖  Ticker format guide"):
        st.markdown("""
| Asset | Ticker |
|---|---|
| US Stocks / ETFs | `AAPL`, `MSFT`, `SPY`, `QQQ` |
| German stocks | `SIE.DE`, `BMW.DE` |
| Dutch stocks | `ASML.AS`, `HEIA.AS` |
| French stocks | `MC.PA`, `TTE.PA` |
| London Stock Exchange | `VOD.L`, `SHEL.L` |
| Bitcoin / Ethereum | `BTC-USD`, `ETH-USD` |
        """)

# ── Benchmark ─────────────────────────────────────────────────────────────────
def render_benchmark():
    st.markdown("## Benchmark")
    st.caption("Portfolio performance vs market indices · rebased to €100 from your first transaction")

    port = portfolio()
    start_date = get_portfolio_start_date(port)

    if start_date is None:
        st.info("Add some transactions first so we have a start date to benchmark from.")
        return

    st.info(f"Benchmarking from your first transaction: **{start_date.strftime('%d %b %Y')}**")

    # Index selection
    selected_indices = st.multiselect(
        "Compare against",
        options=list(INDICES.keys()),
        default=list(INDICES.keys()),
    )

    if st.button("Run Benchmark", type="primary"):
        # ── Fetch portfolio value series ──
        with st.spinner("Building portfolio value series…"):
            port_series = build_portfolio_value_series(port, start_date)

        if port_series is None or port_series.empty:
            st.error("Could not build portfolio series. Check that your tickers are valid.")
            return

        # ── Fetch index series ──
        index_series: Dict[str, pd.Series] = {}
        for name in selected_indices:
            ticker = INDICES[name]
            with st.spinner(f"Fetching {name} ({ticker})…"):
                s = fetch_index_series(ticker, start_date)
            if s is not None:
                index_series[name] = s
            else:
                st.warning(f"Could not fetch data for {name} ({ticker})")

        # ── Normalise everything to 100 ──
        norm_port = normalise(port_series)
        norm_indices = {name: normalise(s.reindex(port_series.index, method="ffill"))
                        for name, s in index_series.items()}

        st.divider()

        # ── Cumulative return chart ──
        st.markdown('<p class="section-title">Growth of €100</p>', unsafe_allow_html=True)
        fig_growth = go.Figure()

        fig_growth.add_trace(go.Scatter(
            x=norm_port.index, y=norm_port.values,
            name="My Portfolio",
            line=dict(color=BLUE, width=2.5),
            hovertemplate="<b>Portfolio</b><br>€%{y:.2f}<extra></extra>",
        ))
        for name, s in norm_indices.items():
            fig_growth.add_trace(go.Scatter(
                x=s.index, y=s.values,
                name=name,
                line=dict(color=BENCH_COLOURS.get(name, "#aaa"), width=1.5, dash="dot"),
                hovertemplate=f"<b>{name}</b><br>€%{{y:.2f}}<extra></extra>",
            ))

        fig_growth.add_hline(y=100, line_color="#333", line_dash="dash", line_width=1)
        fig_growth.update_layout(**_chart_layout(height=420))
        fig_growth.update_layout(
            yaxis=dict(gridcolor="#1e1e1e", tickprefix="€"),
            xaxis=dict(gridcolor="#1e1e1e"),
        )
        st.plotly_chart(fig_growth, use_container_width=True)

        st.divider()

        # ── Drawdown chart ──
        st.markdown('<p class="section-title">Drawdown  (% below previous peak)</p>',
                    unsafe_allow_html=True)
        fig_dd = go.Figure()

        dd_port = compute_drawdown(norm_port)
        fig_dd.add_trace(go.Scatter(
            x=dd_port.index, y=dd_port.values,
            name="My Portfolio",
            fill="tozeroy",
            fillcolor="rgba(91,155,213,0.12)",
            line=dict(color=BLUE, width=2),
            hovertemplate="<b>Portfolio</b><br>%{y:.2f}%<extra></extra>",
        ))
        for name, s in norm_indices.items():
            dd = compute_drawdown(s)
            fig_dd.add_trace(go.Scatter(
                x=dd.index, y=dd.values,
                name=name,
                line=dict(color=BENCH_COLOURS.get(name, "#aaa"), width=1.5, dash="dot"),
                hovertemplate=f"<b>{name}</b><br>%{{y:.2f}}%<extra></extra>",
            ))

        fig_dd.update_layout(**_chart_layout(height=340))
        fig_dd.update_layout(
            yaxis=dict(gridcolor="#1e1e1e", ticksuffix="%"),
            xaxis=dict(gridcolor="#1e1e1e"),
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        st.divider()

        # ── Stats table ──
        st.markdown('<p class="section-title">Key Statistics</p>', unsafe_allow_html=True)

        all_stats = [compute_stats(port_series, "My Portfolio")]
        for name, s in index_series.items():
            aligned = s.reindex(port_series.index, method="ffill").dropna()
            all_stats.append(compute_stats(aligned, name))

        # Remove raw helper columns before display
        display_cols = ["Label", "Total Return", "Ann. Return",
                        "Ann. Volatility", "Sharpe Ratio", "Max Drawdown",
                        "Best Day", "Worst Day", "Days"]
        stats_df = pd.DataFrame(all_stats)[display_cols].rename(columns={"Label": ""})

        def colour_stat(val):
            """Colour positive/negative return and sharpe cells."""
            try:
                num = float(str(val).replace("%", "").replace("+", ""))
                if num > 0: return f"color: {GAIN}"
                if num < 0: return f"color: {LOSS}"
            except Exception:
                pass
            return ""

        styled_stats = (
            stats_df.style
            .applymap(colour_stat, subset=["Total Return", "Ann. Return",
                                           "Max Drawdown", "Best Day", "Worst Day"])
        )
        st.dataframe(styled_stats, use_container_width=True, hide_index=True)

# ── Covariance ────────────────────────────────────────────────────────────────
def render_covariance():
    st.markdown("## Covariance & Correlation Matrix")
    st.caption("Based on 3 years of weekly returns · sourced from Yahoo Finance")

    holdings = portfolio().all_holdings()
    if len(holdings) < 2:
        st.info("You need at least 2 holdings to compute a covariance matrix.")
        return

    all_tickers = [h.ticker for h in holdings]
    selected    = st.multiselect("Select tickers to analyse", options=all_tickers, default=all_tickers)

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
            st.error("No data returned.")
            return

        prices_df = raw["Close"] if len(selected) > 1 else raw[["Close"]]
        if len(selected) == 1:
            prices_df.columns = [selected[0].upper()]
        else:
            prices_df.columns = [c.upper() for c in prices_df.columns]

        returns     = prices_df.pct_change().dropna()
        cov_matrix  = returns.cov()
        corr_matrix = returns.corr()

        st.divider()
        st.markdown('<p class="section-title">Individual Statistics (weekly)</p>',
                    unsafe_allow_html=True)

        cols = st.columns(len(selected))
        for i, ticker in enumerate(returns.columns):
            col     = returns[ticker].dropna()
            avg     = col.mean()
            ann_vol = col.std() * np.sqrt(52)
            cols[i].metric(ticker, f"Ann. Vol: {ann_vol:.1%}",
                           delta=f"Avg weekly: {avg:+.3%}",
                           delta_color="normal" if avg >= 0 else "inverse")

        st.divider()
        col_cov, col_corr = st.columns(2)
        with col_cov:
            st.markdown('<p class="section-title">Covariance Matrix</p>', unsafe_allow_html=True)
            _render_heatmap(cov_matrix, fmt=".5f")
        with col_corr:
            st.markdown('<p class="section-title">Correlation Matrix</p>', unsafe_allow_html=True)
            _render_heatmap(corr_matrix, fmt=".3f", zmin=-1, zmax=1, zmid=0)

        st.divider()
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown('<p class="section-title">Covariance Table</p>', unsafe_allow_html=True)
            st.dataframe(cov_matrix.style.format("{:.6f}"), use_container_width=True)
        with col_t2:
            st.markdown('<p class="section-title">Correlation Table</p>', unsafe_allow_html=True)
            st.dataframe(corr_matrix.style.format("{:.4f}").background_gradient(
                cmap="RdYlGn", vmin=-1, vmax=1), use_container_width=True)

def _render_heatmap(matrix, fmt, zmin=None, zmax=None, zmid=None):
    labels = list(matrix.columns)
    text   = [[format(v, fmt) for v in row] for row in matrix.values]
    fig    = go.Figure(go.Heatmap(
        z=matrix.values.tolist(), x=labels, y=labels,
        text=text, texttemplate="%{text}",
        colorscale="RdYlGn", zmin=zmin, zmax=zmax, zmid=zmid,
        showscale=True,
        hovertemplate="<b>%{y} × %{x}</b><br>%{text}<extra></extra>",
    ))
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, font_color="#cccccc",
                      font_size=11, margin=dict(t=10, b=10, l=10, r=10), height=350)
    st.plotly_chart(fig, use_container_width=True)

# ── Router ────────────────────────────────────────────────────────────────────
def main():
    page = render_sidebar()
    if page == "Dashboard":
        render_dashboard()
    elif page == "Holdings":
        render_holdings()
    elif page == "Add Transaction":
        render_add_transaction()
    elif page == "Benchmark":
        render_benchmark()
    elif page == "Covariance Matrix":
        render_covariance()

if __name__ == "__main__":
    main()
