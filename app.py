"""
app.py  —  Portfolio Tracker (Streamlit)
Run with:  streamlit run app.py
"""

import io, os, tempfile
import numpy as np
from datetime import date, datetime
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from tracker.benchmark import (
    INDICES, build_portfolio_value_series, compute_drawdown,
    compute_stats, fetch_index_series, get_portfolio_start_date, normalise,
)
from tracker.portfolio import Holding, Portfolio, Transaction
from tracker.prices import PriceFetcher
import tracker.exporter as exporter

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Portfolio Tracker", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

# ── Palette ───────────────────────────────────────────────────────────────────
GAIN    = "#4caf7d"
LOSS    = "#e05c5c"
BLUE    = "#5b9bd5"
BG      = "#0f0f0f"
PALETTE = ["#5b9bd5","#4caf7d","#e8a838","#b07fd4","#e05c5c","#4db6ac","#f06292","#a1887f"]
BENCH_COLOURS = {"MSCI World":"#e8a838","S&P 500":"#b07fd4",
                 "NASDAQ 100":"#4db6ac","MSCI Emerging Mkts":"#f06292"}

st.markdown("""
<style>
  [data-testid="metric-container"]{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:8px;padding:14px 18px}
  [data-testid="stMetricValue"]{font-size:1.35rem}
  [data-testid="stMetricDelta"]{font-size:0.85rem}
  [data-testid="stDataFrame"]{border-radius:8px;overflow:hidden}
  [data-testid="stSidebar"]{background:#111}
  .block-container{padding-top:1.5rem}
  .section-title{font-size:.75rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:#555;margin:1.5rem 0 .5rem}
  .news-card{background:#141414;border:1px solid #222;border-radius:8px;padding:12px 16px;margin-bottom:10px}
  .news-card a{color:#5b9bd5;text-decoration:none;font-weight:500}
  .news-card a:hover{text-decoration:underline}
  .news-meta{font-size:.75rem;color:#555;margin-top:4px}
  .news-ticker{display:inline-block;background:#1e2a38;color:#5b9bd5;font-size:.7rem;font-weight:700;border-radius:4px;padding:1px 6px;margin-right:6px}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "portfolio"   not in st.session_state: st.session_state.portfolio   = Portfolio()
if "fetcher"     not in st.session_state: st.session_state.fetcher     = PriceFetcher()
if "prices"      not in st.session_state: st.session_state.prices      = {}
if "news_cache"  not in st.session_state: st.session_state.news_cache  = {}

def portfolio() -> Portfolio:    return st.session_state.portfolio
def fetcher()   -> PriceFetcher: return st.session_state.fetcher

# ── Prices ────────────────────────────────────────────────────────────────────
def get_prices() -> Dict[str, Optional[float]]:
    if not st.session_state.prices:
        tickers = [h.ticker for h in portfolio().all_holdings()]
        if tickers:
            with st.spinner("Fetching live prices…"):
                st.session_state.prices = fetcher().get_prices(tickers)
    return st.session_state.prices

def invalidate_prices():
    fetcher().clear_cache()
    st.session_state.prices = {}

# ── Shared helpers ────────────────────────────────────────────────────────────
def fmt_cur(v: float) -> str: return f"€{v:,.2f}"
def fmt_pct(v: float) -> str: return f"{'+'if v>0 else''}{v:.2f}%"

def _chart_layout(title="", height=400) -> dict:
    return dict(title=title, paper_bgcolor=BG, plot_bgcolor=BG,
                font_color="#cccccc", height=height,
                xaxis=dict(gridcolor="#1e1e1e"),
                yaxis=dict(gridcolor="#1e1e1e"),
                legend=dict(bgcolor="#1a1a1a", bordercolor="#2a2a2a", borderwidth=1),
                margin=dict(t=50, b=20, l=10, r=10), hovermode="x unified")

def _colour_pnl(val):
    if val is None or (isinstance(val, float) and pd.isna(val)): return "color:#888"
    return f"color:{GAIN}" if val > 0 else f"color:{LOSS}"

def _colour_stat(val):
    try:
        n = float(str(val).replace("%","").replace("+",""))
        return f"color:{GAIN}" if n > 0 else (f"color:{LOSS}" if n < 0 else "")
    except Exception:
        return ""

def _excel_bytes(holdings, prices) -> io.BytesIO:
    """Write Excel to a cross-platform temp file, return as BytesIO buffer."""
    buf = io.BytesIO()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        tmp_path = tmp.name
    try:
        exporter.export_to_excel(holdings, prices, filename=tmp_path)
        with open(tmp_path, "rb") as f:
            buf.write(f.read())
    finally:
        os.unlink(tmp_path)
    buf.seek(0)
    return buf

# ── News ──────────────────────────────────────────────────────────────────────
def _fetch_news(tickers: List[str]) -> List[dict]:
    cache = st.session_state.news_cache
    all_news = []
    for ticker in tickers:
        if ticker not in cache:
            try:    cache[ticker] = yf.Ticker(ticker).news or []
            except: cache[ticker] = []
        for a in cache[ticker][:5]:
            all_news.append({**a, "_ticker": ticker})
    all_news.sort(key=lambda x: x.get("providerPublishTime", 0), reverse=True)
    seen, unique = set(), []
    for a in all_news:
        if (t := a.get("title","")) not in seen:
            seen.add(t); unique.append(a)
    return unique[:20]

def _render_news(tickers: List[str]):
    st.markdown('<p class="section-title">Latest News</p>', unsafe_allow_html=True)
    col, _ = st.columns([1, 5])
    with col:
        if st.button("↺  Refresh News", key="refresh_news"):
            st.session_state.news_cache = {}; st.rerun()
    with st.spinner("Loading news…"):
        articles = _fetch_news(tickers)
    if not articles:
        st.caption("No news found for your holdings."); return
    for a in articles:
        try:    pub = datetime.fromtimestamp(a.get("providerPublishTime",0)).strftime("%d %b %Y  %H:%M")
        except: pub = ""
        st.markdown(f"""<div class="news-card">
  <span class="news-ticker">{a.get("_ticker","")}</span>
  <a href="{a.get("link","#")}" target="_blank">{a.get("title","")}</a>
  <div class="news-meta">{a.get("publisher","")}  ·  {pub}</div>
</div>""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## 📈 Portfolio Tracker")
        st.divider()
        page = st.radio("Nav", ["Dashboard","Holdings","Add Transaction",
                                "Benchmark","Covariance Matrix"],
                        label_visibility="collapsed")
        st.divider()
        holdings = portfolio().all_holdings()
        prices   = get_prices()
        tv  = sum(h.current_value(prices[h.ticker]) for h in holdings if prices.get(h.ticker))
        ti  = sum(h.total_invested for h in holdings)
        pnl = tv - ti
        st.metric("Portfolio Value", fmt_cur(tv))
        st.metric("Unrealised P&L",  fmt_cur(pnl), delta=fmt_pct(pnl/ti*100 if ti else 0))
        st.divider()
        if st.button("🔄  Refresh Prices", use_container_width=True):
            invalidate_prices(); st.rerun()
    return page

# ── Dashboard ─────────────────────────────────────────────────────────────────
def render_dashboard():
    st.markdown("## Dashboard")
    holdings = portfolio().all_holdings()
    prices   = get_prices()
    if not holdings:
        st.info("Your portfolio is empty. Go to **Add Transaction** to get started.")
        return

    tv  = sum(h.current_value(prices[h.ticker]) for h in holdings if prices.get(h.ticker))
    ti  = sum(h.total_invested for h in holdings)
    pnl = tv - ti

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Invested",  fmt_cur(ti))
    c2.metric("Portfolio Value", fmt_cur(tv))
    c3.metric("Unrealised P&L",  fmt_cur(pnl), delta=fmt_pct(pnl/ti*100 if ti else 0))
    c4.metric("Holdings",        str(len(holdings)))

    st.divider()
    col_l, col_r = st.columns(2)
    with col_l: _chart_allocation(holdings, prices)
    with col_r: _chart_pnl(holdings, prices)
    st.divider()
    _chart_value(holdings, prices)
    st.divider()
    _render_news([h.ticker for h in holdings])

def _chart_allocation(holdings, prices):
    labels, values, colours = [], [], []
    for i, h in enumerate(holdings):
        if (p := prices.get(h.ticker)) is None: continue
        labels.append(h.ticker); values.append(h.current_value(p))
        colours.append(PALETTE[i % len(PALETTE)])
    if not values: return
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.55,
        marker=dict(colors=colours, line=dict(color=BG, width=2)),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>€%{value:,.2f}<br>%{percent}<extra></extra>"))
    fig.update_layout(**_chart_layout("Allocation", 320))
    fig.update_layout(showlegend=False,
        annotations=[dict(text=f"€{sum(values):,.0f}", x=0.5, y=0.5,
                          font_size=16, showarrow=False, font_color="#cccccc")])
    st.plotly_chart(fig, use_container_width=True)

def _chart_pnl(holdings, prices):
    rows = [(h.ticker, h.unrealised_pnl(p))
            for h in holdings if (p := prices.get(h.ticker)) is not None]
    if not rows: return
    tickers, pnls = zip(*rows)
    fig = go.Figure(go.Bar(x=list(pnls), y=list(tickers), orientation="h",
        marker_color=[GAIN if p >= 0 else LOSS for p in pnls],
        hovertemplate="<b>%{y}</b><br>€%{x:,.2f}<extra></extra>"))
    fig.update_layout(**_chart_layout("Unrealised P&L", 320))
    fig.update_layout(xaxis=dict(gridcolor="#1e1e1e", tickprefix="€",
                                 zeroline=True, zerolinecolor="#444"))
    st.plotly_chart(fig, use_container_width=True)

def _chart_value(holdings, prices):
    rows = [(h.ticker, h.total_invested, h.current_value(p))
            for h in holdings if (p := prices.get(h.ticker)) is not None]
    if not rows: return
    tickers, inv, cur = zip(*rows)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Invested", x=list(tickers), y=list(inv),
        marker_color=BLUE, opacity=0.75,
        hovertemplate="<b>%{x}</b><br>Invested: €%{y:,.2f}<extra></extra>"))
    fig.add_trace(go.Bar(name="Current Value", x=list(tickers), y=list(cur),
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
        st.info("No holdings yet."); return

    # Summary table
    rows = [{"Ticker": h.ticker, "Name": h.name, "Type": h.asset_type.upper(),
             "Qty": round(h.quantity, 4), "Avg Cost": round(h.average_cost, 4),
             "Price":  round(p, 4) if (p := prices.get(h.ticker)) else None,
             "Value":  round(h.current_value(p), 2) if p else None,
             "P&L":    round(h.unrealised_pnl(p), 2) if p else None,
             "P&L %":  round(h.pnl_percent(p), 2) if p else None}
            for h in holdings]
    st.dataframe(
        pd.DataFrame(rows).style
          .applymap(_colour_pnl, subset=["P&L","P&L %"])
          .format({"Avg Cost":"€{:.4f}","Price":"€{:.4f}","Value":"€{:,.2f}",
                   "P&L":"€{:+,.2f}","P&L %":"{:+.2f}%"}, na_rep="—"),
        use_container_width=True, hide_index=True)

    # Export
    st.divider()
    st.markdown('<p class="section-title">Export</p>', unsafe_allow_html=True)
    col_xl, col_csv, _ = st.columns([1, 1, 4])
    with col_xl:
        st.download_button("⬇  Download Excel",
            data=_excel_bytes(holdings, prices),
            file_name=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True)
    with col_csv:
        csv_rows = [{"ticker":h.ticker,"name":h.name,"type":h.asset_type,
                     "quantity":round(h.quantity,6),"avg_cost":round(h.average_cost,4),
                     "current_price":round(p,4) if (p:=prices.get(h.ticker)) else "",
                     "market_value":round(h.current_value(p),2) if p else "",
                     "unrealised_pnl":round(h.unrealised_pnl(p),2) if p else "",
                     "pnl_percent":round(h.pnl_percent(p),2) if p else ""}
                    for h in holdings]
        st.download_button("⬇  Download CSV",
            data=pd.DataFrame(csv_rows).to_csv(index=False),
            file_name=f"portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv", use_container_width=True)

    # Per-holding detail
    st.divider()
    st.markdown('<p class="section-title">Holding Detail</p>', unsafe_allow_html=True)
    for h in holdings:
        with st.expander(f"{h.ticker}  —  {h.name}"):
            p = prices.get(h.ticker)
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Avg Cost",      fmt_cur(h.average_cost))
            m2.metric("Current Price", fmt_cur(p) if p else "—")
            m3.metric("Market Value",  fmt_cur(h.current_value(p)) if p else "—")
            if p:
                m4.metric("P&L", fmt_cur(h.unrealised_pnl(p)), delta=fmt_pct(h.pnl_percent(p)))

            # Price history
            period_map = {"1M":"1mo","3M":"3mo","6M":"6mo","1Y":"1y","3Y":"3y","5Y":"5y"}
            period = st.radio("Period", list(period_map.keys()),
                              horizontal=True, key=f"period_{h.ticker}")
            _chart_price_history(h.ticker, period_map[period], h.transactions)

            # Editable transaction table
            st.markdown('<p class="section-title">Transactions  (click a cell to edit)</p>',
                        unsafe_allow_html=True)
            original_df = pd.DataFrame([{"Date":t.date,"Action":t.action.lower(),
                                          "Quantity":t.quantity,"Price":t.price}
                                         for t in h.transactions])
            edited_df = st.data_editor(
                original_df, key=f"editor_{h.ticker}",
                use_container_width=True, hide_index=True, num_rows="fixed",
                column_config={
                    "Date":     st.column_config.TextColumn("Date", help="YYYY-MM-DD", width="small"),
                    "Action":   st.column_config.SelectboxColumn("Action", options=["buy","sell"], width="small"),
                    "Quantity": st.column_config.NumberColumn("Quantity", min_value=0.0001, format="%.4f", width="small"),
                    "Price":    st.column_config.NumberColumn("Price (€)", min_value=0.0001, format="€%.4f", width="small"),
                })

            if not edited_df.equals(original_df):
                if st.button("💾  Save changes", key=f"save_{h.ticker}", type="primary"):
                    portfolio().holdings[h.ticker].transactions = [
                        Transaction(date=str(r["Date"]).strip(),
                                    action=str(r["Action"]).lower().strip(),
                                    quantity=float(r["Quantity"]),
                                    price=float(r["Price"]))
                        for _, r in edited_df.iterrows()
                    ]
                    portfolio().save()
                    invalidate_prices()
                    st.success("✓ Saved"); st.rerun()

            col_del, _ = st.columns([1, 4])
            with col_del:
                if st.button(f"🗑  Remove {h.ticker}", key=f"remove_{h.ticker}"):
                    portfolio().remove_holding(h.ticker)
                    invalidate_prices(); st.rerun()

# ── Price history chart ───────────────────────────────────────────────────────
def _chart_price_history(ticker: str, period: str, transactions=None):
    try:
        hist = yf.Ticker(ticker).history(period=period)
    except Exception:
        st.caption("Could not load price history."); return
    if hist.empty:
        st.caption("No historical data available."); return

    hist.index = pd.to_datetime(hist.index).tz_localize(None)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], name="Price",
        line=dict(color=BLUE, width=2),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>€%{y:,.4f}<extra></extra>"))
    # Subtle fill under the price line
    fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], fill="tozeroy",
        fillcolor="rgba(91,155,213,0.06)", line=dict(width=0),
        showlegend=False, hoverinfo="skip"))

    if transactions:
        start = hist.index.min().date()
        for action, colour, symbol in [("buy",GAIN,"triangle-up"),("sell",LOSS,"triangle-down")]:
            pts = [(t.date, t.price) for t in transactions
                   if t.action == action
                   and datetime.strptime(t.date, "%Y-%m-%d").date() >= start]
            if pts:
                xs, ys = zip(*pts)
                fig.add_trace(go.Scatter(x=list(xs), y=list(ys), mode="markers",
                    name=action.capitalize(),
                    marker=dict(color=colour, size=10, symbol=symbol),
                    hovertemplate=f"<b>{action.upper()}</b><br>%{{x}}<br>€%{{y:,.4f}}<extra></extra>"))

    fig.update_layout(**_chart_layout(height=300))
    fig.update_layout(yaxis=dict(gridcolor="#1e1e1e", tickprefix="€"))
    st.plotly_chart(fig, use_container_width=True)

# ── Add Transaction ───────────────────────────────────────────────────────────
def render_add_transaction():
    st.markdown("## Add Transaction")
    existing = {h.ticker: h for h in portfolio().all_holdings()}

    with st.form("add_txn", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            action = st.selectbox("Action", ["BUY","SELL"])
            ticker = st.text_input("Ticker", placeholder="e.g. AAPL, BTC-USD, SIE.DE").upper()
            if ticker in existing:
                st.info(f"Adding to existing: {existing[ticker].name}")
                name, asset_type = existing[ticker].name, existing[ticker].asset_type
            else:
                name       = st.text_input("Name", placeholder="e.g. Apple Inc.")
                asset_type = st.selectbox("Asset Type", ["stock","crypto","etf"])
        with col2:
            quantity = st.number_input("Quantity",           min_value=0.0001, step=0.0001, format="%.4f")
            price    = st.number_input("Price per unit (€)", min_value=0.0001, step=0.01,   format="%.4f")
            txn_date = st.date_input("Date", value=date.today())

        if st.form_submit_button("Add Transaction", use_container_width=True, type="primary"):
            if not ticker:
                st.error("Please enter a ticker symbol.")
            elif ticker not in existing and not name:
                st.error("Please enter a name for the new holding.")
            else:
                h = existing.get(ticker)
                portfolio().add_transaction(
                    ticker=ticker,
                    name=h.name if h else name,
                    asset_type=h.asset_type if h else asset_type,
                    action=action.lower(), quantity=quantity,
                    price=price, date=str(txn_date),
                )
                invalidate_prices()
                st.session_state.news_cache = {}
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
def _bench_scatter(x, y, name, color, width=2, dash=None, fill=None, fillcolor=None, tmpl=None):
    """Build a Scatter trace — only pass optional kwargs when actually needed."""
    kw = dict(x=x, y=y, name=name, line=dict(color=color, width=width))
    if dash:      kw["line"]["dash"] = dash
    if fill:      kw["fill"] = fill
    if fillcolor: kw["fillcolor"] = fillcolor
    if tmpl:      kw["hovertemplate"] = tmpl
    return go.Scatter(**kw)

def render_benchmark():
    st.markdown("## Benchmark")
    st.caption("Portfolio vs indices · rebased to €100 from your first transaction")

    port       = portfolio()
    start_date = get_portfolio_start_date(port)
    if start_date is None:
        st.info("Add some transactions first."); return

    st.info(f"Benchmarking from: **{start_date.strftime('%d %b %Y')}**")
    selected_indices = st.multiselect("Compare against", list(INDICES.keys()), list(INDICES.keys()))

    if not st.button("Run Benchmark", type="primary"):
        return

    with st.spinner("Building portfolio value series…"):
        port_series = build_portfolio_value_series(port, start_date)
    if port_series is None or port_series.empty:
        st.error("Could not build portfolio series. Check your ticker symbols."); return

    index_series: Dict[str, pd.Series] = {}
    for name in selected_indices:
        with st.spinner(f"Fetching {name}…"):
            s = fetch_index_series(INDICES[name], start_date)
        if s is not None: index_series[name] = s
        else:             st.warning(f"Could not fetch {name}")

    norm_port    = normalise(port_series)
    norm_indices = {n: normalise(s.reindex(port_series.index, method="ffill"))
                    for n, s in index_series.items()}

    def _add_indices(fig, series_dict, dash="dot", suffix="", tmpl_fmt="€%{{y:.2f}}"):
        for name, s in series_dict.items():
            fig.add_trace(_bench_scatter(s.index, s.values, name,
                BENCH_COLOURS.get(name,"#aaa"), 1.5, dash=dash,
                tmpl=f"<b>{name}</b><br>{tmpl_fmt}<extra></extra>"))

    st.divider()

    # Growth of €100
    st.markdown('<p class="section-title">Growth of €100</p>', unsafe_allow_html=True)
    fig_g = go.Figure()
    fig_g.add_trace(_bench_scatter(norm_port.index, norm_port.values, "My Portfolio",
                                   BLUE, 2.5, tmpl="<b>Portfolio</b><br>€%{y:.2f}<extra></extra>"))
    _add_indices(fig_g, norm_indices)
    fig_g.add_hline(y=100, line_color="#333", line_dash="dash", line_width=1)
    fig_g.update_layout(**_chart_layout(height=420))
    fig_g.update_layout(yaxis=dict(gridcolor="#1e1e1e", tickprefix="€"))
    st.plotly_chart(fig_g, use_container_width=True)

    st.divider()

    # Cumulative P&L
    st.markdown('<p class="section-title">Cumulative P&L over Time</p>', unsafe_allow_html=True)
    pnl_s = port_series - port_series.iloc[0]
    fig_pnl = go.Figure()
    fig_pnl.add_trace(_bench_scatter(pnl_s.index, pnl_s.values, "P&L", GAIN,
        fill="tozeroy", fillcolor="rgba(76,175,125,0.08)",
        tmpl="<b>P&L</b><br>€%{y:+,.2f}<extra></extra>"))
    fig_pnl.add_hline(y=0, line_color="#333", line_dash="dash", line_width=1)
    fig_pnl.update_layout(**_chart_layout(height=300))
    fig_pnl.update_layout(yaxis=dict(gridcolor="#1e1e1e", tickprefix="€"))
    st.plotly_chart(fig_pnl, use_container_width=True)

    st.divider()

    # Drawdown
    st.markdown('<p class="section-title">Drawdown</p>', unsafe_allow_html=True)
    fig_dd = go.Figure()
    fig_dd.add_trace(_bench_scatter(compute_drawdown(norm_port).index,
                                    compute_drawdown(norm_port).values,
                                    "My Portfolio", BLUE, fill="tozeroy",
                                    fillcolor="rgba(91,155,213,0.10)",
                                    tmpl="<b>Portfolio</b><br>%{y:.2f}%<extra></extra>"))
    dd_indices = {n: compute_drawdown(s) for n, s in norm_indices.items()}
    _add_indices(fig_dd, dd_indices, tmpl_fmt="%{y:.2f}%")
    fig_dd.update_layout(**_chart_layout(height=320))
    fig_dd.update_layout(yaxis=dict(gridcolor="#1e1e1e", ticksuffix="%"))
    st.plotly_chart(fig_dd, use_container_width=True)

    st.divider()

    # Stats table
    st.markdown('<p class="section-title">Key Statistics</p>', unsafe_allow_html=True)
    STAT_COLS = ["Label","Total Return","Ann. Return","Ann. Volatility",
                 "Sharpe Ratio","Max Drawdown","Best Day","Worst Day","Days"]
    all_stats = [compute_stats(port_series, "My Portfolio")] + [
        compute_stats(s.reindex(port_series.index, method="ffill").dropna(), name)
        for name, s in index_series.items()
    ]
    colour_cols = ["Total Return","Ann. Return","Max Drawdown","Best Day","Worst Day"]
    st.dataframe(
        pd.DataFrame(all_stats)[STAT_COLS].rename(columns={"Label":""}).style
          .applymap(_colour_stat, subset=colour_cols),
        use_container_width=True, hide_index=True)

# ── Covariance ────────────────────────────────────────────────────────────────
def render_covariance():
    st.markdown("## Covariance & Correlation Matrix")
    st.caption("Based on 3 years of weekly returns · sourced from Yahoo Finance")

    holdings = portfolio().all_holdings()
    if len(holdings) < 2:
        st.info("You need at least 2 holdings to compute a covariance matrix."); return

    all_tickers = [h.ticker for h in holdings]
    selected = st.multiselect("Select tickers", all_tickers, all_tickers)
    if len(selected) < 2:
        st.warning("Please select at least 2 tickers."); return
    if not st.button("Run Analysis", type="primary"):
        return

    with st.spinner("Downloading 3 years of weekly data…"):
        try:
            raw = yf.download(selected, period="3y", interval="1wk",
                              progress=False, auto_adjust=True)
        except Exception as e:
            st.error(f"Failed: {e}"); return
    if raw.empty:
        st.error("No data returned."); return

    # Normalise column structure for 1 vs multiple tickers
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=selected[0].upper())
    else:
        close.columns = [c.upper() for c in close.columns]

    returns     = close.pct_change().dropna()
    cov_matrix  = returns.cov()
    corr_matrix = returns.corr()

    st.divider()
    st.markdown('<p class="section-title">Individual Statistics (weekly)</p>',
                unsafe_allow_html=True)
    cols = st.columns(len(selected))
    for i, ticker in enumerate(returns.columns):
        s = returns[ticker].dropna()
        cols[i].metric(ticker, f"Ann. Vol: {s.std()*np.sqrt(52):.1%}",
                       delta=f"Avg weekly: {s.mean():+.3%}",
                       delta_color="normal" if s.mean() >= 0 else "inverse")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p class="section-title">Covariance Matrix</p>', unsafe_allow_html=True)
        _render_heatmap(cov_matrix, ".5f")
    with c2:
        st.markdown('<p class="section-title">Correlation Matrix</p>', unsafe_allow_html=True)
        _render_heatmap(corr_matrix, ".3f", zmin=-1, zmax=1, zmid=0)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p class="section-title">Covariance Table</p>', unsafe_allow_html=True)
        st.dataframe(cov_matrix.style.format("{:.6f}"), use_container_width=True)
    with c2:
        st.markdown('<p class="section-title">Correlation Table</p>', unsafe_allow_html=True)
        st.dataframe(corr_matrix.style.format("{:.4f}").background_gradient(
            cmap="RdYlGn", vmin=-1, vmax=1), use_container_width=True)

def _render_heatmap(matrix, fmt, zmin=None, zmax=None, zmid=None):
    labels = list(matrix.columns)
    fig = go.Figure(go.Heatmap(
        z=matrix.values.tolist(), x=labels, y=labels,
        text=[[format(v, fmt) for v in row] for row in matrix.values],
        texttemplate="%{text}", colorscale="RdYlGn",
        zmin=zmin, zmax=zmax, zmid=zmid, showscale=True,
        hovertemplate="<b>%{y} × %{x}</b><br>%{text}<extra></extra>"))
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, font_color="#cccccc",
                      font_size=11, margin=dict(t=10,b=10,l=10,r=10), height=350)
    st.plotly_chart(fig, use_container_width=True)

# ── Router ────────────────────────────────────────────────────────────────────
def main():
    page = render_sidebar()
    {"Dashboard":        render_dashboard,
     "Holdings":         render_holdings,
     "Add Transaction":  render_add_transaction,
     "Benchmark":        render_benchmark,
     "Covariance Matrix":render_covariance}.get(page, render_dashboard)()

if __name__ == "__main__":
    main()