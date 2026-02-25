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
        holdings = portfolio().all_holdings()
        tickers  = [h.ticker for h in holdings if not h.manual_price]
        if tickers:
            with st.spinner("Fetching live prices…"):
                st.session_state.prices = fetcher().get_prices(tickers)
        # Merge manual price overrides — they always take priority
        for h in holdings:
            if h.manual_price is not None:
                st.session_state.prices[h.ticker] = h.manual_price
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
                                "Benchmark","Covariance Matrix","Snapshot History","Portfolio Analysis"],
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
        manual_tag = "  🔧" if h.manual_price else ""
        with st.expander(f"{h.ticker}  —  {h.name}{manual_tag}"):
            p = prices.get(h.ticker)

            # ── Manual price override ──
            st.markdown('<p class="section-title">Price Override</p>', unsafe_allow_html=True)
            st.caption("Use this if Yahoo Finance doesn't support this ticker.")
            oc1, oc2, oc3 = st.columns([1, 1, 2])
            with oc1:
                manual_val = st.number_input(
                    "Manual price (€)", min_value=0.0, value=float(h.manual_price or 0),
                    format="%.4f", key=f"manual_input_{h.ticker}",
                    help="Set to 0 to remove override and use live price")
            with oc2:
                st.write("")
                st.write("")
                if st.button("Set price", key=f"set_price_{h.ticker}"):
                    override = manual_val if manual_val > 0 else None
                    portfolio().set_manual_price(h.ticker, override)
                    invalidate_prices()
                    label = fmt_cur(override) if override else "live price"
                    st.success(f"✓ Price set to {label}"); st.rerun()

            st.divider()

            # ── Metrics ──
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Avg Cost",      fmt_cur(h.average_cost))
            m2.metric("Current Price", (fmt_cur(p) + (" 🔧" if h.manual_price else "")) if p else "—")
            m3.metric("Market Value",  fmt_cur(h.current_value(p)) if p else "—")
            if p:
                m4.metric("P&L", fmt_cur(h.unrealised_pnl(p)), delta=fmt_pct(h.pnl_percent(p)))

            # ── Price history ──
            period_map = {"1M":"1mo","3M":"3mo","6M":"6mo","1Y":"1y","3Y":"3y","5Y":"5y"}
            period = st.radio("Period", list(period_map.keys()),
                              horizontal=True, key=f"period_{h.ticker}")
            _chart_price_history(h.ticker, period_map[period], h.transactions)

            # ── Editable transaction table with per-row delete ──
            st.markdown('<p class="section-title">Transactions  (click a cell to edit, ✕ to delete)</p>',
                        unsafe_allow_html=True)
            original_df = pd.DataFrame([{"Date":t.date,"Action":t.action.lower(),
                                          "Quantity":t.quantity,"Price":t.price}
                                         for t in h.transactions])
            edited_df = st.data_editor(
                original_df, key=f"editor_{h.ticker}",
                use_container_width=True, hide_index=True,
                # num_rows="dynamic" allows the user to delete rows with the ✕ button
                num_rows="dynamic",
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
                        for _, r in edited_df.dropna(subset=["Date","Action","Quantity","Price"]).iterrows()
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

# ── Snapshot History ──────────────────────────────────────────────────────────
def render_snapshot_history():
    """
    Let the user manually record the portfolio value at a point in time,
    then plot all snapshots as a line chart to show value growth over time.

    Why manual? Because we don't have a database running 24/7 to auto-record.
    The user clicks "Save Snapshot" whenever they want to log the current value.
    """
    st.markdown("## Snapshot History")
    st.caption("Manually record your portfolio value over time to track growth.")

    holdings = portfolio().all_holdings()
    prices   = get_prices()
    tv  = sum(h.current_value(prices[h.ticker]) for h in holdings if prices.get(h.ticker))
    ti  = sum(h.total_invested for h in holdings)

    # ── Save a new snapshot ──
    st.markdown('<p class="section-title">Record Today's Value</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        st.metric("Current Value", fmt_cur(tv))
    with c2:
        note = st.text_input("Note (optional)", placeholder="e.g. After Q1 rebalance")
    with c3:
        st.write("")
        st.write("")
        if st.button("📸  Save Snapshot", type="primary", use_container_width=True):
            if tv == 0:
                st.warning("Portfolio value is €0 — check prices are loaded.")
            else:
                snap = portfolio().add_snapshot(tv, ti, note)
                st.success(f"✓ Snapshot saved: {fmt_cur(snap.total_value)} on {snap.date}")
                st.rerun()

    st.divider()

    snaps = portfolio().snapshots
    if not snaps:
        st.info("No snapshots yet. Click **Save Snapshot** above to record your first one.")
        return

    # ── Value over time chart ──
    st.markdown('<p class="section-title">Portfolio Value Over Time</p>', unsafe_allow_html=True)
    df = pd.DataFrame([{
        "Date":     s.date,
        "Value":    s.total_value,
        "Invested": s.total_invested,
        "P&L":      s.pnl,
    } for s in snaps]).sort_values("Date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Value"], name="Portfolio Value",
        line=dict(color=BLUE, width=2.5), mode="lines+markers",
        hovertemplate="<b>%{x}</b><br>Value: €%{y:,.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Invested"], name="Total Invested",
        line=dict(color="#888", width=1.5, dash="dot"), mode="lines+markers",
        hovertemplate="<b>%{x}</b><br>Invested: €%{y:,.2f}<extra></extra>"))
    fig.update_layout(**_chart_layout("Portfolio Value Over Time", 380))
    fig.update_layout(yaxis=dict(gridcolor="#1e1e1e", tickprefix="€"))
    st.plotly_chart(fig, use_container_width=True)

    # P&L chart
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=df["Date"], y=df["P&L"], name="P&L",
        marker_color=[GAIN if v >= 0 else LOSS for v in df["P&L"]],
        hovertemplate="<b>%{x}</b><br>P&L: €%{y:+,.2f}<extra></extra>"))
    fig2.add_hline(y=0, line_color="#333", line_dash="dash", line_width=1)
    fig2.update_layout(**_chart_layout("Unrealised P&L at Each Snapshot", 260))
    fig2.update_layout(yaxis=dict(gridcolor="#1e1e1e", tickprefix="€"))
    st.plotly_chart(fig2, use_container_width=True)

    # ── Snapshot table with delete buttons ──
    st.divider()
    st.markdown('<p class="section-title">All Snapshots</p>', unsafe_allow_html=True)
    for i, s in enumerate(reversed(snaps)):
        real_idx = len(snaps) - 1 - i   # index into original list
        pnl_col  = GAIN if s.pnl >= 0 else LOSS
        col_date, col_val, col_inv, col_pnl, col_note, col_del = st.columns([1,1,1,1,2,0.5])
        col_date.caption("Date");     col_date.write(s.date)
        col_val.caption("Value");     col_val.write(fmt_cur(s.total_value))
        col_inv.caption("Invested");  col_inv.write(fmt_cur(s.total_invested))
        col_pnl.caption("P&L");
        col_pnl.markdown(f'<span style="color:{pnl_col}">{fmt_cur(s.pnl)} ({fmt_pct(s.pnl_pct)})</span>',
                         unsafe_allow_html=True)
        col_note.caption("Note");     col_note.write(s.note or "—")
        with col_del:
            st.write("")
            if st.button("✕", key=f"del_snap_{real_idx}", help="Delete this snapshot"):
                portfolio().delete_snapshot(real_idx)
                st.rerun()


# ── Router ────────────────────────────────────────────────────────────────────
def main():
    page = render_sidebar()
    {"Dashboard":        render_dashboard,
     "Holdings":         render_holdings,
     "Add Transaction":  render_add_transaction,
     "Benchmark":        render_benchmark,
     "Covariance Matrix":render_covariance,
     "Snapshot History": render_snapshot_history,
     "Portfolio Analysis": render_analysis}.get(page, render_dashboard)()

if __name__ == "__main__":
    main()

# ── Portfolio Analysis ────────────────────────────────────────────────────────
def render_analysis():
    from tracker.analysis import (
        portfolio_weights, concentration_hhi, by_asset_type, by_geography,
        by_sector, fetch_sectors, fetch_return_matrix, correlation_matrix,
        avg_pairwise_correlation, portfolio_volatility,
        apply_stress, PRESET_SCENARIOS,
    )

    st.markdown("## Portfolio Analysis")
    holdings = portfolio().all_holdings()
    prices   = get_prices()

    if len(holdings) < 2:
        st.info("Add at least 2 holdings to run portfolio analysis.")
        return

    weights = portfolio_weights(holdings, prices)
    tickers = list(weights.keys())

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — DIVERSIFICATION
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### 🗂  Diversification")

    hhi = concentration_hhi(weights)
    hhi_label = ("🟢 Well diversified" if hhi < 1_500
                 else "🟡 Moderately concentrated" if hhi < 2_500
                 else "🔴 Highly concentrated")

    c1, c2, c3 = st.columns(3)
    c1.metric("HHI Concentration Score", f"{hhi:,.0f} / 10,000",
              help="Herfindahl–Hirschman Index. Below 1,500 = diversified, above 2,500 = concentrated.")
    c2.metric("Diversification Rating", hhi_label)
    c3.metric("Number of Holdings", str(len(holdings)))

    st.divider()

    # By asset type + geography side by side
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<p class="section-title">By Asset Type</p>', unsafe_allow_html=True)
        df_type = by_asset_type(holdings, prices)
        if not df_type.empty:
            fig = go.Figure(go.Pie(
                labels=df_type["Asset Type"], values=df_type["Value (€)"],
                hole=0.5, marker=dict(colors=PALETTE, line=dict(color=BG, width=2)),
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>€%{value:,.2f}<br>%{percent}<extra></extra>"))
            fig.update_layout(**_chart_layout(height=280))
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_type.style.format({"Value (€)": "€{:,.2f}", "Weight (%)": "{:.2f}%"}),
                         use_container_width=True, hide_index=True)

    with col_r:
        st.markdown('<p class="section-title">By Geography</p>', unsafe_allow_html=True)
        df_geo = by_geography(holdings, prices)
        if not df_geo.empty:
            fig2 = go.Figure(go.Pie(
                labels=df_geo["Region"], values=df_geo["Value (€)"],
                hole=0.5,
                marker=dict(colors=PALETTE[2:] + PALETTE[:2], line=dict(color=BG, width=2)),
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>€%{value:,.2f}<br>%{percent}<extra></extra>"))
            fig2.update_layout(**_chart_layout(height=280))
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
            st.dataframe(df_geo.style.format({"Value (€)": "€{:,.2f}", "Weight (%)": "{:.2f}%"}),
                         use_container_width=True, hide_index=True)

    # Sector breakdown — requires yfinance calls so put behind a button
    st.divider()
    st.markdown('<p class="section-title">By Sector</p>', unsafe_allow_html=True)
    st.caption("Sector data is fetched from Yahoo Finance — may not be available for all tickers.")

    if "sector_cache" not in st.session_state:
        st.session_state.sector_cache = {}

    if st.button("Fetch Sector Data", key="fetch_sectors"):
        with st.spinner("Fetching sector info…"):
            st.session_state.sector_cache = fetch_sectors(tickers)

    if st.session_state.sector_cache:
        df_sec = by_sector(holdings, prices, st.session_state.sector_cache)
        if not df_sec.empty:
            fig3 = go.Figure(go.Bar(
                x=df_sec["Sector"], y=df_sec["Weight (%)"],
                marker_color=PALETTE * 3,
                hovertemplate="<b>%{x}</b><br>%{y:.2f}%<extra></extra>"))
            fig3.update_layout(**_chart_layout(height=280))
            fig3.update_layout(yaxis=dict(gridcolor="#1e1e1e", ticksuffix="%"),
                               xaxis=dict(gridcolor="#1e1e1e"))
            st.plotly_chart(fig3, use_container_width=True)
            st.dataframe(df_sec.style.format({"Value (€)": "€{:,.2f}", "Weight (%)": "{:.2f}%"}),
                         use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — CORRELATION
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown("### 📊  Correlation & Volatility")

    period_choice = st.radio("Data period", ["3M","6M","1Y","2Y","3Y"],
                             index=2, horizontal=True, key="corr_period")
    period_map    = {"3M":"3mo","6M":"6mo","1Y":"1y","2Y":"2y","3Y":"3y"}

    if st.button("Run Correlation Analysis", type="primary", key="run_corr"):
        with st.spinner("Downloading price data…"):
            returns = fetch_return_matrix(tickers, period_map[period_choice])

        if returns is None or returns.empty:
            st.error("Could not fetch price data.")
        else:
            corr  = correlation_matrix(returns)
            avg_r = avg_pairwise_correlation(corr)
            vol   = portfolio_volatility(returns, weights)

            # Metrics
            r_label = ("🟢 Low" if avg_r < 0.3 else
                       "🟡 Moderate" if avg_r < 0.6 else "🔴 High")
            m1, m2, m3 = st.columns(3)
            m1.metric("Avg Pairwise Correlation", f"{avg_r:.2f}",
                      help="Average correlation between all pairs. Lower = more diversification benefit.")
            m2.metric("Correlation Level", r_label)
            m3.metric("Portfolio Ann. Volatility", f"{vol:.1%}",
                      help="Annualised volatility using the full covariance matrix.")

            st.divider()

            # Correlation heatmap
            st.markdown('<p class="section-title">Correlation Matrix</p>', unsafe_allow_html=True)
            labels = list(corr.columns)
            fig_c = go.Figure(go.Heatmap(
                z=corr.values.tolist(), x=labels, y=labels,
                text=[[f"{v:.2f}" for v in row] for row in corr.values],
                texttemplate="%{text}", colorscale="RdYlGn",
                zmin=-1, zmax=1, zmid=0, showscale=True,
                hovertemplate="<b>%{y} × %{x}</b><br>Correlation: %{text}<extra></extra>"))
            fig_c.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, font_color="#cccccc",
                                font_size=11, margin=dict(t=10,b=10,l=10,r=10), height=400)
            st.plotly_chart(fig_c, use_container_width=True)

            # Individual volatility bars
            st.markdown('<p class="section-title">Individual Annualised Volatility</p>',
                        unsafe_allow_html=True)
            vols = {t: float(returns[t].std() * np.sqrt(252)) for t in returns.columns}
            fig_v = go.Figure(go.Bar(
                x=list(vols.keys()), y=[v * 100 for v in vols.values()],
                marker_color=PALETTE * 5,
                hovertemplate="<b>%{x}</b><br>Volatility: %{y:.1f}%<extra></extra>"))
            fig_v.update_layout(**_chart_layout(height=280))
            fig_v.update_layout(yaxis=dict(gridcolor="#1e1e1e", ticksuffix="%"))
            st.plotly_chart(fig_v, use_container_width=True)

            # Rolling correlation (first two tickers)
            if len(tickers) >= 2:
                st.markdown('<p class="section-title">Rolling 30-Day Correlation '
                            f'({tickers[0]} vs {tickers[1]})</p>', unsafe_allow_html=True)
                rolling = (returns[tickers[0].upper()]
                           .rolling(30)
                           .corr(returns[tickers[1].upper()])
                           .dropna())
                fig_r = go.Figure(go.Scatter(
                    x=rolling.index, y=rolling.values,
                    line=dict(color=BLUE, width=1.5),
                    hovertemplate="%{x|%d %b %Y}<br>Corr: %{y:.2f}<extra></extra>"))
                fig_r.add_hline(y=0,   line_color="#333", line_dash="dash", line_width=1)
                fig_r.add_hline(y=0.7, line_color=LOSS,  line_dash="dot",  line_width=1,
                                annotation_text="High correlation (0.7)")
                fig_r.update_layout(**_chart_layout(height=260))
                fig_r.update_layout(yaxis=dict(gridcolor="#1e1e1e", range=[-1.1, 1.1]))
                st.plotly_chart(fig_r, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — STRESS TESTING
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown("### ⚡  Stress Testing")

    tv = sum(h.current_value(prices[h.ticker]) for h in holdings if prices.get(h.ticker))
    ti = sum(h.total_invested for h in holdings)

    tab_preset, tab_custom = st.tabs(["Preset Scenarios", "Custom Scenario"])

    # ── Preset tab ──
    with tab_preset:
        scenario_name = st.selectbox("Choose scenario", list(PRESET_SCENARIOS.keys()))
        scenario      = PRESET_SCENARIOS[scenario_name]
        st.caption(scenario["description"])

        rows = apply_stress(holdings, prices, scenario["shocks"])
        if not rows:
            st.warning("No priced holdings to stress test.")
        else:
            df_stress = pd.DataFrame(rows)
            total_after  = df_stress["Value After"].sum()
            total_impact = total_after - tv

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Current Value",  fmt_cur(tv))
            m2.metric("Value After",    fmt_cur(total_after))
            m3.metric("Total Impact",   fmt_cur(total_impact),
                      delta=fmt_pct(total_impact / tv * 100 if tv else 0))
            m4.metric("Scenario",       scenario_name)

            # Impact bar chart
            fig_s = go.Figure(go.Bar(
                x=df_stress["Ticker"], y=df_stress["Impact (€)"],
                marker_color=[GAIN if v >= 0 else LOSS for v in df_stress["Impact (€)"]],
                hovertemplate="<b>%{x}</b><br>Impact: €%{y:+,.2f}<extra></extra>"))
            fig_s.add_hline(y=0, line_color="#333", line_dash="dash", line_width=1)
            fig_s.update_layout(**_chart_layout(height=300))
            fig_s.update_layout(yaxis=dict(gridcolor="#1e1e1e", tickprefix="€"))
            st.plotly_chart(fig_s, use_container_width=True)

            # Table
            st.dataframe(
                df_stress[["Ticker","Asset Type","Value Now","Value After","Impact (€)","Impact (%)"]].style
                  .applymap(lambda v: f"color:{GAIN}" if v > 0 else (f"color:{LOSS}" if v < 0 else ""),
                            subset=["Impact (€)", "Impact (%)"])
                  .format({"Value Now": "€{:,.2f}", "Value After": "€{:,.2f}",
                           "Impact (€)": "€{:+,.2f}", "Impact (%)": "{:+.1f}%"}),
                use_container_width=True, hide_index=True)

    # ── Custom tab ──
    with tab_custom:
        st.markdown('<p class="section-title">Set shock per asset type</p>',
                    unsafe_allow_html=True)
        st.caption("Enter the expected % change for each asset type in your scenario.")

        asset_types = list({h.asset_type.lower() for h in holdings})
        custom_shocks = {}
        cols = st.columns(len(asset_types))
        for i, at in enumerate(asset_types):
            with cols[i]:
                val = st.number_input(f"{at.upper()} (%)", min_value=-100.0,
                                      max_value=500.0, value=0.0, step=1.0,
                                      key=f"shock_{at}")
                custom_shocks[at] = val / 100

        custom_name = st.text_input("Scenario name", placeholder="e.g. My bear case")

        if st.button("Run Custom Scenario", type="primary", key="run_custom"):
            rows_c = apply_stress(holdings, prices, custom_shocks)
            if not rows_c:
                st.warning("No priced holdings to stress test.")
            else:
                df_c        = pd.DataFrame(rows_c)
                total_c     = df_c["Value After"].sum()
                impact_c    = total_c - tv

                m1, m2, m3 = st.columns(3)
                m1.metric("Current Value", fmt_cur(tv))
                m2.metric("Value After",   fmt_cur(total_c))
                m3.metric("Total Impact",  fmt_cur(impact_c),
                          delta=fmt_pct(impact_c / tv * 100 if tv else 0))

                fig_cs = go.Figure(go.Bar(
                    x=df_c["Ticker"], y=df_c["Impact (€)"],
                    marker_color=[GAIN if v >= 0 else LOSS for v in df_c["Impact (€)"]],
                    hovertemplate="<b>%{x}</b><br>Impact: €%{y:+,.2f}<extra></extra>"))
                fig_cs.add_hline(y=0, line_color="#333", line_dash="dash", line_width=1)
                fig_cs.update_layout(**_chart_layout(
                    title=custom_name or "Custom Scenario", height=300))
                fig_cs.update_layout(yaxis=dict(gridcolor="#1e1e1e", tickprefix="€"))
                st.plotly_chart(fig_cs, use_container_width=True)

                st.dataframe(
                    df_c[["Ticker","Asset Type","Value Now","Value After","Impact (€)","Impact (%)"]].style
                      .applymap(lambda v: f"color:{GAIN}" if v > 0 else (f"color:{LOSS}" if v < 0 else ""),
                                subset=["Impact (€)", "Impact (%)"])
                      .format({"Value Now": "€{:,.2f}", "Value After": "€{:,.2f}",
                               "Impact (€)": "€{:+,.2f}", "Impact (%)": "{:+.1f}%"}),
                    use_container_width=True, hide_index=True)
