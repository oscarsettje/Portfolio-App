"""
app.py  —  Portfolio Tracker  |  streamlit run app.py
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
from tracker.models import Holding, Snapshot, Transaction
from tracker.portfolio import Portfolio
from tracker.db import Database
from tracker.prices import PriceFetcher, _close_from_download
import tracker.exporter as exporter

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Portfolio Tracker", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

# ── Colours ───────────────────────────────────────────────────────────────────
GAIN    = "#4caf7d"
LOSS    = "#e05c5c"
BLUE    = "#5b9bd5"
BG      = "#0f0f0f"
PALETTE = ["#5b9bd5","#4caf7d","#e8a838","#b07fd4",
           "#e05c5c","#4db6ac","#f06292","#a1887f"]
BENCH_COLOURS = {"MSCI World":"#e8a838","S&P 500":"#b07fd4",
                 "NASDAQ 100":"#4db6ac","MSCI Emerging Mkts":"#f06292"}

st.markdown("""
<style>
  [data-testid="metric-container"]{background:#1a1a1a;border:1px solid #2a2a2a;
      border-radius:8px;padding:14px 18px}
  [data-testid="stMetricValue"]{font-size:1.35rem}
  [data-testid="stMetricDelta"]{font-size:0.85rem}
  [data-testid="stDataFrame"]{border-radius:8px;overflow:hidden}
  [data-testid="stSidebar"]{background:#111}
  .block-container{padding-top:1.5rem}
  .section-title{font-size:.75rem;font-weight:600;letter-spacing:.1em;
      text-transform:uppercase;color:#555;margin:1.5rem 0 .5rem}
  .news-card{background:#141414;border:1px solid #222;border-radius:8px;
      padding:12px 16px;margin-bottom:10px}
  .news-card a{color:#5b9bd5;text-decoration:none;font-weight:500}
  .news-card a:hover{text-decoration:underline}
  .news-meta{font-size:.75rem;color:#555;margin-top:4px}
  .news-ticker{display:inline-block;background:#1e2a38;color:#5b9bd5;
      font-size:.7rem;font-weight:700;border-radius:4px;padding:1px 6px;margin-right:6px}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "db"           not in st.session_state: st.session_state.db           = Database()
if "fetcher"      not in st.session_state: st.session_state.fetcher      = PriceFetcher(db=st.session_state.db)
if "portfolio"    not in st.session_state: st.session_state.portfolio    = Portfolio(db=st.session_state.db)
if "prices"       not in st.session_state: st.session_state.prices       = {}
if "news_cache"   not in st.session_state: st.session_state.news_cache   = {}
if "sector_cache" not in st.session_state: st.session_state.sector_cache = {}
if "stale_prices" not in st.session_state: st.session_state.stale_prices = set()

def portfolio() -> Portfolio:    return st.session_state.portfolio
def fetcher()   -> PriceFetcher: return st.session_state.fetcher
def db()        -> Database:     return st.session_state.db

# ── Prices ────────────────────────────────────────────────────────────────────
def get_prices() -> Dict[str, Optional[float]]:
    if not st.session_state.prices:
        holdings = portfolio().all_holdings()
        tickers  = [h.ticker for h in holdings if not h.manual_price]
        if tickers:
            with st.spinner("Fetching live prices…"):
                st.session_state.prices = fetcher().get_prices(tickers)
            # Track which tickers are using cached (stale) prices
            fresh = fetcher()._cache
            st.session_state.stale_prices = {
                t for t in tickers
                if t not in fresh or fresh.get(t) == fetcher()._disk.get(t)
                   and t in fetcher()._disk
            }
        for h in holdings:
            if h.manual_price is not None:
                st.session_state.prices[h.ticker] = h.manual_price
    return st.session_state.prices

def invalidate_prices():
    fetcher().clear_cache()
    st.session_state.prices = {}

# ── Helpers ───────────────────────────────────────────────────────────────────
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

def _heatmap_fig(matrix: pd.DataFrame, fmt: str,
                 zmin=None, zmax=None, zmid=None, height=350) -> go.Figure:
    labels = list(matrix.columns)
    fig = go.Figure(go.Heatmap(
        z=matrix.values.tolist(), x=labels, y=labels,
        text=[[format(v, fmt) for v in row] for row in matrix.values],
        texttemplate="%{text}", colorscale="RdYlGn",
        zmin=zmin, zmax=zmax, zmid=zmid, showscale=True,
        hovertemplate="<b>%{y} × %{x}</b><br>%{text}<extra></extra>"))
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, font_color="#cccccc",
                      font_size=11, margin=dict(t=10,b=10,l=10,r=10), height=height)
    return fig

def _section(title: str):
    st.markdown(f'<p class="section-title">{title}</p>', unsafe_allow_html=True)

# ── News ──────────────────────────────────────────────────────────────────────
def _fetch_news(tickers: List[str]) -> List[dict]:
    cache, all_news = st.session_state.news_cache, []
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
    _section("Latest News")
    col, _ = st.columns([1, 5])
    with col:
        if st.button("↺  Refresh News", key="refresh_news"):
            st.session_state.news_cache = {}; st.rerun()
    with st.spinner("Loading news…"):
        articles = _fetch_news(tickers)
    if not articles:
        st.caption("No news found."); return
    for a in articles:
        try:    pub = datetime.fromtimestamp(a.get("providerPublishTime",0)).strftime("%d %b %Y  %H:%M")
        except: pub = ""
        st.markdown(f"""<div class="news-card">
  <span class="news-ticker">{a.get("_ticker","")}</span>
  <a href="{a.get("link","#")}" target="_blank">{a.get("title","")}</a>
  <div class="news-meta">{a.get("publisher","")}  ·  {pub}</div>
</div>""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
PAGES = ["Dashboard","Holdings","Add Transaction",
         "Benchmark","Portfolio Analysis","Quant Metrics","Tax & Income","Snapshot History"]

def render_sidebar():
    with st.sidebar:
        st.markdown("## 📈 Portfolio Tracker")
        st.divider()
        page = st.radio("Nav", PAGES, label_visibility="collapsed")
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
        stale = st.session_state.get("stale_prices", set())
        if stale:
            st.warning(f"⚠️ Cached prices in use for: {', '.join(sorted(stale))}\n\nYahoo Finance may be rate-limiting. Try refreshing later.")
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
    with col_r: _chart_pnl_bars(holdings, prices)
    st.divider()
    _chart_value(holdings, prices)
    st.divider()
    _render_news([h.ticker for h in holdings])


def _chart_allocation(holdings, prices):
    labels, values, colours = [], [], []
    for i, h in enumerate(holdings):
        if (p := prices.get(h.ticker)) is None: continue
        labels.append(h.ticker)
        values.append(h.current_value(p))
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


def _chart_pnl_bars(holdings, prices):
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
    _section("Export")
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
    _section("Holding Detail")
    for h in holdings:
        tag = "  🔧" if h.manual_price else ""
        with st.expander(f"{h.ticker}  —  {h.name}{tag}"):
            p = prices.get(h.ticker)

            # Manual price override
            _section("Price Override")
            st.caption("Use this if Yahoo Finance doesn't support this ticker.")
            oc1, oc2, _ = st.columns([1, 1, 2])
            with oc1:
                manual_val = st.number_input("Manual price (€)", min_value=0.0,
                    value=float(h.manual_price or 0), format="%.4f",
                    key=f"manual_input_{h.ticker}",
                    help="Set to 0 to remove override and use live price")
            with oc2:
                st.write(""); st.write("")
                if st.button("Set price", key=f"set_price_{h.ticker}"):
                    override = manual_val if manual_val > 0 else None
                    portfolio().set_manual_price(h.ticker, override)
                    invalidate_prices()
                    st.success(f"✓ Set to {fmt_cur(override) if override else 'live price'}")
                    st.rerun()

            st.divider()
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Avg Cost",      fmt_cur(h.average_cost))
            m2.metric("Current Price", (fmt_cur(p) + (" 🔧" if h.manual_price else "")) if p else "—")
            m3.metric("Market Value",  fmt_cur(h.current_value(p)) if p else "—")
            if p:
                m4.metric("P&L", fmt_cur(h.unrealised_pnl(p)), delta=fmt_pct(h.pnl_percent(p)))

            # Price history chart
            period_map = {"1M":"1mo","3M":"3mo","6M":"6mo","1Y":"1y","3Y":"3y","5Y":"5y"}
            period = st.radio("Period", list(period_map.keys()),
                              horizontal=True, key=f"period_{h.ticker}")
            _chart_price_history(h.ticker, period_map[period], h.transactions)

            # Editable transactions
            _section("Transactions  (click a cell to edit, ✕ to delete a row)")
            original_df = pd.DataFrame([{"Date":t.date,"Action":t.action.lower(),
                                          "Quantity":t.quantity,"Price":t.price,
                                          "Commission":t.commission}
                                         for t in h.transactions])
            edited_df = st.data_editor(
                original_df, key=f"editor_{h.ticker}",
                use_container_width=True, hide_index=True, num_rows="dynamic",
                column_config={
                    "Date":       st.column_config.TextColumn("Date", help="YYYY-MM-DD", width="small"),
                    "Action":     st.column_config.SelectboxColumn("Action", options=["buy","sell"], width="small"),
                    "Quantity":   st.column_config.NumberColumn("Quantity", min_value=0.0001, format="%.4f", width="small"),
                    "Price":      st.column_config.NumberColumn("Price (€)", min_value=0.0001, format="€%.4f", width="small"),
                    "Commission": st.column_config.NumberColumn("Commission (€)", min_value=0.0, format="€%.2f", width="small"),
                })
            if not edited_df.equals(original_df):
                if st.button("💾  Save changes", key=f"save_{h.ticker}", type="primary"):
                    new_txns = [
                        Transaction(date=str(r["Date"]).strip(),
                                    action=str(r["Action"]).lower().strip(),
                                    quantity=float(r["Quantity"]),
                                    price=float(r["Price"]),
                                    commission=float(r.get("Commission", 0) or 0))
                        for _, r in edited_df.dropna(subset=["Date","Action","Quantity","Price"]).iterrows()
                    ]
                    portfolio().replace_transactions(h.ticker, new_txns)
                    invalidate_prices()
                    st.success("✓ Saved"); st.rerun()

            col_del, _ = st.columns([1, 4])
            with col_del:
                if st.button(f"🗑  Remove {h.ticker}", key=f"remove_{h.ticker}"):
                    portfolio().remove_holding(h.ticker)
                    invalidate_prices(); st.rerun()


def _chart_price_history(ticker: str, period: str, transactions=None):
    try:
        hist = yf.Ticker(ticker).history(period=period)
    except Exception:
        st.caption("Could not load price history."); return
    if hist.empty:
        st.caption("No historical data available."); return

    # Handle MultiIndex from newer yfinance
    if isinstance(hist.columns, pd.MultiIndex):
        hist = hist.xs(ticker.upper(), axis=1, level=1) if ticker.upper() in hist.columns.get_level_values(1) else hist
    hist.index = pd.to_datetime(hist.index).tz_localize(None)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], name="Price",
        line=dict(color=BLUE, width=2),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>€%{y:,.4f}<extra></extra>"))
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
                st.info(f"Adding to existing holding: {existing[ticker].name}")
                name, asset_type = existing[ticker].name, existing[ticker].asset_type
            else:
                name       = st.text_input("Name", placeholder="e.g. Apple Inc.")
                asset_type = st.selectbox("Asset Type", ["stock","crypto","etf"])
        with col2:
            quantity   = st.number_input("Quantity",           min_value=0.0001, step=0.0001, format="%.4f")
            price      = st.number_input("Price per unit (€)", min_value=0.0001, step=0.01,   format="%.4f")
            commission = st.number_input("Broker commission (€)", min_value=0.0, step=0.01, format="%.2f",
                                         help="Fixed fee charged by your broker for this trade")
            txn_date   = st.date_input("Date", value=date.today())

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
                    price=price, date=str(txn_date), commission=commission,
                )
                invalidate_prices()
                st.session_state.news_cache = {}
                st.success(f"✓ {action} recorded for {ticker}")
                st.rerun()

    with st.expander("📖  Ticker format guide"):
        st.markdown("""
| Asset | Format | Example |
|---|---|---|
| US Stocks / ETFs | Plain ticker | `AAPL`, `MSFT`, `SPY` |
| German stocks | Ticker + `.DE` | `SIE.DE`, `BMW.DE` |
| Dutch stocks | Ticker + `.AS` | `ASML.AS` |
| French stocks | Ticker + `.PA` | `MC.PA` |
| UK stocks | Ticker + `.L` | `SHEL.L` |
| Crypto | Ticker + `-USD` | `BTC-USD`, `ETH-USD` |
        """)

# ── Benchmark ─────────────────────────────────────────────────────────────────
def render_benchmark():
    st.markdown("## Benchmark")
    st.caption("Portfolio vs indices · rebased to €100 from your first transaction")

    port       = portfolio()
    start_date = get_portfolio_start_date(port)
    if start_date is None:
        st.info("Add some transactions first."); return

    st.info(f"Benchmarking from **{start_date.strftime('%d %b %Y')}**")
    selected = st.multiselect("Compare against", list(INDICES.keys()), list(INDICES.keys()))
    if not st.button("Run Benchmark", type="primary"):
        return

    with st.spinner("Building portfolio value series…"):
        port_series = build_portfolio_value_series(port, start_date)
    if port_series is None or port_series.empty:
        st.error("Could not build portfolio series. Check your ticker symbols."); return

    index_series: Dict[str, pd.Series] = {}
    for name in selected:
        with st.spinner(f"Fetching {name}…"):
            s = fetch_index_series(INDICES[name], start_date)
        if s is not None: index_series[name] = s
        else:             st.warning(f"Could not fetch {name}")

    norm_port    = normalise(port_series)
    norm_indices = {n: normalise(s.reindex(port_series.index, method="ffill"))
                    for n, s in index_series.items()}

    def _scatter(x, y, name, color, width=2, dash=None, fill=None, fillcolor=None, tmpl=None):
        kw = dict(x=x, y=y, name=name, line=dict(color=color, width=width))
        if dash:      kw["line"]["dash"] = dash
        if fill:      kw["fill"] = fill
        if fillcolor: kw["fillcolor"] = fillcolor
        if tmpl:      kw["hovertemplate"] = tmpl
        return go.Scatter(**kw)

    def _add_index_traces(fig, series_dict, tmpl_fmt="%{y:.2f}"):
        for n, s in series_dict.items():
            fig.add_trace(_scatter(s.index, s.values, n, BENCH_COLOURS.get(n,"#aaa"),
                1.5, dash="dot", tmpl=f"<b>{n}</b><br>{tmpl_fmt}<extra></extra>"))

    st.divider()
    _section("Growth of €100")
    fig_g = go.Figure()
    fig_g.add_trace(_scatter(norm_port.index, norm_port.values, "My Portfolio",
                             BLUE, 2.5, tmpl="<b>Portfolio</b><br>€%{y:.2f}<extra></extra>"))
    _add_index_traces(fig_g, norm_indices, "€%{y:.2f}")
    fig_g.add_hline(y=100, line_color="#333", line_dash="dash", line_width=1)
    fig_g.update_layout(**_chart_layout(height=420))
    fig_g.update_layout(yaxis=dict(gridcolor="#1e1e1e", tickprefix="€"))
    st.plotly_chart(fig_g, use_container_width=True)

    st.divider()
    _section("Cumulative P&L over Time")
    pnl_s = port_series - port_series.iloc[0]
    fig_pnl = go.Figure()
    fig_pnl.add_trace(_scatter(pnl_s.index, pnl_s.values, "P&L", GAIN,
        fill="tozeroy", fillcolor="rgba(76,175,125,0.08)",
        tmpl="<b>P&L</b><br>€%{y:+,.2f}<extra></extra>"))
    fig_pnl.add_hline(y=0, line_color="#333", line_dash="dash", line_width=1)
    fig_pnl.update_layout(**_chart_layout(height=300))
    fig_pnl.update_layout(yaxis=dict(gridcolor="#1e1e1e", tickprefix="€"))
    st.plotly_chart(fig_pnl, use_container_width=True)

    st.divider()
    _section("Drawdown")
    fig_dd = go.Figure()
    dd = compute_drawdown(norm_port)
    fig_dd.add_trace(_scatter(dd.index, dd.values, "My Portfolio", BLUE,
        fill="tozeroy", fillcolor="rgba(91,155,213,0.10)",
        tmpl="<b>Portfolio</b><br>%{y:.2f}%<extra></extra>"))
    _add_index_traces(fig_dd, {n: compute_drawdown(s) for n,s in norm_indices.items()}, "%{y:.2f}%")
    fig_dd.update_layout(**_chart_layout(height=320))
    fig_dd.update_layout(yaxis=dict(gridcolor="#1e1e1e", ticksuffix="%"))
    st.plotly_chart(fig_dd, use_container_width=True)

    st.divider()
    _section("Key Statistics")
    STAT_COLS = ["Label","Total Return","Ann. Return","Ann. Volatility",
                 "Sharpe Ratio","Max Drawdown","Best Day","Worst Day","Days"]
    all_stats = [compute_stats(port_series, "My Portfolio")] + [
        compute_stats(s.reindex(port_series.index, method="ffill").dropna(), n)
        for n, s in index_series.items()
    ]
    st.dataframe(
        pd.DataFrame(all_stats)[STAT_COLS].rename(columns={"Label":""}).style
          .applymap(_colour_stat, subset=["Total Return","Ann. Return",
                                          "Max Drawdown","Best Day","Worst Day"]),
        use_container_width=True, hide_index=True)

# ── Portfolio Analysis ────────────────────────────────────────────────────────
def render_analysis():
    from tracker.analysis import (
        portfolio_weights, concentration_hhi, by_asset_type, by_geography,
        by_sector, fetch_sectors, fetch_return_matrix,
        avg_pairwise_correlation, portfolio_volatility,
        apply_stress, PRESET_SCENARIOS,
    )

    st.markdown("## Portfolio Analysis")
    holdings = portfolio().all_holdings()
    prices   = get_prices()
    if len(holdings) < 2:
        st.info("Add at least 2 holdings to run portfolio analysis."); return

    weights = portfolio_weights(holdings, prices)
    tickers = list(weights.keys())

    # ── Section 1: Diversification ──
    st.markdown("### 🗂  Diversification")
    hhi = concentration_hhi(weights)
    rating = ("🟢 Well diversified" if hhi < 1_500
              else "🟡 Moderately concentrated" if hhi < 2_500
              else "🔴 Highly concentrated")
    c1,c2,c3 = st.columns(3)
    c1.metric("HHI Concentration", f"{hhi:,.0f} / 10,000",
              help="Below 1,500 = diversified, above 2,500 = concentrated")
    c2.metric("Rating", rating)
    c3.metric("Holdings", str(len(holdings)))
    st.divider()

    col_l, col_r = st.columns(2)
    for col, fn, key, label_col in [
        (col_l, by_asset_type, "Asset Type", "Asset Type"),
        (col_r, by_geography,  "Region",     "Region"),
    ]:
        with col:
            _section(f"By {label_col}")
            df = fn(holdings, prices)
            if not df.empty:
                fig = go.Figure(go.Pie(labels=df[label_col], values=df["Value (€)"],
                    hole=0.5, marker=dict(colors=PALETTE*3, line=dict(color=BG,width=2)),
                    textinfo="label+percent",
                    hovertemplate="<b>%{label}</b><br>€%{value:,.2f}<br>%{percent}<extra></extra>"))
                fig.update_layout(**_chart_layout(height=280))
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df.style.format({"Value (€)":"€{:,.2f}","Weight (%)":"{:.2f}%"}),
                             use_container_width=True, hide_index=True)

    st.divider()
    _section("By Sector")
    st.caption("Fetched from Yahoo Finance — may not be available for all tickers.")
    if st.button("Fetch Sector Data", key="fetch_sectors"):
        with st.spinner("Fetching sector info…"):
            st.session_state.sector_cache = fetch_sectors(tickers)
    if st.session_state.sector_cache:
        df_sec = by_sector(holdings, prices, st.session_state.sector_cache)
        if not df_sec.empty:
            fig = go.Figure(go.Bar(x=df_sec["Sector"], y=df_sec["Weight (%)"],
                marker_color=PALETTE*3,
                hovertemplate="<b>%{x}</b><br>%{y:.2f}%<extra></extra>"))
            fig.update_layout(**_chart_layout(height=280))
            fig.update_layout(yaxis=dict(gridcolor="#1e1e1e", ticksuffix="%"))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_sec.style.format({"Value (€)":"€{:,.2f}","Weight (%)":"{:.2f}%"}),
                         use_container_width=True, hide_index=True)

    # ── Section 2: Correlation ──
    st.divider()
    st.markdown("### 📊  Correlation & Volatility")
    period_map = {"3M":"3mo","6M":"6mo","1Y":"1y","2Y":"2y","3Y":"3y"}
    period = st.radio("Data period", list(period_map.keys()), index=2,
                      horizontal=True, key="corr_period")

    if st.button("Run Correlation Analysis", type="primary", key="run_corr"):
        with st.spinner("Downloading price data…"):
            returns = fetch_return_matrix(tickers, period_map[period])
        if returns is None or returns.empty:
            st.error("Could not fetch price data.")
        else:
            corr  = returns.corr()
            avg_r = avg_pairwise_correlation(corr)
            vol   = portfolio_volatility(returns, weights)
            r_label = ("🟢 Low" if avg_r < 0.3 else "🟡 Moderate" if avg_r < 0.6 else "🔴 High")

            m1,m2,m3 = st.columns(3)
            m1.metric("Avg Pairwise Correlation", f"{avg_r:.2f}",
                      help="Lower = more diversification benefit")
            m2.metric("Correlation Level", r_label)
            m3.metric("Portfolio Ann. Volatility", f"{vol:.1%}",
                      help="Annualised, accounts for correlations between positions")

            st.divider()
            _section("Correlation Matrix")
            st.plotly_chart(_heatmap_fig(corr, ".2f", zmin=-1, zmax=1, zmid=0, height=400),
                            use_container_width=True)

            _section("Individual Annualised Volatility")
            vols = {t: float(returns[t].std() * np.sqrt(252)) for t in returns.columns}
            fig_v = go.Figure(go.Bar(x=list(vols.keys()), y=[v*100 for v in vols.values()],
                marker_color=PALETTE*5,
                hovertemplate="<b>%{x}</b><br>Volatility: %{y:.1f}%<extra></extra>"))
            fig_v.update_layout(**_chart_layout(height=280))
            fig_v.update_layout(yaxis=dict(gridcolor="#1e1e1e", ticksuffix="%"))
            st.plotly_chart(fig_v, use_container_width=True)

            if len(tickers) >= 2:
                t1, t2 = tickers[0].upper(), tickers[1].upper()
                _section(f"Rolling 30-Day Correlation  ({t1} vs {t2})")
                rolling = returns[t1].rolling(30).corr(returns[t2]).dropna()
                fig_r = go.Figure(go.Scatter(x=rolling.index, y=rolling.values,
                    line=dict(color=BLUE, width=1.5),
                    hovertemplate="%{x|%d %b %Y}<br>Corr: %{y:.2f}<extra></extra>"))
                fig_r.add_hline(y=0,   line_color="#333", line_dash="dash", line_width=1)
                fig_r.add_hline(y=0.7, line_color=LOSS,  line_dash="dot",  line_width=1,
                                annotation_text="High (0.7)")
                fig_r.update_layout(**_chart_layout(height=260))
                fig_r.update_layout(yaxis=dict(gridcolor="#1e1e1e", range=[-1.1, 1.1]))
                st.plotly_chart(fig_r, use_container_width=True)

    # ── Section 3: Stress testing ──
    st.divider()
    st.markdown("### ⚡  Stress Testing")
    tv = sum(h.current_value(prices[h.ticker]) for h in holdings if prices.get(h.ticker))
    tab_preset, tab_custom = st.tabs(["Preset Scenarios", "Custom Scenario"])

    def _show_stress_results(rows, total_before, title=""):
        if not rows:
            st.warning("No priced holdings to stress test."); return
        df      = pd.DataFrame(rows)
        t_after = df["Value After"].sum()
        impact  = t_after - total_before
        m1,m2,m3 = st.columns(3)
        m1.metric("Current Value", fmt_cur(total_before))
        m2.metric("Value After",   fmt_cur(t_after))
        m3.metric("Total Impact",  fmt_cur(impact),
                  delta=fmt_pct(impact/total_before*100 if total_before else 0))
        fig = go.Figure(go.Bar(x=df["Ticker"], y=df["Impact (€)"],
            marker_color=[GAIN if v>=0 else LOSS for v in df["Impact (€)"]],
            hovertemplate="<b>%{x}</b><br>Impact: €%{y:+,.2f}<extra></extra>"))
        fig.add_hline(y=0, line_color="#333", line_dash="dash", line_width=1)
        fig.update_layout(**_chart_layout(title=title, height=300))
        fig.update_layout(yaxis=dict(gridcolor="#1e1e1e", tickprefix="€"))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            df[["Ticker","Asset Type","Value Now","Value After","Impact (€)","Impact (%)"]].style
              .applymap(lambda v: f"color:{GAIN}" if v>0 else (f"color:{LOSS}" if v<0 else ""),
                        subset=["Impact (€)","Impact (%)"])
              .format({"Value Now":"€{:,.2f}","Value After":"€{:,.2f}",
                       "Impact (€)":"€{:+,.2f}","Impact (%)":"{:+.1f}%"}),
            use_container_width=True, hide_index=True)

    with tab_preset:
        name     = st.selectbox("Choose scenario", list(PRESET_SCENARIOS.keys()))
        scenario = PRESET_SCENARIOS[name]
        st.caption(scenario["description"])
        _show_stress_results(apply_stress(holdings, prices, scenario["shocks"]), tv, name)

    with tab_custom:
        _section("Set shock per asset type")
        st.caption("Enter the expected % change for each asset type.")
        asset_types   = list({h.asset_type.lower() for h in holdings})
        custom_shocks = {}
        for i, at in enumerate(st.columns(len(asset_types))):
            val = at.number_input(f"{asset_types[i].upper()} (%)", min_value=-100.0,
                                  max_value=500.0, value=0.0, step=1.0,
                                  key=f"shock_{asset_types[i]}")
            custom_shocks[asset_types[i]] = val / 100
        custom_name = st.text_input("Scenario name", placeholder="e.g. My bear case")
        if st.button("Run Custom Scenario", type="primary", key="run_custom"):
            _show_stress_results(apply_stress(holdings, prices, custom_shocks), tv, custom_name)

# ── Snapshot History ──────────────────────────────────────────────────────────
def render_snapshot_history():
    st.markdown("## Snapshot History")
    st.caption("Manually record your portfolio value over time to track growth.")

    holdings = portfolio().all_holdings()
    prices   = get_prices()
    tv = sum(h.current_value(prices[h.ticker]) for h in holdings if prices.get(h.ticker))
    ti = sum(h.total_invested for h in holdings)

    _section("Record Today's Value")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        st.metric("Current Value", fmt_cur(tv))
    with c2:
        note = st.text_input("Note (optional)", placeholder="e.g. After Q1 rebalance")
    with c3:
        st.write(""); st.write("")
        if st.button("📸  Save Snapshot", type="primary", use_container_width=True):
            if tv == 0:
                st.warning("Portfolio value is €0 — check prices are loaded.")
            else:
                snap = portfolio().add_snapshot(tv, ti, note)
                st.success(f"✓ Saved {fmt_cur(snap.total_value)} on {snap.date}")
                st.rerun()

    st.divider()
    snaps = portfolio().snapshots
    if not snaps:
        st.info("No snapshots yet. Click **Save Snapshot** above to record your first one.")
        return

    df = pd.DataFrame([{"Date":s.date,"Value":s.total_value,
                         "Invested":s.total_invested,"P&L":s.pnl}
                        for s in snaps]).sort_values("Date")

    _section("Portfolio Value Over Time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Value"], name="Value",
        line=dict(color=BLUE, width=2.5), mode="lines+markers",
        hovertemplate="<b>%{x}</b><br>€%{y:,.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Invested"], name="Invested",
        line=dict(color="#888", width=1.5, dash="dot"), mode="lines+markers",
        hovertemplate="<b>%{x}</b><br>€%{y:,.2f}<extra></extra>"))
    fig.update_layout(**_chart_layout(height=380))
    fig.update_layout(yaxis=dict(gridcolor="#1e1e1e", tickprefix="€"))
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure(go.Bar(x=df["Date"], y=df["P&L"],
        marker_color=[GAIN if v>=0 else LOSS for v in df["P&L"]],
        hovertemplate="<b>%{x}</b><br>€%{y:+,.2f}<extra></extra>"))
    fig2.add_hline(y=0, line_color="#333", line_dash="dash", line_width=1)
    fig2.update_layout(**_chart_layout("Unrealised P&L at Each Snapshot", 260))
    fig2.update_layout(yaxis=dict(gridcolor="#1e1e1e", tickprefix="€"))
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    _section("All Snapshots")
    for i, s in enumerate(reversed(snaps)):
        real_idx = len(snaps) - 1 - i
        pnl_col  = GAIN if s.pnl >= 0 else LOSS
        c_date, c_val, c_inv, c_pnl, c_note, c_del = st.columns([1,1,1,1,2,0.5])
        c_date.caption("Date");     c_date.write(s.date)
        c_val.caption("Value");     c_val.write(fmt_cur(s.total_value))
        c_inv.caption("Invested");  c_inv.write(fmt_cur(s.total_invested))
        c_pnl.caption("P&L")
        c_pnl.markdown(f'<span style="color:{pnl_col}">{fmt_cur(s.pnl)} ({fmt_pct(s.pnl_pct)})</span>',
                       unsafe_allow_html=True)
        c_note.caption("Note");     c_note.write(s.note or "—")
        with c_del:
            st.write("")
            if st.button("✕", key=f"del_snap_{real_idx}", help="Delete"):
                portfolio().delete_snapshot(real_idx); st.rerun()

# ── Router ────────────────────────────────────────────────────────────────────

def render_quant():
    from tracker.quant import (
        fetch_weekly_returns, sharpe_ratio, sortino_ratio, beta_and_alpha,
        max_drawdown, calmar_ratio, value_at_risk, cvar,
        rolling_sharpe, rolling_sortino, rolling_beta,
        compute_full_metrics, RISK_FREE_ANNUAL, WEEKS_PER_YEAR,
    )
    from tracker.benchmark import (
        build_portfolio_value_series, get_portfolio_start_date, INDICES,
    )

    st.markdown("## Quant Metrics")
    st.caption("Advanced portfolio statistics based on weekly returns · benchmark-relative measures use S&P 500 by default")

    port     = portfolio()
    holdings = port.all_holdings()
    if not holdings:
        st.info("Add some holdings first."); return

    start_date = get_portfolio_start_date(port)
    if start_date is None:
        st.info("Add some transactions first."); return

    # ── Controls ──
    col1, col2, col3 = st.columns(3)
    with col1:
        period = st.selectbox("Data period", ["1Y","2Y","3Y","5Y"], index=1)
    with col2:
        bench_name = st.selectbox("Benchmark", list(INDICES.keys()), index=1)
    with col3:
        roll_window = st.selectbox("Rolling window", ["26 weeks (6M)","52 weeks (1Y)","104 weeks (2Y)"], index=1)
    window = int(roll_window.split()[0])
    rf_pct = st.number_input("Risk-free rate (% annual, e.g. ECB rate)",
                              min_value=0.0, max_value=20.0,
                              value=float(RISK_FREE_ANNUAL * 100), step=0.25,
                              format="%.2f") / 100

    period_map = {"1Y":"1y","2Y":"2y","3Y":"3y","5Y":"5y"}
    if not st.button("Compute Metrics", type="primary"):
        return

    bench_ticker = INDICES[bench_name]
    all_tickers  = [h.ticker for h in holdings] + [bench_ticker]

    with st.spinner("Downloading weekly price data…"):
        returns_df = fetch_weekly_returns(all_tickers, period_map[period])

    if returns_df is None or returns_df.empty:
        st.error("Could not download price data. Try refreshing prices first."); return

    # Build portfolio weekly returns (value-weighted)
    prices   = get_prices()
    weights  = {}
    tv       = sum(h.current_value(prices[h.ticker]) for h in holdings if prices.get(h.ticker))
    for h in holdings:
        p = prices.get(h.ticker)
        if p: weights[h.ticker.upper()] = h.current_value(p) / tv if tv else 0

    # Weighted portfolio return series
    cols_available = [t for t in weights if t in returns_df.columns]
    if not cols_available:
        st.error("No matching ticker data found."); return

    w_arr = np.array([weights[t] for t in cols_available])
    w_arr = w_arr / w_arr.sum()  # renormalise in case some tickers had no data
    port_returns  = returns_df[cols_available].dot(w_arr)
    port_returns.name = "Portfolio"

    bench_col = bench_ticker.upper()
    if bench_col not in returns_df.columns:
        st.error(f"Could not load benchmark data for {bench_ticker}."); return
    bench_returns = returns_df[bench_col]
    bench_returns.name = bench_name

    # Align
    aligned = pd.concat([port_returns, bench_returns], axis=1).dropna()
    port_r  = aligned.iloc[:, 0]
    bench_r = aligned.iloc[:, 1]

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Headline metrics scorecard
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown("### 📋  Metrics Scorecard")

    metrics = compute_full_metrics(port_r, bench_r, "Portfolio", bench_name)
    raw     = metrics["_raw"]

    # Headline cards — most important metrics at a glance
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Sharpe Ratio",   f"{raw['sharpe']:.2f}",
              help="Excess return per unit of total risk. >1 = good, >2 = excellent.")
    m2.metric("Sortino Ratio",  f"{raw['sortino']:.2f}",
              help="Like Sharpe but only penalises downside volatility.")
    m3.metric("Jensen's Alpha", f"{raw['alpha']:+.2%}",
              help="Annualised outperformance above CAPM expectation. Positive = outperforming.")
    m4.metric("Beta",           f"{raw['beta']:.2f}",
              help="Sensitivity to benchmark. 1.0 = moves with market, <1 = defensive.")

    m5,m6,m7,m8 = st.columns(4)
    m5.metric("Max Drawdown",     f"{raw['mdd']:.2%}",
              help="Worst peak-to-trough decline over the period.")
    m6.metric("Calmar Ratio",     f"{raw['calmar']:.2f}",
              help="Annualised return / max drawdown. Higher = better risk-adjusted return.")
    m7.metric("VaR (95%)",        f"{raw['var95']:.2%}",
              help="Weekly loss not exceeded 95% of the time.")
    m8.metric("CVaR (95%)",       f"{raw['cvar95']:.2%}",
              help="Average loss in the worst 5% of weeks (Expected Shortfall).")

    # Full comparison table
    st.divider()
    _section("Full Metrics Table — Portfolio vs Benchmark")

    # Build benchmark metrics too
    beta_b, alpha_b = beta_and_alpha(bench_r, bench_r)
    ann_ret_b = (1 + bench_r.mean()) ** WEEKS_PER_YEAR - 1
    ann_vol_b = bench_r.std() * np.sqrt(WEEKS_PER_YEAR)
    mdd_b     = max_drawdown(bench_r)

    table_data = {
        "Metric": metrics["Metric"],
        "Portfolio": metrics["Portfolio"],
        bench_name: [
            f"{ann_ret_b:+.2%}", f"{ann_vol_b:.2%}",
            f"{sharpe_ratio(bench_r, rf=rf_pct/_WEEKLY_SCALE(rf_pct)):.2f}",
            f"{sortino_ratio(bench_r, rf=rf_pct/_WEEKLY_SCALE(rf_pct)):.2f}",
            "1.00", "—",
            f"{mdd_b:.2%}",
            f"{calmar_ratio(bench_r):.2f}",
            f"{value_at_risk(bench_r, 0.95):.2%}",
            f"{cvar(bench_r, 0.95):.2%}",
        ],
    }

    def _colour_metric(val, metric_name):
        """Green if the value is 'good', red if 'bad' — direction depends on metric."""
        try:
            n = float(str(val).replace("%","").replace("+",""))
        except Exception:
            return ""
        # Metrics where higher = better
        if any(x in metric_name for x in ["Return","Sharpe","Sortino","Alpha","Calmar"]):
            return f"color:{GAIN}" if n > 0 else f"color:{LOSS}"
        # Metrics where less negative = better (drawdown, VaR, CVaR, volatility)
        if any(x in metric_name for x in ["Drawdown","VaR","CVaR","Volatility"]):
            return f"color:{GAIN}" if n > -0.05 else f"color:{LOSS}"
        return ""

    df_table = pd.DataFrame(table_data)
    styled = df_table.style
    for col in ["Portfolio", bench_name]:
        for i, metric in enumerate(df_table["Metric"]):
            styled = styled.applymap(
                lambda v, m=metric: _colour_metric(v, m),
                subset=pd.IndexSlice[i, col]
            )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Return distribution
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown("### 📊  Return Distribution")

    col_l, col_r = st.columns(2)
    with col_l:
        _section("Weekly Return Histogram — Portfolio")
        var95  = value_at_risk(port_r, 0.95)
        cvar95 = cvar(port_r, 0.95)
        fig_h = go.Figure()
        fig_h.add_trace(go.Histogram(
            x=port_r * 100, nbinsx=40, name="Weekly Returns",
            marker_color=BLUE, opacity=0.75,
            hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>"))
        fig_h.add_vline(x=var95*100,  line_color=LOSS, line_dash="dash", line_width=1.5,
                        annotation_text=f"VaR 95%: {var95:.2%}", annotation_font_color=LOSS)
        fig_h.add_vline(x=cvar95*100, line_color="#ff8c00", line_dash="dot", line_width=1.5,
                        annotation_text=f"CVaR 95%: {cvar95:.2%}", annotation_font_color="#ff8c00")
        fig_h.update_layout(**_chart_layout(height=320))
        fig_h.update_layout(xaxis=dict(gridcolor="#1e1e1e", ticksuffix="%"),
                            yaxis=dict(gridcolor="#1e1e1e"))
        st.plotly_chart(fig_h, use_container_width=True)

    with col_r:
        _section("Upside vs Downside Weeks")
        up   = (port_r > 0).sum()
        down = (port_r < 0).sum()
        flat = (port_r == 0).sum()
        avg_up   = port_r[port_r > 0].mean() * 100 if up > 0 else 0
        avg_down = port_r[port_r < 0].mean() * 100 if down > 0 else 0
        fig_ud = go.Figure(go.Bar(
            x=["Up weeks", "Down weeks", "Flat"],
            y=[up, down, flat],
            marker_color=[GAIN, LOSS, "#888"],
            text=[f"{up}<br>avg {avg_up:+.2f}%",
                  f"{down}<br>avg {avg_down:+.2f}%",
                  str(flat)],
            textposition="auto",
            hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>"))
        fig_ud.update_layout(**_chart_layout(height=320))
        fig_ud.update_layout(yaxis=dict(gridcolor="#1e1e1e"))
        st.plotly_chart(fig_ud, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — Rolling metrics
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown(f"### 📈  Rolling Metrics  ({roll_window})")

    # Rolling Sharpe
    _section("Rolling Sharpe Ratio")
    rf_w = (1 + rf_pct) ** (1 / WEEKS_PER_YEAR) - 1
    r_sharpe_p = rolling_sharpe(port_r,  window, rf=rf_w)
    r_sharpe_b = rolling_sharpe(bench_r, window, rf=rf_w)

    fig_rs = go.Figure()
    fig_rs.add_trace(go.Scatter(x=r_sharpe_p.index, y=r_sharpe_p.values,
        name="Portfolio", line=dict(color=BLUE, width=2),
        hovertemplate="%{x|%d %b %Y}<br>Sharpe: %{y:.2f}<extra></extra>"))
    fig_rs.add_trace(go.Scatter(x=r_sharpe_b.index, y=r_sharpe_b.values,
        name=bench_name, line=dict(color=BENCH_COLOURS.get(bench_name,"#aaa"),
        width=1.5, dash="dot"),
        hovertemplate="%{x|%d %b %Y}<br>Sharpe: %{y:.2f}<extra></extra>"))
    fig_rs.add_hline(y=1, line_color="#4caf7d", line_dash="dot", line_width=1,
                     annotation_text="Good (1.0)")
    fig_rs.add_hline(y=0, line_color="#444",    line_dash="dash", line_width=1)
    fig_rs.update_layout(**_chart_layout(height=300))
    fig_rs.update_layout(yaxis=dict(gridcolor="#1e1e1e"))
    st.plotly_chart(fig_rs, use_container_width=True)

    # Rolling Sortino
    _section("Rolling Sortino Ratio")
    r_sortino_p = rolling_sortino(port_r,  window, rf=rf_w)
    r_sortino_b = rolling_sortino(bench_r, window, rf=rf_w)

    fig_rso = go.Figure()
    fig_rso.add_trace(go.Scatter(x=r_sortino_p.index, y=r_sortino_p.values,
        name="Portfolio", line=dict(color=BLUE, width=2),
        hovertemplate="%{x|%d %b %Y}<br>Sortino: %{y:.2f}<extra></extra>"))
    fig_rso.add_trace(go.Scatter(x=r_sortino_b.index, y=r_sortino_b.values,
        name=bench_name, line=dict(color=BENCH_COLOURS.get(bench_name,"#aaa"),
        width=1.5, dash="dot"),
        hovertemplate="%{x|%d %b %Y}<br>Sortino: %{y:.2f}<extra></extra>"))
    fig_rso.add_hline(y=0, line_color="#444", line_dash="dash", line_width=1)
    fig_rso.update_layout(**_chart_layout(height=300))
    fig_rso.update_layout(yaxis=dict(gridcolor="#1e1e1e"))
    st.plotly_chart(fig_rso, use_container_width=True)

    # Rolling Beta
    _section(f"Rolling Beta vs {bench_name}")
    r_beta = rolling_beta(port_r, bench_r, window)

    fig_rb = go.Figure()
    fig_rb.add_trace(go.Scatter(x=r_beta.index, y=r_beta.values,
        name="Beta", line=dict(color="#e8a838", width=2),
        fill="tozeroy", fillcolor="rgba(232,168,56,0.08)",
        hovertemplate="%{x|%d %b %Y}<br>Beta: %{y:.2f}<extra></extra>"))
    fig_rb.add_hline(y=1, line_color="#888", line_dash="dash", line_width=1,
                     annotation_text="Market (β=1)")
    fig_rb.add_hline(y=0, line_color="#444", line_dash="dot",  line_width=1)
    fig_rb.update_layout(**_chart_layout(height=300))
    fig_rb.update_layout(yaxis=dict(gridcolor="#1e1e1e"))
    st.plotly_chart(fig_rb, use_container_width=True)

    # Cumulative return comparison
    st.divider()
    _section("Cumulative Return — Portfolio vs Benchmark")
    cum_port  = (1 + port_r).cumprod() - 1
    cum_bench = (1 + bench_r).cumprod() - 1

    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(x=cum_port.index, y=cum_port.values * 100,
        name="Portfolio", line=dict(color=BLUE, width=2.5),
        hovertemplate="%{x|%d %b %Y}<br>%{y:+.2f}%<extra></extra>"))
    fig_cum.add_trace(go.Scatter(x=cum_bench.index, y=cum_bench.values * 100,
        name=bench_name, line=dict(color=BENCH_COLOURS.get(bench_name,"#aaa"),
        width=1.5, dash="dot"),
        hovertemplate="%{x|%d %b %Y}<br>%{y:+.2f}%<extra></extra>"))
    fig_cum.add_hline(y=0, line_color="#333", line_dash="dash", line_width=1)
    fig_cum.update_layout(**_chart_layout(height=340))
    fig_cum.update_layout(yaxis=dict(gridcolor="#1e1e1e", ticksuffix="%"))
    st.plotly_chart(fig_cum, use_container_width=True)


def _WEEKLY_SCALE(annual_rf: float) -> float:
    """Convert annual risk-free rate to weekly."""
    return (1 + annual_rf) ** (1 / 52) - 1


# ── Tax & Income ─────────────────────────────────────────────────────────────
def render_tax():
    from tracker.tax import (
        year_summary, compute_realised_gains, all_active_years,
        EFFECTIVE_RATE, SPARERPAUSCHBETRAG,
    )
    from datetime import datetime as _dt

    st.markdown("## Tax & Income")
    st.caption("German Abgeltungsteuer (25% + 5.5% Soli = 26.375%) · Sparerpauschbetrag €1,000 · FIFO cost basis")

    holdings  = portfolio().holdings
    dividends = portfolio().all_dividends()

    if not holdings:
        st.info("Add some transactions first."); return

    # ── Record a dividend ──
    with st.expander("➕  Record Dividend Payment"):
        tickers = [h.ticker for h in portfolio().all_holdings()]
        if not tickers:
            st.caption("No holdings yet.")
        else:
            dc1, dc2, dc3, dc4 = st.columns(4)
            div_ticker = dc1.selectbox("Ticker", tickers, key="div_ticker")
            div_date   = dc2.date_input("Date", value=date.today(), key="div_date")
            div_amount = dc3.number_input("Gross Amount (€)", min_value=0.01,
                                          step=0.01, format="%.2f", key="div_amount")
            div_wht    = dc4.number_input("Withholding Tax (€)", min_value=0.0,
                                          step=0.01, format="%.2f", key="div_wht",
                                          help="Tax already deducted at source by your broker")
            if st.button("Save Dividend", type="primary", key="save_div"):
                portfolio().add_dividend(div_ticker, str(div_date), div_amount, div_wht)
                st.success(f"✓ Dividend of {fmt_cur(div_amount)} recorded for {div_ticker}")
                st.rerun()

    st.divider()

    # ── Year selector ──
    active_years = all_active_years(holdings, dividends)
    if not active_years:
        st.info("No transactions or dividends recorded yet."); return

    current_year = _dt.today().year
    if current_year not in active_years:
        active_years.append(current_year)
    active_years = sorted(set(active_years), reverse=True)
    sel_year = st.selectbox("Tax Year", active_years, key="tax_year")
    summary  = year_summary(sel_year, holdings, dividends)

    # ── Headline cards ──
    st.markdown(f"### {sel_year} Tax Summary")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Realised Gains",    fmt_cur(summary.realised_gains),
              help="Total gains from sell transactions (FIFO cost basis)")
    c2.metric("Realised Losses",   f"-{fmt_cur(summary.realised_losses)}",
              help="Total losses from sell transactions")
    c3.metric("Dividend Income",   fmt_cur(summary.dividend_income),
              help="Gross dividends received")
    c4.metric("Total Commissions", fmt_cur(summary.total_commissions),
              help="Broker fees paid — already included in cost basis")

    c5,c6,c7,c8 = st.columns(4)
    c5.metric("Net Taxable Base",  fmt_cur(summary.net_taxable),
              help="Gains + dividends - losses (before Sparerpauschbetrag)")
    c6.metric("After Allowance",   fmt_cur(summary.after_allowance),
              help=f"After €{SPARERPAUSCHBETRAG:,.0f} Sparerpauschbetrag")
    c7.metric("Estimated Tax",     fmt_cur(summary.tax_owed),
              help="Abgeltungsteuer + Soli − withholding tax credit")
    c8.metric("Withholding Credit",fmt_cur(summary.withholding_credit),
              help="Tax already paid at source, credited against your bill")

    # Sparerpauschbetrag bar
    st.divider()
    _section("Sparerpauschbetrag (Annual Allowance)")
    pct_used = min(1.0, summary.allowance_used / SPARERPAUSCHBETRAG)
    colour   = "#4caf7d" if pct_used < 0.8 else "#e8a838" if pct_used < 1.0 else "#e05c5c"
    st.markdown(f"""
    <div style="background:#1e1e1e;border-radius:8px;overflow:hidden;height:28px">
      <div style="background:{colour};width:{pct_used*100:.1f}%;height:100%;
           display:flex;align-items:center;padding-left:10px;
           font-size:.85rem;font-weight:600;color:#fff">
        {pct_used*100:.1f}% used
      </div>
    </div>""", unsafe_allow_html=True)
    st.caption(f"Used: {fmt_cur(summary.allowance_used)}  ·  Remaining: {fmt_cur(summary.allowance_remaining)}")

    # Tax breakdown
    st.divider()
    _section("Tax Calculation Breakdown")
    breakdown = [
        ("Realised gains (sells)",        fmt_cur(summary.realised_gains)),
        ("Realised losses (sells)",       f"−{fmt_cur(summary.realised_losses)}"),
        ("Dividend income",               fmt_cur(summary.dividend_income)),
        ("Net taxable before allowance",  fmt_cur(summary.net_taxable)),
        ("Sparerpauschbetrag deduction", f"−{fmt_cur(summary.allowance_used)}"),
        ("Taxable base",                  fmt_cur(summary.after_allowance)),
        ("Abgeltungsteuer (25%)",        fmt_cur(summary.abgeltungsteuer)),
        ("Solidaritätszuschlag (5.5%)",  fmt_cur(summary.soli)),
        ("Withholding tax credit",       f"−{fmt_cur(summary.withholding_credit)}"),
        ("Estimated tax owed",            fmt_cur(summary.tax_owed)),
    ]
    st.dataframe(pd.DataFrame(breakdown, columns=["Item", "Amount"]),
                 use_container_width=True, hide_index=True)
    st.caption("⚠️  Estimate only. Kirchensteuer (8-9%) not included. Consult a Steuerberater for your official filing.")

    # Realised gains detail
    st.divider()
    _section(f"Realised Gains / Losses — {sel_year}")
    all_gains = [g for g in compute_realised_gains(holdings) if g.sell_date.startswith(str(sel_year))]
    if not all_gains:
        st.caption("No sell transactions in this year.")
    else:
        rows = [{"Ticker": g.ticker, "Sell Date": g.sell_date,
                 "Qty": round(g.quantity,4), "Proceeds": round(g.proceeds,2),
                 "Commission": round(g.commission,2),
                 "Cost Basis": round(g.cost_basis,2),
                 "Gain/Loss":  round(g.gain,2)} for g in all_gains]
        st.dataframe(
            pd.DataFrame(rows).style
              .applymap(lambda v: f"color:{GAIN}" if v > 0 else (f"color:{LOSS}" if v < 0 else ""),
                        subset=["Gain/Loss"])
              .format({"Proceeds":"€{:,.2f}","Commission":"€{:,.2f}",
                       "Cost Basis":"€{:,.2f}","Gain/Loss":"€{:+,.2f}"}),
            use_container_width=True, hide_index=True)

    # Dividend income
    st.divider()
    _section(f"Dividend Income — {sel_year}")
    year_divs = [d for d in dividends if d.date.startswith(str(sel_year))]
    if not year_divs:
        st.caption("No dividends recorded for this year.")
    else:
        div_rows = [{"Ticker": d.ticker, "Date": d.date,
                     "Gross (€)": round(d.amount,2),
                     "Withholding (€)": round(d.withholding_tax,2),
                     "Net (€)": round(d.net_amount,2)}
                    for d in sorted(year_divs, key=lambda x: x.date)]
        st.dataframe(pd.DataFrame(div_rows).style.format(
            {"Gross (€)":"€{:,.2f}","Withholding (€)":"€{:,.2f}","Net (€)":"€{:,.2f}"}),
            use_container_width=True, hide_index=True)

        by_ticker = {}
        for d in year_divs:
            by_ticker[d.ticker] = by_ticker.get(d.ticker, 0) + d.amount
        fig = go.Figure(go.Bar(x=list(by_ticker.keys()), y=list(by_ticker.values()),
            marker_color=PALETTE * 3,
            hovertemplate="<b>%{x}</b><br>€%{y:,.2f}<extra></extra>"))
        fig.update_layout(**_chart_layout("Dividends by Holding", 260))
        fig.update_layout(yaxis=dict(gridcolor="#1e1e1e", tickprefix="€"))
        st.plotly_chart(fig, use_container_width=True)

    # All dividend records with delete
    st.divider()
    _section("All Dividend Records")
    all_div_rows = db().get_dividends_with_ids()
    if not all_div_rows:
        st.caption("No dividends recorded yet.")
    else:
        for row in all_div_rows:
            c1,c2,c3,c4,c5,c_del = st.columns([1,1,1,1,1,0.4])
            c1.caption("Ticker");  c1.write(row["ticker"])
            c2.caption("Date");    c2.write(row["date"])
            c3.caption("Gross");   c3.write(fmt_cur(row["amount"]))
            c4.caption("WHT");     c4.write(fmt_cur(row["withholding_tax"]))
            c5.caption("Net");     c5.write(fmt_cur(row["amount"] - row["withholding_tax"]))
            with c_del:
                st.write("")
                if st.button("✕", key=f"del_div_{row['id']}", help="Delete"):
                    portfolio().delete_dividend(row["id"]); st.rerun()

    # Multi-year overview
    if len(active_years) > 1:
        st.divider()
        _section("Multi-Year Overview")
        yearly = [year_summary(y, holdings, dividends) for y in sorted(active_years)]
        st.dataframe(pd.DataFrame([{
            "Year":           s.year,
            "Gains":          fmt_cur(s.realised_gains),
            "Losses":        f"-{fmt_cur(s.realised_losses)}",
            "Dividends":      fmt_cur(s.dividend_income),
            "Net Taxable":    fmt_cur(s.net_taxable),
            "Allowance Used": fmt_cur(s.allowance_used),
            "Est. Tax":       fmt_cur(s.tax_owed),
        } for s in yearly]), use_container_width=True, hide_index=True)


def main():
    page = render_sidebar()
    {
        "Dashboard":          render_dashboard,
        "Holdings":           render_holdings,
        "Add Transaction":    render_add_transaction,
        "Benchmark":          render_benchmark,
        "Portfolio Analysis": render_analysis,
        "Snapshot History":   render_snapshot_history,
        "Quant Metrics":      render_quant,
        "Tax & Income":       render_tax,
    }.get(page, render_dashboard)()

if __name__ == "__main__":
    main()

# ── Quant Metrics ─────────────────────────────────────────────────────────────
