"""
app.py
======
Streamlit web interface for the Portfolio Tracker.
Run with:  streamlit run app.py
"""

import io
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
from tracker.portfolio import Holding, Portfolio
from tracker.prices import PriceFetcher
import tracker.exporter as exporter

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
BENCH_COLOURS = {
    "MSCI World":         "#e8a838",
    "S&P 500":            "#b07fd4",
    "NASDAQ 100":         "#4db6ac",
    "MSCI Emerging Mkts": "#f06292",
}

st.markdown("""
<style>
  [data-testid="metric-container"] {
      background:#1a1a1a; border:1px solid #2a2a2a;
      border-radius:8px; padding:14px 18px;
  }
  [data-testid="stMetricValue"] { font-size:1.35rem; }
  [data-testid="stMetricDelta"] { font-size:0.85rem; }
  [data-testid="stDataFrame"]   { border-radius:8px; overflow:hidden; }
  [data-testid="stSidebar"]     { background:#111111; }
  .block-container              { padding-top:1.5rem; }
  .section-title {
      font-size:0.75rem; font-weight:600; letter-spacing:0.1em;
      text-transform:uppercase; color:#555; margin:1.5rem 0 0.5rem 0;
  }
  /* News cards */
  .news-card {
      background:#141414; border:1px solid #222; border-radius:8px;
      padding:12px 16px; margin-bottom:10px;
  }
  .news-card a { color:#5b9bd5; text-decoration:none; font-weight:500; }
  .news-card a:hover { text-decoration:underline; }
  .news-meta { font-size:0.75rem; color:#555; margin-top:4px; }
  .news-ticker {
      display:inline-block; background:#1e2a38; color:#5b9bd5;
      font-size:0.7rem; font-weight:700; border-radius:4px;
      padding:1px 6px; margin-right:6px;
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
if "news_cache" not in st.session_state:
    st.session_state.news_cache = {}


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
    return f"{'+'if v>0 else ''}{v:.2f}%"

def _chart_layout(title="", height=400) -> dict:
    return dict(
        title=title,
        paper_bgcolor=BG, plot_bgcolor=BG,
        font_color="#cccccc", height=height,
        xaxis=dict(gridcolor="#1e1e1e"),
        yaxis=dict(gridcolor="#1e1e1e"),
        legend=dict(bgcolor="#1a1a1a", bordercolor="#2a2a2a", borderwidth=1),
        margin=dict(t=50, b=20, l=10, r=10),
        hovermode="x unified",
    )


# ── News fetching ─────────────────────────────────────────────────────────────
def fetch_news(tickers: List[str]) -> List[dict]:
    """
    Fetch recent news headlines for each ticker using yfinance.
    Results are cached in session state so we don't re-fetch on every rerun.

    yfinance returns news as a list of dicts with keys:
      title, link, publisher, providerPublishTime, relatedTickers
    """
    cache = st.session_state.news_cache
    all_news = []

    for ticker in tickers:
        if ticker in cache:
            articles = cache[ticker]
        else:
            try:
                t        = yf.Ticker(ticker)
                articles = t.news or []
                cache[ticker] = articles
            except Exception:
                articles = []

        # Tag each article with which ticker it came from
        for article in articles[:5]:   # Max 5 per ticker
            article["_ticker"] = ticker
            all_news.append(article)

    # Sort by publish time, newest first
    all_news.sort(key=lambda x: x.get("providerPublishTime", 0), reverse=True)

    # Deduplicate by title
    seen, unique = set(), []
    for article in all_news:
        title = article.get("title", "")
        if title not in seen:
            seen.add(title)
            unique.append(article)

    return unique[:20]   # Show max 20 articles total


def _render_news(tickers: List[str]) -> None:
    """Render the news feed section on the dashboard."""
    st.markdown('<p class="section-title">Latest News</p>', unsafe_allow_html=True)

    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        if st.button("↺  Refresh News", key="refresh_news"):
            st.session_state.news_cache = {}
            st.rerun()

    with st.spinner("Loading news…"):
        articles = fetch_news(tickers)

    if not articles:
        st.caption("No news found for your holdings.")
        return

    for article in articles:
        title     = article.get("title", "No title")
        link      = article.get("link", "#")
        publisher = article.get("publisher", "")
        ts        = article.get("providerPublishTime", 0)
        ticker    = article.get("_ticker", "")

        # Format the timestamp into a readable date
        try:
            pub_date = datetime.fromtimestamp(ts).strftime("%d %b %Y  %H:%M")
        except Exception:
            pub_date = ""

        st.markdown(f"""
<div class="news-card">
  <span class="news-ticker">{ticker}</span>
  <a href="{link}" target="_blank">{title}</a>
  <div class="news-meta">{publisher}  ·  {pub_date}</div>
</div>
""", unsafe_allow_html=True)


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

        holdings       = portfolio().all_holdings()
        prices         = get_prices()
        total_value    = sum(h.current_value(prices[h.ticker]) for h in holdings if prices.get(h.ticker))
        total_invested = sum(h.total_invested for h in holdings)
        total_pnl      = total_value - total_invested

        st.metric("Portfolio Value", fmt_cur(total_value))
        st.metric(
            "Unrealised P&L", fmt_cur(total_pnl),
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

    # ── Metric cards ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Invested",  fmt_cur(total_invested))
    c2.metric("Portfolio Value", fmt_cur(total_value))
    c3.metric("Unrealised P&L",  fmt_cur(total_pnl), delta=fmt_pct(overall_pct))
    c4.metric("Holdings",        str(len(holdings)))

    st.divider()

    # ── Charts ──
    col_l, col_r = st.columns(2)
    with col_l:
        _render_allocation_chart(holdings, prices)
    with col_r:
        _render_pnl_chart(holdings, prices)

    st.divider()
    _render_value_chart(holdings, prices)

    st.divider()

    # ── News feed ──
    tickers = [h.ticker for h in holdings]
    _render_news(tickers)


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

    # ── Summary table ──
    rows = []
    for h in holdings:
        p = prices.get(h.ticker)
        rows.append({
            "Ticker":  h.ticker, "Name": h.name, "Type": h.asset_type.upper(),
            "Qty":     round(h.quantity, 4), "Avg Cost": round(h.average_cost, 4),
            "Price":   round(p, 4) if p else None,
            "Value":   round(h.current_value(p), 2) if p else None,
            "P&L":     round(h.unrealised_pnl(p), 2) if p else None,
            "P&L %":   round(h.pnl_percent(p), 2) if p else None,
        })

    df = pd.DataFrame(rows)

    def colour_pnl(val):
        if val is None or (isinstance(val, float) and pd.isna(val)): return "color:#888"
        return f"color:{GAIN}" if val > 0 else f"color:{LOSS}"

    st.dataframe(
        df.style
          .applymap(colour_pnl, subset=["P&L", "P&L %"])
          .format({"Avg Cost":"€{:.4f}", "Price":"€{:.4f}",
                   "Value":"€{:,.2f}", "P&L":"€{:+,.2f}", "P&L %":"{:+.2f}%"},
                  na_rep="—"),
        use_container_width=True, hide_index=True,
    )

    # ── Export buttons ──
    st.divider()
    st.markdown('<p class="section-title">Export</p>', unsafe_allow_html=True)
    col_xl, col_csv, _ = st.columns([1, 1, 4])

    with col_xl:
        import tempfile, os
        buf = io.BytesIO()
        fname = f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp_path = tmp.name
        exporter.export_to_excel(holdings, prices, filename=tmp_path)
        with open(tmp_path, "rb") as f:
            buf.write(f.read())
        os.unlink(tmp_path)
        buf.seek(0)
        st.download_button(
            "⬇  Download Excel",
            data=buf,
            file_name=fname,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    with col_csv:
        csv_rows = []
        for h in holdings:
            p = prices.get(h.ticker)
            csv_rows.append({
                "ticker": h.ticker, "name": h.name, "type": h.asset_type,
                "quantity": round(h.quantity, 6),
                "avg_cost": round(h.average_cost, 4),
                "current_price": round(p, 4) if p else "",
                "market_value":  round(h.current_value(p), 2) if p else "",
                "unrealised_pnl":round(h.unrealised_pnl(p), 2) if p else "",
                "pnl_percent":   round(h.pnl_percent(p), 2) if p else "",
            })
        csv_str = pd.DataFrame(csv_rows).to_csv(index=False)
        st.download_button(
            "⬇  Download CSV",
            data=csv_str,
            file_name=f"portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.divider()

    # ── Per-holding detail with price history ──
    st.markdown('<p class="section-title">Holding Detail</p>', unsafe_allow_html=True)

    for h in holdings:
        with st.expander(f"{h.ticker}  —  {h.name}"):
            p = prices.get(h.ticker)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Avg Cost",      fmt_cur(h.average_cost))
            m2.metric("Current Price", fmt_cur(p) if p else "—")
            m3.metric("Market Value",  fmt_cur(h.current_value(p)) if p else "—")
            if p:
                m4.metric("P&L", fmt_cur(h.unrealised_pnl(p)),
                          delta=fmt_pct(h.pnl_percent(p)))

            # ── Price history chart ──
            period_map = {"1M": "1mo", "3M": "3mo", "6M": "6mo",
                          "1Y": "1y",  "3Y": "3y",  "5Y": "5y"}
            period_choice = st.radio(
                "Period", list(period_map.keys()),
                horizontal=True, key=f"period_{h.ticker}",
            )
            _render_price_history(h.ticker, period_map[period_choice], h.transactions)

            # ── Transaction history table ──
            t_rows = [{"Date": t.date, "Action": t.action.upper(),
                       "Quantity": t.quantity, "Price": t.price,
                       "Total": round(t.total_cost, 2)} for t in h.transactions]
            t_df = pd.DataFrame(t_rows)

            def colour_action(val):
                return f"color:{GAIN}" if val == "BUY" else f"color:{LOSS}"

            st.dataframe(
                t_df.style.applymap(colour_action, subset=["Action"])
                    .format({"Price": "€{:.4f}", "Total": "€{:,.2f}"}),
                use_container_width=True, hide_index=True,
            )

            if st.button(f"🗑  Remove {h.ticker}", key=f"remove_{h.ticker}"):
                portfolio().remove_holding(h.ticker)
                st.session_state.prices = {}
                st.rerun()


def _render_price_history(ticker: str, period: str, transactions=None) -> None:
    """
    Fetch and plot price history for a single ticker.
    Overlays buy/sell transaction markers on the price line.

    Key concept: yf.Ticker(ticker).history(period=...) returns a DataFrame
    with Date as index and Open/High/Low/Close/Volume as columns.
    """
    try:
        hist = yf.Ticker(ticker).history(period=period)
    except Exception:
        st.caption("Could not load price history.")
        return

    if hist.empty:
        st.caption("No historical data available.")
        return

    hist.index = pd.to_datetime(hist.index).tz_localize(None)

    fig = go.Figure()

    # Main price line
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["Close"],
        name="Price", line=dict(color=BLUE, width=2),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>€%{y:,.4f}<extra></extra>",
    ))

    # Fill under the line
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["Close"],
        fill="tozeroy", fillcolor="rgba(91,155,213,0.06)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))

    # ── Overlay buy/sell markers ──
    if transactions:
        period_start = hist.index.min().date()
        buys  = [(t.date, t.price) for t in transactions
                 if t.action == "buy" and datetime.strptime(t.date, "%Y-%m-%d").date() >= period_start]
        sells = [(t.date, t.price) for t in transactions
                 if t.action == "sell" and datetime.strptime(t.date, "%Y-%m-%d").date() >= period_start]

        if buys:
            bx, by = zip(*buys)
            fig.add_trace(go.Scatter(
                x=list(bx), y=list(by), mode="markers", name="Buy",
                marker=dict(color=GAIN, size=10, symbol="triangle-up"),
                hovertemplate="<b>BUY</b><br>%{x}<br>€%{y:,.4f}<extra></extra>",
            ))
        if sells:
            sx, sy = zip(*sells)
            fig.add_trace(go.Scatter(
                x=list(sx), y=list(sy), mode="markers", name="Sell",
                marker=dict(color=LOSS, size=10, symbol="triangle-down"),
                hovertemplate="<b>SELL</b><br>%{x}<br>€%{y:,.4f}<extra></extra>",
            ))

    fig.update_layout(**_chart_layout(height=300))
    fig.update_layout(yaxis=dict(gridcolor="#1e1e1e", tickprefix="€"),
                      showlegend=True)
    st.plotly_chart(fig, use_container_width=True)


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
            quantity = st.number_input("Quantity",           min_value=0.0001, step=0.0001, format="%.4f")
            price    = st.number_input("Price per unit (€)", min_value=0.0001, step=0.01,   format="%.4f")
            txn_date = st.date_input("Date", value=date.today())

        if st.form_submit_button("Add Transaction", use_container_width=True, type="primary"):
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
def render_benchmark():
    st.markdown("## Benchmark")
    st.caption("Portfolio performance vs market indices · rebased to €100 from your first transaction")

    port       = portfolio()
    start_date = get_portfolio_start_date(port)

    if start_date is None:
        st.info("Add some transactions first.")
        return

    st.info(f"Benchmarking from your first transaction: **{start_date.strftime('%d %b %Y')}**")

    selected_indices = st.multiselect(
        "Compare against", options=list(INDICES.keys()), default=list(INDICES.keys()),
    )

    if not st.button("Run Benchmark", type="primary"):
        return

    with st.spinner("Building portfolio value series…"):
        port_series = build_portfolio_value_series(port, start_date)

    if port_series is None or port_series.empty:
        st.error("Could not build portfolio series.")
        return

    index_series: Dict[str, pd.Series] = {}
    for name in selected_indices:
        ticker = INDICES[name]
        with st.spinner(f"Fetching {name}…"):
            s = fetch_index_series(ticker, start_date)
        if s is not None:
            index_series[name] = s
        else:
            st.warning(f"Could not fetch {name}")

    norm_port    = normalise(port_series)
    norm_indices = {n: normalise(s.reindex(port_series.index, method="ffill"))
                    for n, s in index_series.items()}

    st.divider()

    # ── Growth of €100 ──
    st.markdown('<p class="section-title">Growth of €100</p>', unsafe_allow_html=True)
    fig_g = go.Figure()
    fig_g.add_trace(go.Scatter(
        x=norm_port.index, y=norm_port.values, name="My Portfolio",
        line=dict(color=BLUE, width=2.5),
        hovertemplate="<b>Portfolio</b><br>€%{y:.2f}<extra></extra>",
    ))
    for name, s in norm_indices.items():
        fig_g.add_trace(go.Scatter(
            x=s.index, y=s.values, name=name,
            line=dict(color=BENCH_COLOURS.get(name, "#aaa"), width=1.5, dash="dot"),
            hovertemplate=f"<b>{name}</b><br>€%{{y:.2f}}<extra></extra>",
        ))
    fig_g.add_hline(y=100, line_color="#333", line_dash="dash", line_width=1)
    fig_g.update_layout(**_chart_layout(height=420))
    fig_g.update_layout(yaxis=dict(gridcolor="#1e1e1e", tickprefix="€"))
    st.plotly_chart(fig_g, use_container_width=True)

    st.divider()

    # ── P&L over time ──
    st.markdown('<p class="section-title">Cumulative P&L over Time</p>', unsafe_allow_html=True)

    # P&L = portfolio value minus the "invested" baseline on each day
    # We approximate invested capital as the starting value of the portfolio
    # (since we can't easily track cash flows per day here)
    invested_baseline = port_series.iloc[0]
    pnl_series = port_series - invested_baseline

    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Scatter(
        x=pnl_series.index, y=pnl_series.values,
        name="P&L",
        fill="tozeroy",
        fillcolor="rgba(76,175,125,0.08)",
        line=dict(color=GAIN, width=2),
        hovertemplate="<b>P&L</b><br>€%{y:+,.2f}<extra></extra>",
    ))
    fig_pnl.add_hline(y=0, line_color="#333", line_dash="dash", line_width=1)
    fig_pnl.update_layout(**_chart_layout(height=300))
    fig_pnl.update_layout(yaxis=dict(gridcolor="#1e1e1e", tickprefix="€"))
    st.plotly_chart(fig_pnl, use_container_width=True)

    st.divider()

    # ── Drawdown ──
    st.markdown('<p class="section-title">Drawdown</p>', unsafe_allow_html=True)
    fig_dd = go.Figure()
    dd_port = compute_drawdown(norm_port)
    fig_dd.add_trace(go.Scatter(
        x=dd_port.index, y=dd_port.values, name="My Portfolio",
        fill="tozeroy", fillcolor="rgba(91,155,213,0.10)",
        line=dict(color=BLUE, width=2),
        hovertemplate="<b>Portfolio</b><br>%{y:.2f}%<extra></extra>",
    ))
    for name, s in norm_indices.items():
        dd = compute_drawdown(s)
        fig_dd.add_trace(go.Scatter(
            x=dd.index, y=dd.values, name=name,
            line=dict(color=BENCH_COLOURS.get(name, "#aaa"), width=1.5, dash="dot"),
            hovertemplate=f"<b>{name}</b><br>%{{y:.2f}}%<extra></extra>",
        ))
    fig_dd.update_layout(**_chart_layout(height=320))
    fig_dd.update_layout(yaxis=dict(gridcolor="#1e1e1e", ticksuffix="%"))
    st.plotly_chart(fig_dd, use_container_width=True)

    st.divider()

    # ── Stats table ──
    st.markdown('<p class="section-title">Key Statistics</p>', unsafe_allow_html=True)
    all_stats = [compute_stats(port_series, "My Portfolio")]
    for name, s in index_series.items():
        aligned = s.reindex(port_series.index, method="ffill").dropna()
        all_stats.append(compute_stats(aligned, name))

    display_cols = ["Label", "Total Return", "Ann. Return", "Ann. Volatility",
                    "Sharpe Ratio", "Max Drawdown", "Best Day", "Worst Day", "Days"]
    stats_df = pd.DataFrame(all_stats)[display_cols].rename(columns={"Label": ""})

    def colour_stat(val):
        try:
            num = float(str(val).replace("%", "").replace("+", ""))
            if num > 0: return f"color:{GAIN}"
            if num < 0: return f"color:{LOSS}"
        except Exception:
            pass
        return ""

    st.dataframe(
        stats_df.style.applymap(colour_stat, subset=["Total Return", "Ann. Return",
                                                      "Max Drawdown", "Best Day", "Worst Day"]),
        use_container_width=True, hide_index=True,
    )


# ── Covariance ────────────────────────────────────────────────────────────────
def render_covariance():
    import numpy as np

    st.markdown("## Covariance & Correlation Matrix")
    st.caption("Based on 3 years of weekly returns · sourced from Yahoo Finance")

    holdings = portfolio().all_holdings()
    if len(holdings) < 2:
        st.info("You need at least 2 holdings to compute a covariance matrix.")
        return

    all_tickers = [h.ticker for h in holdings]
    selected    = st.multiselect("Select tickers", options=all_tickers, default=all_tickers)
    if len(selected) < 2:
        st.warning("Please select at least 2 tickers.")
        return

    if not st.button("Run Analysis", type="primary"):
        return

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
        text=text, texttemplate="%{text}", colorscale="RdYlGn",
        zmin=zmin, zmax=zmax, zmid=zmid, showscale=True,
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
