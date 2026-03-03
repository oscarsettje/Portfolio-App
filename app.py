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
from tracker.validation import (validate_ticker, validate_transaction,
    validate_transaction_list, validate_dividend, validate_name)
from tracker.prices import PriceFetcher, _close_from_download
import tracker.exporter as exporter
from tracker.importer import parse_pp_csv, execute_import, IMPORT_AS_BUY, IMPORT_AS_SELL, IMPORT_AS_DIVIDEND

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
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@600;700;800&family=Inter:wght@300;400;500;600&display=swap');

  /* ── Global ── */
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #d4d4d4;
  }
  .block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1400px; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #080808;
    border-right: 1px solid #1c1c1c;
  }
  [data-testid="stSidebar"] .stRadio label {
    font-family: 'Inter', sans-serif;
    font-size: .85rem;
    font-weight: 500;
    color: #888;
    padding: 6px 10px;
    border-radius: 6px;
    transition: color .15s, background .15s;
    cursor: pointer;
  }
  [data-testid="stSidebar"] .stRadio label:hover { color: #ddd; background: #141414; }
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    font-family: 'Inter', sans-serif;
  }

  /* ── Metric cards ── */
  [data-testid="metric-container"] {
    background: #0e0e0e;
    border: 1px solid #1e1e1e;
    border-radius: 10px;
    padding: 16px 20px;
    transition: border-color .2s;
  }
  [data-testid="metric-container"]:hover { border-color: #2e2e2e; }
  [data-testid="stMetricLabel"] {
    font-family: 'Inter', sans-serif;
    font-size: .7rem !important;
    font-weight: 600;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: #555 !important;
  }
  [data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 1.3rem !important;
    font-weight: 500;
    color: #e8e8e8 !important;
    letter-spacing: -.02em;
  }
  [data-testid="stMetricDelta"] {
    font-family: 'DM Mono', monospace !important;
    font-size: .78rem !important;
  }

  /* ── Page header ── */
  .page-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 18px 24px;
    background: linear-gradient(135deg, #0d0d0d 0%, #111 100%);
    border: 1px solid #1e1e1e;
    border-radius: 12px;
    margin-bottom: 1.6rem;
  }
  .page-header-icon {
    font-size: 1.6rem;
    line-height: 1;
  }
  .page-header-text h1 {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #f0f0f0;
    margin: 0;
    line-height: 1.2;
    letter-spacing: -.02em;
  }
  .page-header-text p {
    font-size: .78rem;
    color: #555;
    margin: 2px 0 0;
    font-weight: 400;
  }

  /* ── Section titles ── */
  .section-title {
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: 'Inter', sans-serif;
    font-size: .68rem;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: #444;
    margin: 1.6rem 0 .7rem;
  }
  .section-title::before {
    content: '';
    display: inline-block;
    width: 3px;
    height: 12px;
    background: #5b9bd5;
    border-radius: 2px;
    flex-shrink: 0;
  }

  /* ── DataFrames ── */
  [data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #1a1a1a;
  }

  /* ── Buttons ── */
  .stButton > button {
    font-family: 'Inter', sans-serif;
    font-size: .82rem;
    font-weight: 500;
    border-radius: 8px;
    border: 1px solid #2a2a2a;
    background: #111;
    color: #bbb;
    transition: all .15s;
  }
  .stButton > button:hover {
    border-color: #3a3a3a;
    background: #1a1a1a;
    color: #eee;
  }
  .stButton > button[kind="primary"] {
    background: #1a2a3a;
    border-color: #2a4a6a;
    color: #7ab3d9;
  }
  .stButton > button[kind="primary"]:hover {
    background: #1e3348;
    border-color: #3a6a9a;
    color: #9ecbeb;
  }

  /* ── Inputs ── */
  .stTextInput > div > div > input,
  .stNumberInput > div > div > input,
  .stSelectbox > div > div,
  .stDateInput > div > div > input {
    background: #0e0e0e !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 8px !important;
    color: #d4d4d4 !important;
    font-family: 'Inter', sans-serif;
    font-size: .85rem;
  }
  .stTextInput > div > div > input:focus,
  .stNumberInput > div > div > input:focus {
    border-color: #4a7aaa !important;
    box-shadow: 0 0 0 2px rgba(91,155,213,.12) !important;
  }

  /* ── Expanders ── */
  [data-testid="stExpander"] {
    border: 1px solid #1e1e1e !important;
    border-radius: 10px !important;
    background: #0b0b0b;
    margin-bottom: .5rem;
  }
  [data-testid="stExpander"] summary {
    font-family: 'Inter', sans-serif;
    font-size: .85rem;
    font-weight: 500;
    color: #888;
    padding: 10px 14px;
  }
  [data-testid="stExpander"] summary:hover { color: #ccc; }

  /* ── Dividers ── */
  hr { border-color: #161616 !important; margin: 1.2rem 0 !important; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: transparent;
    border-bottom: 1px solid #1e1e1e;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-size: .82rem;
    font-weight: 500;
    color: #555;
    background: transparent;
    border-radius: 8px 8px 0 0;
    padding: 8px 16px;
  }
  .stTabs [aria-selected="true"] { color: #5b9bd5 !important; background: #0e1a26 !important; }

  /* ── Alerts / info boxes ── */
  [data-testid="stAlert"] {
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
    font-size: .83rem;
  }

  /* ── Captions ── */
  [data-testid="stCaptionContainer"] p {
    font-family: 'Inter', sans-serif;
    font-size: .75rem;
    color: #4a4a4a;
  }

  /* ── Spinners ── */
  [data-testid="stSpinner"] p { font-family: 'Inter', sans-serif; font-size: .82rem; color: #555; }

  /* ── News cards ── */
  .news-card {
    background: #0c0c0c;
    border: 1px solid #1c1c1c;
    border-radius: 10px;
    padding: 13px 18px;
    margin-bottom: 8px;
    transition: border-color .15s;
  }
  .news-card:hover { border-color: #2a2a2a; }
  .news-card a {
    font-family: 'Inter', sans-serif;
    color: #c8dff0;
    text-decoration: none;
    font-weight: 500;
    font-size: .88rem;
    line-height: 1.45;
  }
  .news-card a:hover { color: #7ab3d9; }
  .news-meta { font-size: .72rem; color: #444; margin-top: 5px; font-family: 'DM Mono', monospace; }
  .news-ticker {
    display: inline-block;
    background: #0e1e2e;
    color: #5b9bd5;
    font-family: 'DM Mono', monospace;
    font-size: .68rem;
    font-weight: 500;
    border-radius: 4px;
    padding: 2px 7px;
    margin-right: 8px;
    border: 1px solid #1a3a5a;
    letter-spacing: .04em;
  }

  /* ── Download buttons ── */
  .stDownloadButton > button {
    font-family: 'Inter', sans-serif;
    font-size: .82rem;
    font-weight: 500;
    border-radius: 8px;
    background: #0e1a0e;
    border: 1px solid #1e3a1e;
    color: #6abf6a;
    transition: all .15s;
  }
  .stDownloadButton > button:hover {
    background: #122012;
    border-color: #2a5a2a;
    color: #8ad48a;
  }

  /* ── Top-performer / laggard strip ── */
  .perf-strip { display: flex; gap: 12px; margin: .8rem 0; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #0a0a0a; }
  ::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #3a3a3a; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "db"              not in st.session_state: st.session_state.db              = Database()
if "fetcher"         not in st.session_state: st.session_state.fetcher         = PriceFetcher(db=st.session_state.db)
if "user_id"         not in st.session_state: st.session_state.user_id         = None
if "username"        not in st.session_state: st.session_state.username        = None
if "portfolio"       not in st.session_state: st.session_state.portfolio       = None
if "prices"          not in st.session_state: st.session_state.prices          = {}
if "news_cache"      not in st.session_state: st.session_state.news_cache      = {}
if "sector_cache"    not in st.session_state: st.session_state.sector_cache    = {}
if "stale_prices"    not in st.session_state: st.session_state.stale_prices    = set()
if "missing_prices"  not in st.session_state: st.session_state.missing_prices  = set()
if "bench_cache"     not in st.session_state: st.session_state.bench_cache     = {}
if "quant_cache"     not in st.session_state: st.session_state.quant_cache     = {}
if "portfolio_stats" not in st.session_state: st.session_state.portfolio_stats = {}

def portfolio() -> Portfolio:    return st.session_state.portfolio
def fetcher()   -> PriceFetcher: return st.session_state.fetcher
def db()        -> Database:     return st.session_state.db

def _login_as(username: str) -> None:
    """Switch the active user — resets all per-user session state."""
    uid = db().get_or_create_user(username)
    st.session_state.user_id   = uid
    st.session_state.username  = username
    st.session_state.portfolio = Portfolio(db=db(), user_id=uid, username=username)
    # Clear all user-specific caches
    st.session_state.prices          = {}
    st.session_state.news_cache      = {}
    st.session_state.sector_cache    = {}
    st.session_state.stale_prices    = set()
    st.session_state.missing_prices  = set()
    st.session_state.bench_cache     = {}
    st.session_state.quant_cache     = {}
    st.session_state.portfolio_stats = {}

def _render_login() -> bool:
    """
    Show login screen if no user is selected.
    Returns True if we should proceed to the main app, False if login is needed.
    """
    if st.session_state.user_id is not None:
        return True

    # Centre the login card
    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        users = db().get_all_users()
        usernames = [u["username"] for u in users]

        st.markdown("""
        <div style="text-align:center;padding:40px 0 24px">
          <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;
                      color:#e8e8e8;letter-spacing:-.03em">Portfolio</div>
          <div style="font-family:'DM Mono',monospace;font-size:.7rem;
                      color:#3a6a9a;letter-spacing:.2em;margin-top:4px">TRACKER</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#0e0e0e;border:1px solid #1e1e1e;border-radius:14px;
                    padding:28px 28px 24px">
        """, unsafe_allow_html=True)

        if usernames:
            st.markdown('<p style="font-family:\'Inter\',sans-serif;font-size:.8rem;'
                        'color:#666;margin-bottom:4px">Who\'s viewing?</p>',
                        unsafe_allow_html=True)
            selected = st.selectbox("Select user", usernames,
                                    label_visibility="collapsed")
            if st.button("Continue", type="primary", use_container_width=True):
                _login_as(selected)
                st.rerun()

            st.markdown('<div style="text-align:center;padding:14px 0 4px;'
                        'font-family:\'Inter\',sans-serif;font-size:.72rem;color:#333">or</div>',
                        unsafe_allow_html=True)

        new_name = st.text_input("New user name", placeholder="Enter a name to create a new account",
                                 label_visibility="collapsed")
        if st.button("Create & Continue", use_container_width=True,
                     type="primary" if not usernames else "secondary"):
            new_name = new_name.strip()
            if not new_name:
                st.error("Please enter a name.")
            elif len(new_name) < 2:
                st.error("Name must be at least 2 characters.")
            elif len(new_name) > 30:
                st.error("Name must be 30 characters or less.")
            elif new_name in usernames:
                st.error(f'"{new_name}" already exists. Select it above.')
            else:
                _login_as(new_name)
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
    return False

# ── Prices ────────────────────────────────────────────────────────────────────
def get_prices() -> Dict[str, Optional[float]]:
    if not st.session_state.prices:
        holdings = portfolio().all_holdings()
        tickers  = [h.ticker for h in holdings if not h.manual_price]
        if tickers:
            with st.spinner("Fetching live prices…"):
                try:
                    st.session_state.prices = fetcher().get_prices(tickers)
                except Exception as e:
                    st.warning(f"Price fetch encountered an error: {e}. Using cached prices where available.")
                    st.session_state.prices = {t: fetcher()._cache.get(t) for t in tickers}
                st.session_state.stale_prices = {
                    t for t in tickers if fetcher().is_stale(t)
                }
                # Surface tickers with no price at all (neither live nor cached)
                missing = [t for t in tickers if not st.session_state.prices.get(t)]
                if missing:
                    st.session_state.missing_prices = set(missing)
        for h in holdings:
            if h.manual_price is not None:
                st.session_state.prices[h.ticker] = h.manual_price
    return st.session_state.prices

def invalidate_prices():
    fetcher().clear_cache()
    st.session_state.prices          = {}
    st.session_state.portfolio_stats = {}
    st.session_state.missing_prices  = set()

@st.cache_data(ttl=300, show_spinner=False)
def _cached_tax_summary(holdings_key: str, dividends_key: str):
    """
    Cache tax summaries for 5 minutes — they only change when transactions
    or dividends change, not on every Streamlit rerun.
    Returns dict of {year: TaxSummary} or {} on error.
    """
    from tracker.tax import year_summary, all_active_years
    try:
        holdings  = portfolio().holdings
        dividends = portfolio().all_dividends()
        years     = all_active_years(holdings, dividends)
        return {y: year_summary(y, holdings, dividends) for y in years}
    except Exception as e:
        # Return empty so callers fall back to live computation with visible error
        print(f"[Warning] Tax summary cache failed: {e}")
        return {}

# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_cur(v: float) -> str: return f"€{v:,.2f}"

def _portfolio_key() -> str:
    """A short hash of current holdings + transaction count — used as cache key."""
    import hashlib
    h = portfolio()
    raw = f"{len(h.holdings)}:{sum(len(hh.transactions) for hh in h.holdings.values())}"
    return hashlib.md5(raw.encode()).hexdigest()[:8]

def _dividends_key() -> str:
    import hashlib
    divs = portfolio().all_dividends()
    raw  = f"{len(divs)}:{sum(d.amount for d in divs):.2f}"
    return hashlib.md5(raw.encode()).hexdigest()[:8]
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

def _excel_bytes(holdings, prices, dividends=None) -> Optional[io.BytesIO]:
    """Returns BytesIO on success, or None and shows st.error() on failure."""
    buf = io.BytesIO()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        tmp_path = tmp.name
    try:
        exporter.export_to_excel(holdings, prices,
                                 dividends=dividends or [],
                                 filename=tmp_path)
        with open(tmp_path, "rb") as f:
            buf.write(f.read())
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"Export failed: {e}. Try again or check that openpyxl is installed.")
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

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
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

def _page_header(icon: str, title: str, subtitle: str = ""):
    sub = f'<p>{subtitle}</p>' if subtitle else ''
    st.markdown(f"""
    <div class="page-header">
      <div class="page-header-icon">{icon}</div>
      <div class="page-header-text"><h1>{title}</h1>{sub}</div>
    </div>""", unsafe_allow_html=True)

# ── News ──────────────────────────────────────────────────────────────────────
def _normalise_article(raw: dict, ticker: str) -> Optional[dict]:
    """
    Normalise a yfinance news article into a consistent format.

    yfinance changed its news response structure in v0.2.40:
      Old: flat dict  — title, link, publisher, providerPublishTime (unix ts)
      New: nested     — content.title, content.canonicalUrl.url,
                        content.provider.displayName, content.pubDate (ISO str)

    Returns None if the article is missing essential fields.
    """
    # ── New nested format (yfinance >= 0.2.40) ──
    if "content" in raw and isinstance(raw["content"], dict):
        c = raw["content"]
        title = c.get("title", "").strip()
        if not title:
            return None
        # URL: prefer landingPageUrl (full article), fallback to canonicalUrl
        url_obj = c.get("canonicalUrl") or {}
        link    = url_obj.get("landingPageUrl") or url_obj.get("url") or "#"
        pub     = (c.get("provider") or {}).get("displayName", "")
        # pubDate is ISO 8601 string e.g. "2024-11-14T10:30:00Z"
        pub_date = c.get("pubDate", "")
        try:
            from datetime import timezone
            ts = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
            pub_time = ts.astimezone(timezone.utc).strftime("%d %b %Y  %H:%M")
        except Exception:
            pub_time = pub_date[:10] if pub_date else ""
        return {"title": title, "link": link, "publisher": pub,
                "pub_time": pub_time, "_ticker": ticker}

    # ── Old flat format (yfinance < 0.2.40) ──
    title = raw.get("title", "").strip()
    if not title:
        return None
    link = raw.get("link") or raw.get("url") or "#"
    pub  = raw.get("publisher") or raw.get("source") or ""
    try:
        ts       = raw.get("providerPublishTime") or 0
        pub_time = datetime.fromtimestamp(ts).strftime("%d %b %Y  %H:%M") if ts else ""
    except Exception:
        pub_time = ""
    return {"title": title, "link": link, "publisher": pub,
            "pub_time": pub_time, "_ticker": ticker,
            "_ts": raw.get("providerPublishTime", 0)}


def _fetch_news(tickers: List[str]) -> List[dict]:
    cache, all_news = st.session_state.news_cache, []
    for ticker in tickers:
        if ticker not in cache:
            try:
                raw_list = yf.Ticker(ticker).news or []
                cache[ticker] = raw_list
            except Exception as e:
                print(f"[Warning] News fetch failed for {ticker}: {e}")
                cache[ticker] = []
        for raw in cache[ticker][:6]:
            article = _normalise_article(raw, ticker)
            if article:
                all_news.append(article)

    # Sort by timestamp descending — use _ts for old format, pub_time string for new
    def _sort_key(a):
        if "_ts" in a and a["_ts"]:
            return a["_ts"]
        # Parse pub_time string as fallback sort key
        try:
            return datetime.strptime(a["pub_time"], "%d %b %Y  %H:%M").timestamp()
        except Exception:
            return 0
    all_news.sort(key=_sort_key, reverse=True)

    # Deduplicate by title
    seen, unique = set(), []
    for a in all_news:
        if a["title"] not in seen:
            seen.add(a["title"]); unique.append(a)
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
        st.caption("No news available for your holdings right now.")
        return
    for a in articles:
        st.markdown(f"""<div class="news-card">
  <span class="news-ticker">{a["_ticker"]}</span>
  <a href="{a["link"]}" target="_blank">{a["title"]}</a>
  <div class="news-meta">{a["publisher"]}{"  ·  " if a["publisher"] else ""}{a["pub_time"]}</div>
</div>""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
PAGES = ["Dashboard","Holdings","Add Transaction",
         "Benchmark","Portfolio Analysis","Quant Metrics","Tax & Income","Snapshot History"]

def render_sidebar():
    with st.sidebar:
        # ── Logo ──
        st.markdown("""
        <div style="padding:20px 4px 16px;border-bottom:1px solid #1a1a1a;margin-bottom:16px">
          <div style="font-family:'Syne',sans-serif;font-size:1.15rem;font-weight:800;
                      color:#e8e8e8;letter-spacing:-.02em;line-height:1">
            Portfolio
          </div>
          <div style="font-family:'DM Mono',monospace;font-size:.65rem;
                      color:#3a6a9a;letter-spacing:.18em;margin-top:3px">
            TRACKER
          </div>
        </div>""", unsafe_allow_html=True)

        # ── Nav ──
        page = st.radio("Nav", PAGES, label_visibility="collapsed")

        # ── Portfolio value summary ──
        holdings = portfolio().all_holdings()
        prices   = get_prices()
        tv  = sum(h.current_value(prices[h.ticker]) for h in holdings if prices.get(h.ticker))
        ti  = sum(h.total_invested for h in holdings)
        pnl = tv - ti
        pnl_pct    = pnl / ti * 100 if ti else 0
        pnl_colour = "#4caf7d" if pnl >= 0 else "#e05c5c"
        arrow      = "▲" if pnl >= 0 else "▼"

        st.markdown(f"""
        <div style="margin:16px 0 8px;padding:14px 16px;background:#0b0b0b;
                    border:1px solid #1a1a1a;border-radius:10px">
          <div style="font-family:'Inter',sans-serif;font-size:.62rem;font-weight:700;
                      letter-spacing:.1em;text-transform:uppercase;color:#3a3a3a;
                      margin-bottom:6px">Portfolio Value</div>
          <div style="font-family:'DM Mono',monospace;font-size:1.25rem;font-weight:500;
                      color:#e8e8e8;letter-spacing:-.02em">{fmt_cur(tv)}</div>
          <div style="font-family:'DM Mono',monospace;font-size:.78rem;
                      color:{pnl_colour};margin-top:4px">
            {arrow} {fmt_cur(pnl)} &nbsp;
            <span style="opacity:.7">({fmt_pct(pnl_pct)})</span>
          </div>
        </div>""", unsafe_allow_html=True)

        if st.button("⟳  Refresh Prices", use_container_width=True):
            invalidate_prices(); st.rerun()

        stale   = st.session_state.get("stale_prices", set())
        missing = st.session_state.get("missing_prices", set())
        if missing:
            st.error(f"No price data for: {', '.join(sorted(missing))}\nCheck ticker symbols or set a manual price.")
        elif stale:
            st.warning(f"Cached prices: {', '.join(sorted(stale))}\nYahoo Finance may be rate-limiting.")

        # ── User switcher ──
        st.markdown("""<div style="margin-top:16px;border-top:1px solid #1a1a1a;
                        padding-top:14px"></div>""", unsafe_allow_html=True)
        username = st.session_state.get("username", "")
        st.markdown(f"""
        <div style="font-family:'DM Mono',monospace;font-size:.68rem;
                    color:#3a3a3a;letter-spacing:.08em;margin-bottom:4px">SIGNED IN AS</div>
        <div style="font-family:'Inter',sans-serif;font-size:.9rem;font-weight:600;
                    color:#888;margin-bottom:8px">{username}</div>
        """, unsafe_allow_html=True)
        if st.button("Switch User", use_container_width=True):
            st.session_state.user_id  = None
            st.session_state.username = None
            st.session_state.portfolio = None
            st.rerun()

    return page

# ── Dashboard ─────────────────────────────────────────────────────────────────
def render_dashboard():
    holdings  = portfolio().all_holdings()
    prices    = get_prices()
    snapshots = portfolio().snapshots
    dividends = portfolio().all_dividends()

    if not holdings:
        _page_header("◈", "Dashboard", "Portfolio overview & key metrics")
        st.info("Your portfolio is empty. Go to **Add Transaction** to get started.")
        return

    # ── Core calculations ──────────────────────────────────────────────────────
    tv       = sum(h.current_value(prices[h.ticker]) for h in holdings if prices.get(h.ticker))
    ti       = sum(h.total_invested for h in holdings)
    pnl      = tv - ti
    pnl_pct  = pnl / ti * 100 if ti else 0
    total_div = sum(d.net_amount for d in dividends)
    total_comm = sum(h.total_commissions for h in holdings)
    n_gains  = sum(1 for h in holdings if prices.get(h.ticker) and h.unrealised_pnl(prices[h.ticker]) > 0)
    n_losses = len(holdings) - n_gains
    best_h   = max(holdings, key=lambda h: h.pnl_percent(prices[h.ticker]) if prices.get(h.ticker) else -999, default=None)
    worst_h  = min(holdings, key=lambda h: h.pnl_percent(prices[h.ticker]) if prices.get(h.ticker) else 999, default=None)

    # ── Row 1: Headline metrics ────────────────────────────────────────────────
    _page_header("◈", "Dashboard", "Portfolio overview & key metrics")

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Portfolio Value",  fmt_cur(tv))
    c2.metric("Total Invested",   fmt_cur(ti))
    c3.metric("Unrealised P&L",   fmt_cur(pnl),
              delta=fmt_pct(pnl_pct),
              delta_color="normal")
    c4.metric("Dividend Income",  fmt_cur(total_div),
              help="Total net dividends received (after withholding tax)")
    c5.metric("Commissions Paid", fmt_cur(total_comm),
              help="Total broker fees — already factored into cost basis")
    c6.metric("Holdings",
              f"{len(holdings)}",
              delta=f"{n_gains}↑  {n_losses}↓",
              delta_color="off",
              help=f"{n_gains} in profit, {n_losses} in loss")

    # ── Row 2: Best / Worst callout strip ─────────────────────────────────────
    if best_h and worst_h and best_h.ticker != worst_h.ticker:
        bp = prices.get(best_h.ticker)
        wp = prices.get(worst_h.ticker)
        st.markdown(f"""
        <div style="display:flex;gap:12px;margin:0.8rem 0">
          <div style="flex:1;background:#0d1f14;border:1px solid #1a3a22;border-radius:8px;padding:10px 16px">
            <span style="font-size:.7rem;color:#4caf7d;font-weight:700;letter-spacing:.08em">TOP PERFORMER</span><br>
            <span style="font-size:1.1rem;font-weight:700;color:#eee">{best_h.ticker}</span>
            <span style="font-size:.85rem;color:#aaa;margin-left:8px">{best_h.name}</span>
            <span style="float:right;font-size:1.05rem;font-weight:700;color:#4caf7d">
              {fmt_pct(best_h.pnl_percent(bp))}&nbsp;&nbsp;{fmt_cur(best_h.unrealised_pnl(bp))}
            </span>
          </div>
          <div style="flex:1;background:#1f0d0d;border:1px solid #3a1a1a;border-radius:8px;padding:10px 16px">
            <span style="font-size:.7rem;color:#e05c5c;font-weight:700;letter-spacing:.08em">LAGGING</span><br>
            <span style="font-size:1.1rem;font-weight:700;color:#eee">{worst_h.ticker}</span>
            <span style="font-size:.85rem;color:#aaa;margin-left:8px">{worst_h.name}</span>
            <span style="float:right;font-size:1.05rem;font-weight:700;color:#e05c5c">
              {fmt_pct(worst_h.pnl_percent(wp))}&nbsp;&nbsp;{fmt_cur(worst_h.unrealised_pnl(wp))}
            </span>
          </div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Row 3: Allocation donut + Holdings scorecard ───────────────────────────
    col_l, col_r = st.columns([1, 1.6])

    with col_l:
        _section("Allocation")
        labels, values, colours = [], [], []
        for i, h in enumerate(sorted(holdings, key=lambda h: h.current_value(prices.get(h.ticker,0) or 0), reverse=True)):
            if not (p := prices.get(h.ticker)): continue
            labels.append(h.ticker)
            values.append(h.current_value(p))
            colours.append(PALETTE[i % len(PALETTE)])
        if values:
            fig = go.Figure(go.Pie(
                labels=labels, values=values, hole=0.6,
                marker=dict(colors=colours, line=dict(color=BG, width=2)),
                textinfo="label+percent", textfont_size=11,
                hovertemplate="<b>%{label}</b><br>€%{value:,.2f}  ·  %{percent}<extra></extra>"))
            fig.update_layout(**_chart_layout(height=300))
            fig.update_layout(showlegend=False, margin=dict(t=10,b=10,l=10,r=10),
                annotations=[dict(
                    text=f"<b>{fmt_cur(tv)}</b>", x=0.5, y=0.52,
                    font_size=14, showarrow=False, font_color="#cccccc"),
                  dict(text="total", x=0.5, y=0.42,
                    font_size=11, showarrow=False, font_color="#555")])
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        _section("Holdings at a Glance")
        sorted_holdings = sorted(
            [h for h in holdings if prices.get(h.ticker)],
            key=lambda h: h.current_value(prices[h.ticker]),
            reverse=True
        )
        total_val = tv or 1
        for h in sorted_holdings:
            p      = prices[h.ticker]
            val    = h.current_value(p)
            upnl   = h.unrealised_pnl(p)
            pct    = h.pnl_percent(p)
            weight = val / total_val * 100
            bar_w  = max(2, int(abs(pct) / max(abs(h2.pnl_percent(prices[h2.ticker]))
                         for h2 in sorted_holdings if prices.get(h2.ticker)) * 60))
            colour = GAIN if upnl >= 0 else LOSS
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;padding:5px 0;
                        border-bottom:1px solid #1a1a1a">
              <div style="width:52px;font-size:.8rem;font-weight:700;color:#ccc">{h.ticker}</div>
              <div style="width:36px;font-size:.7rem;color:#555;text-align:right">{weight:.1f}%</div>
              <div style="flex:1;background:#111;border-radius:3px;height:6px;overflow:hidden">
                <div style="width:{bar_w}%;height:100%;background:{colour};border-radius:3px"></div>
              </div>
              <div style="width:80px;text-align:right;font-size:.82rem;color:#bbb">{fmt_cur(val)}</div>
              <div style="width:72px;text-align:right;font-size:.82rem;font-weight:600;
                          color:{colour}">{fmt_pct(pct)}</div>
              <div style="width:82px;text-align:right;font-size:.8rem;color:{colour}">{fmt_cur(upnl)}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Row 4: Snapshot value history OR invested vs value ────────────────────
    if len(snapshots) >= 2:
        _section("Portfolio Value Over Time")
        snap_dates  = [s.date for s in snapshots]
        snap_values = [s.total_value for s in snapshots]
        snap_inv    = [s.total_invested for s in snapshots]
        fig_snap = go.Figure()
        fig_snap.add_trace(go.Scatter(
            x=snap_dates, y=snap_values, name="Portfolio Value",
            line=dict(color=BLUE, width=2.5),
            fill="tozeroy", fillcolor="rgba(91,155,213,0.07)",
            hovertemplate="%{x}<br>Value: €%{y:,.2f}<extra></extra>"))
        fig_snap.add_trace(go.Scatter(
            x=snap_dates, y=snap_inv, name="Invested",
            line=dict(color="#555", width=1.5, dash="dot"),
            hovertemplate="%{x}<br>Invested: €%{y:,.2f}<extra></extra>"))
        # Add current value as a dot at today
        fig_snap.add_trace(go.Scatter(
            x=[date.today().isoformat()], y=[tv],
            name="Today", mode="markers",
            marker=dict(color=GAIN if pnl >= 0 else LOSS, size=10),
            hovertemplate=f"Today<br>Value: {fmt_cur(tv)}<extra></extra>"))
        fig_snap.update_layout(**_chart_layout(height=300))
        fig_snap.update_layout(yaxis=dict(gridcolor="#1e1e1e", tickprefix="€"))
        st.plotly_chart(fig_snap, use_container_width=True)
        st.caption(f"Based on {len(snapshots)} manual snapshots. Use **Snapshot History** to add more.")
    else:
        _section("Invested vs Current Value")
        rows = sorted(
            [(h.ticker, h.total_invested, h.current_value(p))
             for h in holdings if (p := prices.get(h.ticker))],
            key=lambda x: x[2], reverse=True)
        if rows:
            tickers, inv, cur = zip(*rows)
            fig_v = go.Figure()
            fig_v.add_trace(go.Bar(name="Cost Basis", x=list(tickers), y=list(inv),
                marker_color=BLUE, opacity=0.6,
                hovertemplate="<b>%{x}</b><br>Cost basis: €%{y:,.2f}<extra></extra>"))
            fig_v.add_trace(go.Bar(name="Current Value", x=list(tickers), y=list(cur),
                marker_color=[GAIN if c >= i else LOSS for c, i in zip(cur, inv)],
                hovertemplate="<b>%{x}</b><br>Value: €%{y:,.2f}<extra></extra>"))
            fig_v.update_layout(**_chart_layout(height=300))
            fig_v.update_layout(barmode="group", yaxis=dict(gridcolor="#1e1e1e", tickprefix="€"))
            st.plotly_chart(fig_v, use_container_width=True)
        if len(snapshots) == 0:
            st.caption("💡 Add manual snapshots in **Snapshot History** to track portfolio value over time.")
        else:
            st.caption("Add at least 2 snapshots in **Snapshot History** to see a value chart here.")

    # ── Row 5: P&L waterfall + Asset type breakdown ───────────────────────────
    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        _section("P&L by Holding")
        rows_pnl = sorted(
            [(h.ticker, h.unrealised_pnl(p), h.pnl_percent(p))
             for h in holdings if (p := prices.get(h.ticker))],
            key=lambda x: x[1], reverse=True)
        if rows_pnl:
            tickers_p, pnls_p, pcts_p = zip(*rows_pnl)
            fig_pnl = go.Figure(go.Bar(
                x=list(pnls_p), y=list(tickers_p), orientation="h",
                marker_color=[GAIN if p >= 0 else LOSS for p in pnls_p],
                text=[fmt_pct(p) for p in pcts_p],
                textposition="outside",
                textfont=dict(size=10, color="#aaa"),
                hovertemplate="<b>%{y}</b><br>P&L: €%{x:,.2f}<extra></extra>"))
            fig_pnl.update_layout(**_chart_layout(height=max(220, len(rows_pnl)*38)))
            fig_pnl.update_layout(xaxis=dict(gridcolor="#1e1e1e", tickprefix="€",
                                             zeroline=True, zerolinecolor="#444"))
            st.plotly_chart(fig_pnl, use_container_width=True)

    with col_b:
        _section("Exposure by Asset Type")
        type_vals: Dict[str, float] = {}
        for h in holdings:
            if (p := prices.get(h.ticker)):
                t = h.asset_type.upper()
                type_vals[t] = type_vals.get(t, 0) + h.current_value(p)
        if type_vals:
            labels_t  = list(type_vals.keys())
            values_t  = list(type_vals.values())
            colours_t = [PALETTE[i % len(PALETTE)] for i in range(len(labels_t))]
            fig_type = go.Figure(go.Pie(
                labels=labels_t, values=values_t, hole=0.5,
                marker=dict(colors=colours_t, line=dict(color=BG, width=2)),
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>€%{value:,.2f}  ·  %{percent}<extra></extra>"))
            fig_type.update_layout(**_chart_layout(height=max(220, len(rows_pnl)*38) if rows_pnl else 220))
            fig_type.update_layout(showlegend=True,
                legend=dict(orientation="h", y=-0.1))
            st.plotly_chart(fig_type, use_container_width=True)

    # ── Row 6: Dividends bar (only if any recorded) ───────────────────────────
    if dividends:
        st.divider()
        _section("Dividend Income by Year")
        by_year: Dict[str, float] = {}
        for d in dividends:
            y = d.date[:4]
            by_year[y] = by_year.get(y, 0) + d.net_amount
        years  = sorted(by_year.keys())
        fig_div = go.Figure(go.Bar(
            x=years, y=[by_year[y] for y in years],
            marker_color=GAIN,
            text=[fmt_cur(by_year[y]) for y in years],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Net dividends: €%{y:,.2f}<extra></extra>"))
        fig_div.update_layout(**_chart_layout(height=240))
        fig_div.update_layout(yaxis=dict(gridcolor="#1e1e1e", tickprefix="€"))
        st.plotly_chart(fig_div, use_container_width=True)

    # ── Row 7: News ───────────────────────────────────────────────────────────
    st.divider()
    _render_news([h.ticker for h in holdings])



# ── Holdings ──────────────────────────────────────────────────────────────────
def render_holdings():
    _page_header("▦", "Holdings", "Positions, transactions & price history")
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
        _xlsx = _excel_bytes(holdings, prices, portfolio().all_dividends())
        if _xlsx:
            st.download_button("⬇  Download Excel",
                data=_xlsx,
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
                    errors = validate_transaction_list(h.ticker, new_txns)
                    if errors:
                        for e in errors:
                            st.error(e)
                    else:
                        try:
                            portfolio().replace_transactions(h.ticker, new_txns)
                            invalidate_prices()
                            st.success("✓ Saved"); st.rerun()
                        except Exception as e:
                            st.error(f"Failed to save changes: {e}")

            col_del, col_conf, _ = st.columns([1, 1.5, 2.5])
            with col_del:
                remove_clicked = st.button(f"🗑  Remove {h.ticker}",
                                           key=f"remove_{h.ticker}",
                                           help="Permanently delete this holding and all its transactions")
            if remove_clicked:
                st.session_state[f"confirm_remove_{h.ticker}"] = True
            if st.session_state.get(f"confirm_remove_{h.ticker}"):
                with col_conf:
                    st.warning(f"Delete all {h.ticker} data?")
                c_yes, c_no = st.columns(2)
                if c_yes.button("Yes, delete", key=f"yes_remove_{h.ticker}", type="primary"):
                    portfolio().remove_holding(h.ticker)
                    st.session_state.pop(f"confirm_remove_{h.ticker}", None)
                    invalidate_prices(); st.rerun()
                if c_no.button("Cancel", key=f"no_remove_{h.ticker}"):
                    st.session_state.pop(f"confirm_remove_{h.ticker}", None)
                    st.rerun()


def _chart_price_history(ticker: str, period: str, transactions=None):
    try:
        hist = yf.Ticker(ticker).history(period=period)
    except Exception as e:
        msg = str(e).lower()
        if "rate" in msg or "429" in msg or "too many" in msg:
            st.caption("⚠ Price history unavailable — Yahoo Finance rate limit. Try again in a few minutes.")
        else:
            st.caption(f"⚠ Could not load price history for {ticker}: {e}")
        return
    if hist.empty:
        st.caption(f"No historical price data available for {ticker}. "
                   f"The ticker may be invalid or unsupported by Yahoo Finance.")
        return

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
    _page_header("＋", "Add Transaction", "Record a buy or sell · or import from Portfolio Performance")

    tab_manual, tab_import = st.tabs(["Manual Entry", "Import from Portfolio Performance"])

    # ── Tab 1: Manual entry ───────────────────────────────────────────────────
    with tab_manual:
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
                h = existing.get(ticker)
                errors = []
                errors += validate_ticker(ticker)
                if not errors:
                    if ticker not in existing:
                        errors += validate_name(name)
                    errors += validate_transaction(
                        action=action.lower(), quantity=quantity, price=price,
                        txn_date=txn_date, commission=commission, holding=h,
                    )
                if errors:
                    for e in errors: st.error(e)
                else:
                    try:
                        portfolio().add_transaction(
                            ticker=ticker, name=h.name if h else name,
                            asset_type=h.asset_type if h else asset_type,
                            action=action.lower(), quantity=quantity,
                            price=price, date=str(txn_date), commission=commission,
                        )
                        invalidate_prices()
                        st.session_state.news_cache = {}
                        st.success(f"✓ {action} {quantity:,.4f} × {ticker} @ {fmt_cur(price)} recorded.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to save transaction: {e}")

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

    # ── Tab 2: Portfolio Performance import ───────────────────────────────────
    with tab_import:
        st.markdown(
            "Upload the **Alle Buchungen** CSV export from Portfolio Performance. "
            "Buys, sells and dividends are imported. Cash flows and fractional "
            "deliveries (Einlieferung) are automatically skipped.")
        st.caption("Export path in PP: File → Export → Alle Buchungen (CSV)")

        uploaded = st.file_uploader("Choose CSV file", type=["csv"],
                                    label_visibility="collapsed")
        if not uploaded:
            return

        file_bytes = uploaded.read()
        rows, parse_warnings = parse_pp_csv(file_bytes)

        for w in parse_warnings:
            st.warning(w)

        if not rows:
            st.error("No importable rows found. Check that this is a valid Portfolio Performance export.")
            return

        # Preview
        buys      = [r for r in rows if r.row_type == "buy"]
        sells     = [r for r in rows if r.row_type == "sell"]
        dividends = [r for r in rows if r.row_type == "dividend"]

        _section("Preview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Buys",      len(buys))
        c2.metric("Sells",     len(sells))
        c3.metric("Dividends", len(dividends))

        preview_data = []
        for r in rows:
            preview_data.append({
                "Type":       r.row_type.capitalize(),
                "Symbol":     r.symbol,
                "Name":       r.name[:28] + "…" if len(r.name) > 28 else r.name,
                "Date":       r.date,
                "Qty":        f"{r.quantity:,.6f}" if r.row_type != "dividend" else "—",
                "Price":      f"€{r.price:,.4f}"  if r.price  else "—",
                "Amount":     f"€{r.amount:,.2f}",
                "Commission": f"€{r.commission:,.2f}" if r.commission else "—",
            })
        st.dataframe(pd.DataFrame(preview_data),
                     use_container_width=True, hide_index=True, height=280)

        # Asset type per ticker
        _section("Asset Types")
        st.caption("Confirm the asset type for each ticker — check that ETFs are marked as 'etf'.")
        from tracker.importer import _infer_asset_type
        symbols = sorted({r.symbol for r in rows})
        asset_type_map = {}
        cols = st.columns(min(len(symbols), 4))
        for i, sym in enumerate(symbols):
            existing_h = portfolio().get_holding(sym)
            default = existing_h.asset_type if existing_h else _infer_asset_type(sym)
            with cols[i % 4]:
                asset_type_map[sym] = st.selectbox(
                    sym, ["stock", "etf", "crypto"],
                    index=["stock","etf","crypto"].index(default)
                          if default in ["stock","etf","crypto"] else 0,
                    key=f"import_atype_{sym}")

        # Duplicate warning
        _section("Import")
        p = portfolio()
        n_txns = sum(len(h.transactions) for h in p.holdings.values())
        n_divs = len(p.all_dividends())
        if n_txns > 0 or n_divs > 0:
            st.info(
                f"Your portfolio already has **{n_txns}** transaction(s) and "
                f"**{n_divs}** dividend(s). Duplicates are skipped automatically.")

        if st.button("⇩  Import Now", type="primary"):
            with st.spinner("Importing…"):
                try:
                    result = execute_import(rows, portfolio(), asset_type_map)
                    invalidate_prices()
                except Exception as e:
                    st.error(f"Import failed: {e}"); return

            if result.total_imported > 0:
                st.success(
                    f"✓ Imported **{result.imported_buys}** buy(s), "
                    f"**{result.imported_sells}** sell(s), "
                    f"**{result.imported_dividends}** dividend(s).")
            if result.skipped_duplicate > 0:
                st.warning(f"{result.skipped_duplicate} duplicate(s) skipped.")
            for err in result.errors:
                st.error(f"Error: {err}")
            if result.total_imported > 0:
                st.rerun()

# ── Benchmark ─────────────────────────────────────────────────────────────────
def render_benchmark():
    _page_header("⟁", "Benchmark", "Portfolio vs market indices")

    port       = portfolio()
    start_date = get_portfolio_start_date(port)
    if start_date is None:
        st.info("Add some transactions first."); return

    st.info(f"Benchmarking from **{start_date.strftime('%d %b %Y')}** · Returns are **time-weighted** (cash inflows from regular savings plans are neutralised so performance is comparable to a buy-and-hold index)")
    selected = st.multiselect("Compare against", list(INDICES.keys()), list(INDICES.keys()))

    cache = st.session_state.bench_cache
    has_cache = "port_series" in cache

    col_run, col_clear = st.columns([1, 1])
    run_clicked   = col_run.button("Run Benchmark", type="primary", disabled=has_cache)
    clear_clicked = col_clear.button("🔄 Re-fetch Data", disabled=not has_cache,
                                     help="Clear cached data and download fresh from Yahoo Finance")

    if clear_clicked:
        st.session_state.bench_cache = {}
        st.rerun()

    if not has_cache and not run_clicked:
        st.caption("Click **Run Benchmark** to fetch data. Results are cached so "
                   "re-runs won't count against the Yahoo Finance rate limit.")
        return

    if run_clicked and not has_cache:
        with st.spinner("Building portfolio value series…"):
            port_series, port_warn = build_portfolio_value_series(port, start_date)
        if port_series is None:
            st.error(f"Could not build portfolio series: {port_warn}"); return
        if port_warn:
            st.warning(port_warn)

        index_series: Dict[str, pd.Series] = {}
        for name in selected:
            with st.spinner(f"Fetching {name}…"):
                s, s_err = fetch_index_series(INDICES[name], start_date)
            if s is not None: index_series[name] = s
            else:             st.warning(f"Could not fetch {name}: {s_err}")

        # Store in session cache — survives Streamlit reruns within the session
        st.session_state.bench_cache = {
            "port_series":  port_series,
            "index_series": index_series,
            "selected":     selected,
        }
        cache = st.session_state.bench_cache

    port_series  = cache["port_series"]
    index_series = cache.get("index_series", {})
    # Filter to currently selected indices
    index_series = {n: s for n, s in index_series.items() if n in selected}

    norm_port    = port_series   # already rebased to 100 (TWR)
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
    _section("Cumulative Return %")
    ret_port = norm_port - 100
    fig_pnl = go.Figure()
    fig_pnl.add_trace(_scatter(ret_port.index, ret_port.values, "My Portfolio", GAIN,
        fill="tozeroy", fillcolor="rgba(76,175,125,0.08)",
        tmpl="<b>Portfolio</b><br>%{y:+.2f}%<extra></extra>"))
    for n, s in norm_indices.items():
        ret_s = s - 100
        fig_pnl.add_trace(_scatter(ret_s.index, ret_s.values, n,
            BENCH_COLOURS.get(n, "#aaa"), 1.5, dash="dot",
            tmpl=f"<b>{n}</b><br>%{{y:+.2f}}%<extra></extra>"))
    fig_pnl.add_hline(y=0, line_color="#333", line_dash="dash", line_width=1)
    fig_pnl.update_layout(**_chart_layout(height=300))
    fig_pnl.update_layout(yaxis=dict(gridcolor="#1e1e1e", ticksuffix="%"))
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
    all_stats = [compute_stats(norm_port, "My Portfolio")] + [
        compute_stats(norm_indices[n].dropna(), n)
        for n in index_series if n in norm_indices
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

    _page_header("◎", "Portfolio Analysis", "Diversification, correlation & stress testing")
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

    corr_cache    = st.session_state.get("corr_cache", {})
    corr_key      = f"{','.join(sorted(tickers))}_{period}"
    has_corr      = corr_cache.get("key") == corr_key

    c_run, c_clear = st.columns([1,1])
    run_corr    = c_run.button("Run Correlation Analysis", type="primary",
                                key="run_corr", disabled=has_corr)
    clear_corr  = c_clear.button("🔄 Re-fetch", key="clear_corr", disabled=not has_corr,
                                  help="Download fresh data")
    if clear_corr:
        st.session_state.corr_cache = {}; st.rerun()

    if not has_corr and not run_corr:
        st.caption("Click **Run Correlation Analysis** to fetch data. Results are cached.")
    elif run_corr and not has_corr:
        with st.spinner("Downloading price data…"):
            returns, ret_err = fetch_return_matrix(tickers, period_map[period])
        if returns is None:
            st.error(f"Could not fetch price data: {ret_err}")
        else:
            st.session_state.corr_cache = {"key": corr_key, "returns": returns}
            has_corr = True

    if has_corr:
        returns = st.session_state.corr_cache["returns"]
        if True:
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
    _page_header("◷", "Snapshot History", "Manual portfolio value checkpoints over time")
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
                try:
                    snap = portfolio().add_snapshot(tv, ti, note)
                    st.success(f"✓ Saved {fmt_cur(snap.total_value)} on {snap.date}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to save snapshot: {e}")

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
            if st.button("✕", key=f"del_snap_{real_idx}", help="Delete snapshot"):
                st.session_state[f"confirm_snap_{real_idx}"] = True
        if st.session_state.get(f"confirm_snap_{real_idx}"):
            st.warning(f"Delete snapshot from {s.date}?")
            cy2, cn2 = st.columns(2)
            if cy2.button("Yes, delete", key=f"yes_snap_{real_idx}", type="primary"):
                portfolio().delete_snapshot(real_idx)
                st.session_state.pop(f"confirm_snap_{real_idx}", None); st.rerun()
            if cn2.button("Cancel", key=f"no_snap_{real_idx}"):
                st.session_state.pop(f"confirm_snap_{real_idx}", None); st.rerun()

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

    _page_header("∿", "Quant Metrics", "Sharpe, Sortino, Beta, VaR & rolling analytics")
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
    quant_key = f"{','.join(sorted(h.ticker for h in holdings))}_{bench_name}_{period}"
    qcache    = st.session_state.quant_cache
    has_qdata = qcache.get("key") == quant_key

    q_run, q_clear = st.columns([1,1])
    run_quant   = q_run.button("Compute Metrics", type="primary", disabled=has_qdata)
    clear_quant = q_clear.button("🔄 Re-fetch Data", disabled=not has_qdata,
                                  help="Download fresh data from Yahoo Finance")
    if clear_quant:
        st.session_state.quant_cache = {}; st.rerun()

    if not has_qdata and not run_quant:
        st.caption("Click **Compute Metrics** to fetch data. Results are cached for the session.")
        return

    bench_ticker = INDICES[bench_name]
    all_tickers  = [h.ticker for h in holdings] + [bench_ticker]

    if run_quant and not has_qdata:
        with st.spinner("Downloading weekly price data…"):
            returns_df, dl_err = fetch_weekly_returns(all_tickers, period_map[period])
        if returns_df is None:
            st.error(f"Could not download price data: {dl_err}"); return
        st.session_state.quant_cache = {"key": quant_key, "returns_df": returns_df}
        qcache = st.session_state.quant_cache

    returns_df = qcache["returns_df"]

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

    _page_header("⊕", "Tax & Income", "German Abgeltungsteuer · FIFO · Sparerpauschbetrag")

    holdings  = portfolio().holdings
    dividends = portfolio().all_dividends()

    if not holdings:
        st.info("Add some transactions first."); return

    # Pre-compute tax summaries with caching (invalidates when data changes)
    _p_key = _portfolio_key()
    _d_key = _dividends_key()

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
                errors = validate_dividend(div_amount, div_wht, div_date)
                if errors:
                    for e in errors:
                        st.error(e)
                else:
                    portfolio().add_dividend(div_ticker, str(div_date), div_amount, div_wht)
                    st.success(f"✓ Dividend of {fmt_cur(div_amount)} (net: {fmt_cur(div_amount - div_wht)}) recorded for {div_ticker}")
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
    all_summaries = _cached_tax_summary(_p_key, _d_key)
    summary = all_summaries.get(sel_year) or year_summary(sel_year, holdings, dividends)

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
    all_div_rows = db().get_dividends_with_ids(st.session_state.user_id)
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
                if st.button("✕", key=f"del_div_{row['id']}", help="Delete dividend"):
                    st.session_state[f"confirm_div_{row['id']}"] = True
            if st.session_state.get(f"confirm_div_{row['id']}"):
                st.warning(f"Delete this dividend record?")
                cy, cn = st.columns(2)
                if cy.button("Yes, delete", key=f"yes_div_{row['id']}", type="primary"):
                    portfolio().delete_dividend(row["id"])
                    st.session_state.pop(f"confirm_div_{row['id']}", None); st.rerun()
                if cn.button("Cancel", key=f"no_div_{row['id']}"):
                    st.session_state.pop(f"confirm_div_{row['id']}", None); st.rerun()

    # Multi-year overview
    if len(active_years) > 1:
        st.divider()
        _section("Multi-Year Overview")
        yearly = [all_summaries.get(y) or year_summary(y, holdings, dividends) for y in sorted(active_years)]
        st.dataframe(pd.DataFrame([{
            "Year":           s.year,
            "Gains":          fmt_cur(s.realised_gains),
            "Losses":        f"-{fmt_cur(s.realised_losses)}",
            "Dividends":      fmt_cur(s.dividend_income),
            "Net Taxable":    fmt_cur(s.net_taxable),
            "Allowance Used": fmt_cur(s.allowance_used),
            "Est. Tax":       fmt_cur(s.tax_owed),
        } for s in yearly]), use_container_width=True, hide_index=True)


# ── Import ────────────────────────────────────────────────────────────────────
def render_import():
    _page_header("⇩", "Import", "Import transactions from Portfolio Performance")

    st.markdown(
        "Upload the **Alle Buchungen** CSV export from Portfolio Performance. "
        "Buys, sells and dividends are imported. Cash flows (Einlage/Entnahme) "
        "and fractional deliveries (Einlieferung) are automatically skipped.",
        unsafe_allow_html=False)

    uploaded = st.file_uploader("Choose CSV file", type=["csv"],
                                 label_visibility="collapsed")
    if not uploaded:
        st.caption("Export from Portfolio Performance: File → Export → Alle Buchungen (CSV)")
        return

    file_bytes = uploaded.read()
    rows, parse_warnings = parse_pp_csv(file_bytes)

    for w in parse_warnings:
        st.warning(w)

    if not rows:
        st.error("No importable rows found. Check that the file is a valid Portfolio Performance CSV export.")
        return

    # ── Preview ────────────────────────────────────────────────────────────────
    buys      = [r for r in rows if r.row_type == "buy"]
    sells     = [r for r in rows if r.row_type == "sell"]
    dividends = [r for r in rows if r.row_type == "dividend"]

    _section("Preview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Buys",      len(buys))
    c2.metric("Sells",     len(sells))
    c3.metric("Dividends", len(dividends))

    # Show preview table
    preview_data = []
    for r in rows:
        preview_data.append({
            "Type":    r.row_type.capitalize(),
            "Symbol":  r.symbol,
            "Name":    r.name[:30] + "…" if len(r.name) > 30 else r.name,
            "Date":    r.date,
            "Qty":     f"{r.quantity:,.6f}" if r.row_type != "dividend" else "—",
            "Price":   f"€{r.price:,.4f}" if r.price else "—",
            "Amount":  f"€{r.amount:,.2f}",
            "Commission": f"€{r.commission:,.2f}" if r.commission else "—",
        })

    st.dataframe(
        pd.DataFrame(preview_data),
        use_container_width=True, hide_index=True, height=300)

    # ── Asset type overrides ───────────────────────────────────────────────────
    symbols = sorted({r.symbol for r in rows})
    _section("Asset Types")
    st.caption("Confirm the asset type for each ticker. The app auto-detects most — check crypto and ETFs.")

    asset_type_map = {}
    cols = st.columns(min(len(symbols), 4))
    for i, sym in enumerate(symbols):
        from tracker.importer import _infer_asset_type
        default = _infer_asset_type(sym)
        # Check if already in portfolio
        existing = portfolio().get_holding(sym)
        if existing:
            default = existing.asset_type
        with cols[i % 4]:
            asset_type_map[sym] = st.selectbox(
                sym, ["stock", "etf", "crypto"],
                index=["stock", "etf", "crypto"].index(default) if default in ["stock", "etf", "crypto"] else 0,
                key=f"import_atype_{sym}")

    # ── Execute ────────────────────────────────────────────────────────────────
    _section("Import")
    existing_p = portfolio()
    n_existing_txns = sum(len(h.transactions) for h in existing_p.holdings.values())
    n_existing_divs = len(existing_p.all_dividends())

    if n_existing_txns > 0 or n_existing_divs > 0:
        st.info(
            f"Your portfolio already has **{n_existing_txns} transaction(s)** and "
            f"**{n_existing_divs} dividend(s)**. "
            f"Duplicates (same ticker + date + quantity) will be skipped automatically.")

    if st.button("⇩  Import Now", type="primary", use_container_width=False):
        with st.spinner("Importing…"):
            try:
                result = execute_import(rows, portfolio(), asset_type_map)
                invalidate_prices()
            except Exception as e:
                st.error(f"Import failed: {e}")
                return

        # ── Results ────────────────────────────────────────────────────────────
        if result.total_imported > 0:
            st.success(
                f"✓ Imported **{result.imported_buys}** buy(s), "
                f"**{result.imported_sells}** sell(s), "
                f"**{result.imported_dividends}** dividend(s).")

        col_a, col_b = st.columns(2)
        if result.skipped_duplicate > 0:
            col_a.warning(f"{result.skipped_duplicate} duplicate(s) skipped.")
        if result.skipped_type > 0:
            col_b.info(f"{result.skipped_type} unsupported row type(s) skipped.")
        for err in result.errors:
            st.error(f"Error: {err}")

        if result.total_imported > 0:
            st.rerun()


def _render_crash(error: Exception, context: str = ""):
    """Friendly full-page error for unrecoverable crashes."""
    import traceback
    tb = traceback.format_exc()
    st.markdown(f"""
    <div style="background:#1a0a0a;border:1px solid #4a1a1a;border-radius:12px;
                padding:28px 32px;margin:2rem 0">
      <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;
                  color:#e05c5c;margin-bottom:8px">Something went wrong</div>
      <div style="font-family:'Inter',sans-serif;font-size:.85rem;color:#888;
                  margin-bottom:16px">{context or "An unexpected error occurred."}</div>
      <details>
        <summary style="font-family:'DM Mono',monospace;font-size:.75rem;
                        color:#555;cursor:pointer">Show technical details</summary>
        <pre style="font-family:'DM Mono',monospace;font-size:.72rem;color:#666;
                    margin-top:10px;white-space:pre-wrap;word-break:break-all">{tb}</pre>
      </details>
    </div>""", unsafe_allow_html=True)
    st.info("Try **refreshing the page**. If the problem persists, check that `portfolio.db` is not open in another program.")


def main():
    # Show login screen if no user selected — gate everything else behind it
    if not _render_login():
        return
    try:
        page = render_sidebar()
    except Exception as e:
        st.error(f"Sidebar failed to load: {e}")
        return
    try:
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
    except Exception as e:
        _render_crash(e, f"Error on page '{page}': {e}")

if __name__ == "__main__":
    main()

# ── Quant Metrics ─────────────────────────────────────────────────────────────
