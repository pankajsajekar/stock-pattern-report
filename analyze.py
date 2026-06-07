#!/usr/bin/env python3
"""
Indian Stock Chart Pattern Analyzer
-----------------------------------
Fetches NSE/BSE historical data, computes technical indicators, detects classic
chart patterns, and writes a single HTML report with INTERACTIVE candlestick
charts (zoom/pan/hover, toggleable SMA/volume/RSI) and per-stock trading-method
suggestions.

Usage:
    python analyze.py                         # uses stocks.txt, 6mo period
    python analyze.py --stocks stocks.txt --period 6mo --out report.html
    python analyze.py --symbols RELIANCE.NS TCS.NS
    python analyze.py --offline-charts        # embed plotly.js (works without internet)

Patterns detected (heuristic, see notes in report):
    Double Top / Double Bottom, Head & Shoulders (+ Inverse), Bull/Bear Flag,
    Cup & Handle, Golden/Death Cross, Support & Resistance levels.

NOTE: This is a technical-analysis tool for research/education. It is NOT
investment advice. Pattern detection is heuristic and can produce false signals.
"""

import argparse
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")  # silence yfinance noise

# ---- Heavy / optional imports guarded so failures give a clear message --------
try:
    import yfinance as yf
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots
    from scipy.signal import find_peaks
    from jinja2 import Template
except ImportError as e:  # pragma: no cover
    sys.exit(
        f"Missing dependency: {e.name}. "
        "Install with: pip install -r requirements.txt"
    )


# =============================================================================
# Data structures
# =============================================================================
@dataclass
class PatternHit:
    name: str
    bias: str          # "bullish" | "bearish" | "neutral"
    confidence: float  # 0..1 heuristic confidence
    note: str = ""


@dataclass
class StockResult:
    symbol: str
    ok: bool = True
    error: str = ""
    last_price: float = 0.0
    change_pct: float = 0.0          # period change %
    indicators: dict = field(default_factory=dict)
    patterns: list = field(default_factory=list)  # list[PatternHit]
    support: list = field(default_factory=list)
    resistance: list = field(default_factory=list)
    suggestion: str = ""
    overall_bias: str = "neutral"
    chart_html: str = ""             # interactive plotly div


# =============================================================================
# Data fetching
# =============================================================================
def fetch_history(symbol: str, period: str, interval: str = "1d", retries: int = 2):
    """Fetch OHLCV. Returns a cleaned DataFrame or raises ValueError."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            df = yf.download(
                symbol, period=period, interval=interval,
                auto_adjust=True, progress=False, threads=False,
            )
            # yfinance may return MultiIndex columns for a single ticker
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            if df.empty or len(df) < 20:
                raise ValueError("no/insufficient data returned")
            need = {"Open", "High", "Low", "Close", "Volume"}
            if not need.issubset(set(df.columns)):
                raise ValueError(f"missing columns: {need - set(df.columns)}")
            return df[["Open", "High", "Low", "Close", "Volume"]].copy()
        except Exception as e:  # network/empty/etc — retry with backoff
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise ValueError(f"fetch failed: {last_err}")


# =============================================================================
# Indicators (computed with pandas/numpy — no external TA lib needed)
# =============================================================================
def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def _macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def _bollinger(close: pd.Series, period=20, std=2):
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std()
    return mid + std * sd, mid, mid - std * sd


def compute_series(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of all indicator series, indexed like df.

    Used both for charting (full series) and the scalar snapshot (last values)."""
    close = df["Close"]
    macd_line, macd_sig, macd_hist = _macd(close)
    bb_up, bb_mid, bb_low = _bollinger(close)
    return pd.DataFrame({
        "rsi": _rsi(close),
        "macd": macd_line, "macd_signal": macd_sig, "macd_hist": macd_hist,
        "bb_upper": bb_up, "bb_mid": bb_mid, "bb_lower": bb_low,
        "sma20": close.rolling(20).mean(),
        "sma50": close.rolling(50).mean(),
        "sma200": close.rolling(200).mean(),
        "vol_avg": df["Volume"].rolling(20).mean(),
    }, index=df.index)


def indicator_snapshot(df: pd.DataFrame, s: pd.DataFrame) -> dict:
    """Scalar last-values of each indicator for the summary panel."""
    def last(col):
        v = s[col].dropna()
        return float(v.iloc[-1]) if len(v) else float("nan")

    vol_avg = last("vol_avg")
    return {
        "rsi": last("rsi"),
        "macd": last("macd"),
        "macd_signal": last("macd_signal"),
        "macd_hist": last("macd_hist"),
        "bb_upper": last("bb_upper"),
        "bb_lower": last("bb_lower"),
        "sma20": last("sma20"),
        "sma50": last("sma50"),
        "sma200": last("sma200"),
        "vol_ratio": (float(df["Volume"].iloc[-1]) / vol_avg) if vol_avg else float("nan"),
    }


# =============================================================================
# Pattern detection (heuristic)
# =============================================================================
def _peaks_troughs(close: np.ndarray):
    """Return indices of significant peaks and troughs via prominence filtering."""
    rng = np.nanmax(close) - np.nanmin(close)
    prom = max(rng * 0.03, 1e-9)         # ignore wiggles < 3% of range
    dist = max(len(close) // 20, 3)      # min spacing between extrema
    peaks, _ = find_peaks(close, prominence=prom, distance=dist)
    troughs, _ = find_peaks(-close, prominence=prom, distance=dist)
    return peaks, troughs


def detect_double_top_bottom(close, peaks, troughs):
    hits = []
    last = close[-1]
    if len(peaks) >= 2:
        p1, p2 = peaks[-2], peaks[-1]
        h1, h2 = close[p1], close[p2]
        if abs(h1 - h2) / max(h1, h2) < 0.03:
            mids = [t for t in troughs if p1 < t < p2]
            if mids:
                neck = close[mids[-1]]
                conf = 0.55 + (0.25 if last < neck else 0.0)
                hits.append(PatternHit(
                    "Double Top", "bearish", min(conf, 0.85),
                    f"Twin peaks ~{h1:.1f}; neckline {neck:.1f}"
                    + (" (broken — confirmed)" if last < neck else " (watch for break)")))
    if len(troughs) >= 2:
        t1, t2 = troughs[-2], troughs[-1]
        l1, l2 = close[t1], close[t2]
        if abs(l1 - l2) / max(l1, l2) < 0.03:
            mids = [p for p in peaks if t1 < p < t2]
            if mids:
                neck = close[mids[-1]]
                conf = 0.55 + (0.25 if last > neck else 0.0)
                hits.append(PatternHit(
                    "Double Bottom", "bullish", min(conf, 0.85),
                    f"Twin troughs ~{l1:.1f}; neckline {neck:.1f}"
                    + (" (broken — confirmed)" if last > neck else " (watch for break)")))
    return hits


def detect_head_shoulders(close, peaks, troughs):
    hits = []
    if len(peaks) >= 3:
        l, h, r = peaks[-3], peaks[-2], peaks[-1]
        hl, hh, hr = close[l], close[h], close[r]
        if hh > hl and hh > hr and abs(hl - hr) / max(hl, hr) < 0.05:
            hits.append(PatternHit(
                "Head & Shoulders", "bearish", 0.7,
                f"Head {hh:.1f} above shoulders {hl:.1f}/{hr:.1f} — reversal top"))
    if len(troughs) >= 3:
        l, h, r = troughs[-3], troughs[-2], troughs[-1]
        ll, lh, lr = close[l], close[h], close[r]
        if lh < ll and lh < lr and abs(ll - lr) / max(ll, lr) < 0.05:
            hits.append(PatternHit(
                "Inverse Head & Shoulders", "bullish", 0.7,
                f"Head {lh:.1f} below shoulders {ll:.1f}/{lr:.1f} — reversal bottom"))
    return hits


def detect_flag(close):
    """Bull/Bear flag: strong directional pole then a tight consolidation."""
    hits = []
    n = len(close)
    if n < 30:
        return hits
    pole = close[-30:-10]
    flag = close[-10:]
    pole_move = (pole[-1] - pole[0]) / pole[0]
    flag_range = (flag.max() - flag.min()) / flag.mean()
    if flag_range < 0.06:
        if pole_move > 0.08:
            hits.append(PatternHit("Bull Flag", "bullish", 0.6,
                                   f"+{pole_move*100:.0f}% pole then tight range — continuation"))
        elif pole_move < -0.08:
            hits.append(PatternHit("Bear Flag", "bearish", 0.6,
                                   f"{pole_move*100:.0f}% pole then tight range — continuation"))
    return hits


def detect_cup_handle(close):
    """Rough Cup & Handle: U-shaped recovery to prior high, then small handle dip."""
    hits = []
    n = len(close)
    if n < 60:
        return hits
    cup = close[-60:-10]
    handle = close[-10:]
    left, bottom, right = cup[0], cup.min(), cup[-1]
    bottom_idx = int(np.argmin(cup))
    centered = 0.25 * len(cup) < bottom_idx < 0.75 * len(cup)
    rims_equal = abs(left - right) / max(left, right) < 0.07
    depth = (max(left, right) - bottom) / max(left, right)
    handle_dip = (right - handle.min()) / right
    if centered and rims_equal and 0.1 < depth < 0.5 and 0 < handle_dip < 0.12:
        hits.append(PatternHit("Cup & Handle", "bullish", 0.6,
                               f"U-base depth {depth*100:.0f}% + shallow handle — bullish continuation"))
    return hits


def detect_crosses(s: pd.DataFrame):
    """Golden/Death cross from SMA50 vs SMA200 over the last ~10 sessions."""
    hits = []
    df = s[["sma50", "sma200"]].dropna()
    if len(df) < 11:
        return hits
    a = df["sma50"].values
    b = df["sma200"].values
    window = min(10, len(df) - 1)
    for i in range(len(df) - window, len(df)):
        if a[i - 1] <= b[i - 1] and a[i] > b[i]:
            hits.append(PatternHit("Golden Cross", "bullish", 0.75,
                                   "SMA50 crossed above SMA200 — major bullish signal"))
            break
        if a[i - 1] >= b[i - 1] and a[i] < b[i]:
            hits.append(PatternHit("Death Cross", "bearish", 0.75,
                                   "SMA50 crossed below SMA200 — major bearish signal"))
            break
    return hits


def support_resistance(close, peaks, troughs, n_levels=3):
    """Cluster recent extrema into support (troughs) and resistance (peaks)."""
    def cluster(vals):
        if len(vals) == 0:
            return []
        vals = sorted(vals)
        groups, cur = [], [vals[0]]
        tol = (max(close) - min(close)) * 0.02
        for v in vals[1:]:
            if v - cur[-1] <= tol:
                cur.append(v)
            else:
                groups.append(np.mean(cur)); cur = [v]
        groups.append(np.mean(cur))
        return groups

    res = cluster([close[p] for p in peaks])
    sup = cluster([close[t] for t in troughs])
    last = close[-1]
    res = sorted([r for r in res if r >= last])[:n_levels]
    sup = sorted([s for s in sup if s <= last], reverse=True)[:n_levels]
    return [round(float(s), 2) for s in sup], [round(float(r), 2) for r in res]


def detect_patterns(df: pd.DataFrame, s: pd.DataFrame):
    close = df["Close"].values.astype(float)
    peaks, troughs = _peaks_troughs(close)
    hits = []
    hits += detect_double_top_bottom(close, peaks, troughs)
    hits += detect_head_shoulders(close, peaks, troughs)
    hits += detect_flag(close)
    hits += detect_cup_handle(close)
    hits += detect_crosses(s)
    sup, res = support_resistance(close, peaks, troughs)
    return hits, sup, res


# =============================================================================
# Scoring & suggestions
# =============================================================================
def build_suggestion(ind: dict, patterns, sup, res, last_price):
    """Combine indicators + patterns into a bias and a plain-language plan."""
    score = 0.0
    for p in patterns:
        w = p.confidence
        score += w if p.bias == "bullish" else (-w if p.bias == "bearish" else 0)

    rsi_v = ind.get("rsi", 50)
    if rsi_v < 30:
        score += 0.4
    elif rsi_v > 70:
        score -= 0.4
    if ind.get("macd_hist", 0) > 0:
        score += 0.2
    else:
        score -= 0.2
    if not np.isnan(ind.get("sma50", np.nan)) and not np.isnan(ind.get("sma200", np.nan)):
        score += 0.2 if ind["sma50"] > ind["sma200"] else -0.2

    if score >= 0.6:
        bias = "bullish"
    elif score <= -0.6:
        bias = "bearish"
    else:
        bias = "neutral"

    pat_names = ", ".join(p.name for p in patterns) or "no major pattern"
    parts = [f"Bias: {bias.upper()} (score {score:+.2f}). Signals: {pat_names}."]

    if bias == "bullish":
        entry = sup[0] if sup else last_price
        target = res[0] if res else last_price * 1.08
        stop = sup[1] if len(sup) > 1 else last_price * 0.95
        parts.append(
            f"Method: momentum/breakout long. Consider entry near support ₹{entry:.1f} "
            f"or on breakout above ₹{(res[0] if res else last_price):.1f}; "
            f"target ₹{target:.1f}; stop-loss below ₹{stop:.1f}.")
        if rsi_v > 70:
            parts.append("Caution: RSI overbought — wait for a pullback before fresh entry.")
    elif bias == "bearish":
        entry = res[0] if res else last_price
        target = sup[0] if sup else last_price * 0.92
        stop = res[1] if len(res) > 1 else last_price * 1.05
        parts.append(
            f"Method: reversal/short or exit longs. Resistance ₹{entry:.1f}; "
            f"downside target ₹{target:.1f}; protective stop above ₹{stop:.1f}.")
        if rsi_v < 30:
            parts.append("Caution: RSI oversold — bounce possible before further downside.")
    else:
        parts.append(
            "Method: range/wait. No clear edge — trade the range "
            f"(buy near ₹{(sup[0] if sup else last_price*0.97):.1f}, "
            f"sell near ₹{(res[0] if res else last_price*1.03):.1f}) or stay flat until a pattern confirms.")

    return bias, " ".join(parts)


# =============================================================================
# Charting — interactive Plotly (price + SMAs + S/R, volume, RSI)
# =============================================================================
# Color palette (matches report theme)
C_UP, C_DOWN = "#26a69a", "#ef5350"
C_SMA = {"sma20": "#ffa726", "sma50": "#42a5f5", "sma200": "#ab47bc"}


def make_chart_html(df: pd.DataFrame, s: pd.DataFrame, symbol: str, sup, res) -> str:
    """Build an interactive 3-panel Plotly chart, return an HTML div string.

    The Plotly library itself is loaded once in the report <head> (see
    render_report), so every chart div is library-free (include_plotlyjs=False).
    This avoids per-chart bloat and guarantees Plotly is defined before any
    chart script runs, regardless of how the cards are sorted."""
    try:
        x = pd.to_datetime(df.index)
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.62, 0.18, 0.20], vertical_spacing=0.025,
            subplot_titles=("", "Volume", "RSI (14)"),
        )

        # --- Row 1: candlesticks + moving averages ---
        fig.add_trace(go.Candlestick(
            x=x, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="Price", increasing_line_color=C_UP, decreasing_line_color=C_DOWN,
            increasing_fillcolor=C_UP, decreasing_fillcolor=C_DOWN,
        ), row=1, col=1)
        for col, color in C_SMA.items():
            if s[col].notna().any():
                fig.add_trace(go.Scatter(
                    x=x, y=s[col], name=col.upper(), mode="lines",
                    line=dict(width=1.3, color=color),
                    hovertemplate=col.upper() + ": %{y:.1f}<extra></extra>",
                ), row=1, col=1)
        # Support (green) / resistance (red) levels
        for lvl in sup:
            fig.add_hline(y=lvl, line=dict(color=C_UP, width=1, dash="dot"),
                          annotation_text=f"S {lvl:g}", annotation_position="right",
                          annotation_font_size=9, row=1, col=1)
        for lvl in res:
            fig.add_hline(y=lvl, line=dict(color=C_DOWN, width=1, dash="dot"),
                          annotation_text=f"R {lvl:g}", annotation_position="right",
                          annotation_font_size=9, row=1, col=1)

        # --- Row 2: volume (colored by up/down day) ---
        vol_colors = [C_UP if c >= o else C_DOWN
                      for o, c in zip(df["Open"], df["Close"])]
        fig.add_trace(go.Bar(x=x, y=df["Volume"], name="Volume",
                             marker_color=vol_colors, showlegend=False,
                             hovertemplate="Vol: %{y:,.0f}<extra></extra>"),
                      row=2, col=1)

        # --- Row 3: RSI with 30/70 bands ---
        fig.add_trace(go.Scatter(x=x, y=s["rsi"], name="RSI", mode="lines",
                                 line=dict(color="#ffca28", width=1.3), showlegend=False,
                                 hovertemplate="RSI: %{y:.1f}<extra></extra>"),
                      row=3, col=1)
        fig.add_hline(y=70, line=dict(color=C_DOWN, width=0.8, dash="dash"), row=3, col=1)
        fig.add_hline(y=30, line=dict(color=C_UP, width=0.8, dash="dash"), row=3, col=1)
        fig.update_yaxes(range=[0, 100], row=3, col=1)

        fig.update_layout(
            template="plotly_dark",
            height=600, margin=dict(l=8, r=8, t=24, b=8),
            paper_bgcolor="#1a1d27", plot_bgcolor="#13161e",
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", y=1.05, x=0, font=dict(size=11)),
            hovermode="x unified", dragmode="zoom",
            font=dict(color="#cfd3dc", size=11),
        )
        # Hide weekend gaps on the daily x-axis
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

        return pio.to_html(
            fig, include_plotlyjs=False,
            full_html=False, default_height="600px",
            config={"responsive": True, "displaylogo": False,
                    "modeBarButtonsToRemove": ["select2d", "lasso2d"]},
        )
    except Exception as e:
        print(f"  [chart] {symbol}: {e}", file=sys.stderr)
        return f'<p style="color:#ef5350">chart unavailable: {e}</p>'


# =============================================================================
# Analysis driver
# =============================================================================
def analyze_symbol(symbol: str, period: str) -> StockResult:
    res = StockResult(symbol=symbol)
    try:
        df = fetch_history(symbol, period)
        s = compute_series(df)
        ind = indicator_snapshot(df, s)
        patterns, sup, resist = detect_patterns(df, s)
        close = df["Close"]
        res.last_price = float(close.iloc[-1])
        res.change_pct = float((close.iloc[-1] / close.iloc[0] - 1) * 100)
        res.indicators = ind
        res.patterns = patterns
        res.support, res.resistance = sup, resist
        bias, suggestion = build_suggestion(ind, patterns, sup, resist, res.last_price)
        res.overall_bias = bias
        res.suggestion = suggestion
        res.chart_html = make_chart_html(df, s, symbol, sup, resist)
    except Exception as e:
        res.ok = False
        res.error = str(e)
    return res


# =============================================================================
# HTML report
# =============================================================================
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stock Pattern Analysis — {{ generated }}</title>
{% if embed_js %}<script charset="utf-8">{{ plotly_js|safe }}</script>{% else %}<script src="https://cdn.plot.ly/plotly-3.6.0.min.js" charset="utf-8"></script>{% endif %}
<style>
  :root { --bull:#26a69a; --bear:#ef5350; --neu:#7a7a7a; --bg:#0d0f15; --card:#1a1d27;
          --line:#262a36; --txt:#e6e6e6; --muted:#8b92a1; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
         background:var(--bg); color:var(--txt); }
  header { padding:22px 32px; border-bottom:1px solid var(--line);
           background:linear-gradient(180deg,#161a24,#0d0f15); position:sticky; top:0; z-index:50; }
  header h1 { margin:0 0 3px; font-size:21px; }
  header p { margin:0; color:var(--muted); font-size:13px; }
  .controls { margin-top:10px; display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
  .controls input { background:#11141c; border:1px solid var(--line); color:var(--txt);
                    padding:6px 10px; border-radius:7px; font-size:13px; width:200px; }
  .fbtn { background:#11141c; border:1px solid var(--line); color:var(--muted);
          padding:6px 12px; border-radius:7px; font-size:12px; cursor:pointer; }
  .fbtn.active { color:#fff; border-color:#3a4150; }
  .fbtn.bull.active { background:var(--bull); border-color:var(--bull);}
  .fbtn.bear.active { background:var(--bear); border-color:var(--bear);}
  .fbtn.neu.active  { background:var(--neu); border-color:var(--neu);}
  .summary { padding:16px 32px; }
  table.idx { width:100%; border-collapse:collapse; font-size:13px; }
  table.idx th, table.idx td { padding:8px 10px; text-align:left; border-bottom:1px solid var(--line); }
  table.idx th { color:var(--muted); font-weight:600; cursor:pointer; user-select:none; position:sticky; top:0; }
  table.idx th:hover { color:#fff; }
  table.idx tbody tr:hover { background:#161922; }
  .pill { display:inline-block; padding:2px 10px; border-radius:11px; font-size:11px; font-weight:700; color:#0d0f15; }
  .bullish { background:var(--bull);} .bearish { background:var(--bear); color:#fff;} .neutral { background:var(--neu); color:#fff;}
  .cards { padding:8px 32px 48px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:14px;
          margin:20px 0; padding:18px 22px; scroll-margin-top:120px; }
  .card h2 { margin:0 0 10px; font-size:19px; display:flex; align-items:center; gap:10px; }
  .card h2 .chg { font-size:13px; font-weight:600; }
  .row { display:flex; flex-wrap:wrap; gap:24px; align-items:flex-start; }
  .col-chart { flex:2 1 600px; min-width:0; }
  .col-meta { flex:1 1 280px; }
  .kv { font-size:13px; line-height:2.0; }
  .kv b { color:var(--muted); font-weight:600; display:inline-block; min-width:104px; }
  .pat { font-size:12.5px; margin:5px 0; padding:7px 11px; border-radius:8px;
         background:#13161e; border-left:3px solid var(--neu); }
  .pat.bullish { border-left-color:var(--bull);} .pat.bearish { border-left-color:var(--bear);}
  .sugg { margin-top:14px; font-size:13.5px; line-height:1.65; background:#13161e;
          padding:13px 15px; border-radius:9px; border:1px solid var(--line); }
  .err { color:var(--bear); }
  .disclaimer { padding:0 32px 36px; color:#5f6573; font-size:12px; line-height:1.6; }
  a.anchor { color:inherit; text-decoration:none; }
  a.anchor:hover { color:#42a5f5; }
  .hidden { display:none !important; }
</style></head><body>
<header>
  <h1>📈 Indian Stock Chart Pattern Analysis</h1>
  <p>Generated {{ generated }} · Period {{ period }} · {{ ok_count }}/{{ total }} stocks analyzed
     · <b style="color:var(--bull)">{{ n_bull }} bullish</b>
     · <b style="color:var(--neu)">{{ n_neu }} neutral</b>
     · <b style="color:var(--bear)">{{ n_bear }} bearish</b></p>
  <div class="controls">
    <input id="search" type="text" placeholder="🔍 filter symbol…" oninput="applyFilter()">
    <button class="fbtn active" data-bias="all" onclick="setBias(this)">All</button>
    <button class="fbtn bull" data-bias="bullish" onclick="setBias(this)">Bullish</button>
    <button class="fbtn neu" data-bias="neutral" onclick="setBias(this)">Neutral</button>
    <button class="fbtn bear" data-bias="bearish" onclick="setBias(this)">Bearish</button>
  </div>
</header>

<div class="summary">
  <table class="idx" id="idx">
    <thead><tr>
      <th>#</th><th>Symbol</th><th>Last (₹)</th><th>Chg %</th><th>Bias</th>
      <th>RSI</th><th>Patterns</th>
    </tr></thead>
    <tbody>
    {% for r in results if r.ok %}
      <tr data-bias="{{ r.overall_bias }}" data-sym="{{ r.symbol|lower }}">
        <td>{{ loop.index }}</td>
        <td><a class="anchor" href="#{{ r.symbol }}"><b>{{ r.symbol }}</b></a></td>
        <td>{{ "%.2f"|format(r.last_price) }}</td>
        <td style="color:{{ 'var(--bull)' if r.change_pct>=0 else 'var(--bear)' }}">{{ "%+.1f"|format(r.change_pct) }}</td>
        <td><span class="pill {{ r.overall_bias }}">{{ r.overall_bias }}</span></td>
        <td>{{ "%.0f"|format(r.indicators.rsi) }}</td>
        <td>{{ r.patterns|map(attribute='name')|join(', ') if r.patterns else '—' }}</td>
      </tr>
    {% endfor %}
    {% for r in results if not r.ok %}
      <tr data-bias="error" data-sym="{{ r.symbol|lower }}">
        <td>—</td><td><b>{{ r.symbol }}</b></td>
        <td colspan="5" class="err">ERROR: {{ r.error }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
</div>

<div class="cards">
{% for r in results if r.ok %}
  <div class="card" id="{{ r.symbol }}" data-bias="{{ r.overall_bias }}" data-sym="{{ r.symbol|lower }}">
    <h2>{{ r.symbol }}
      <span class="pill {{ r.overall_bias }}">{{ r.overall_bias }}</span>
      <span class="chg" style="color:{{ 'var(--bull)' if r.change_pct>=0 else 'var(--bear)' }}">
        ₹{{ "%.2f"|format(r.last_price) }} ({{ "%+.1f"|format(r.change_pct) }}%)</span>
    </h2>
    <div class="row">
      <div class="col-chart">{{ r.chart_html|safe }}</div>
      <div class="col-meta">
        <div class="kv">
          <div><b>RSI(14)</b> {{ "%.1f"|format(r.indicators.rsi) }}</div>
          <div><b>MACD hist</b> {{ "%+.2f"|format(r.indicators.macd_hist) }}</div>
          <div><b>SMA 50/200</b> {{ "%.1f"|format(r.indicators.sma50) }} / {{ "%.1f"|format(r.indicators.sma200) }}</div>
          <div><b>Vol vs avg</b> {{ "%.2f"|format(r.indicators.vol_ratio) }}×</div>
          <div><b>Support</b> {{ r.support|join(', ') if r.support else '—' }}</div>
          <div><b>Resistance</b> {{ r.resistance|join(', ') if r.resistance else '—' }}</div>
        </div>
        <div style="margin-top:10px">
        {% for p in r.patterns %}
          <div class="pat {{ p.bias }}"><b>{{ p.name }}</b> · {{ "%.0f"|format(p.confidence*100) }}% — {{ p.note }}</div>
        {% else %}<div class="pat">No major chart pattern detected.</div>{% endfor %}
        </div>
      </div>
    </div>
    <div class="sugg">💡 {{ r.suggestion }}</div>
  </div>
{% endfor %}
</div>

<p class="disclaimer">
  ⚠️ <b>Disclaimer:</b> This report is generated by heuristic pattern-detection algorithms for
  educational/research purposes only. It is <b>not</b> investment advice. Pattern signals can be
  false; data may be delayed or inaccurate. Always do your own research and consult a SEBI-registered
  advisor before trading. Data source: Yahoo Finance via yfinance.
  Charts are interactive — drag to zoom, double-click to reset, hover for values, click legend items to toggle.
</p>

<script>
// --- Index table sorting (click header) ---
document.querySelectorAll('#idx th').forEach((th, col) => th.addEventListener('click', () => {
  const tb = document.querySelector('#idx tbody');
  const rows = [...tb.rows].filter(r => r.dataset.bias !== 'error');
  const num = col >= 2 && col <= 5;
  rows.sort((a,b) => {
    const x=a.cells[col]?.innerText||'', y=b.cells[col]?.innerText||'';
    return num ? (parseFloat(x)||0)-(parseFloat(y)||0) : x.localeCompare(y);
  });
  if (th.dataset.asc==='1'){ rows.reverse(); th.dataset.asc='0'; } else th.dataset.asc='1';
  rows.forEach(r=>tb.appendChild(r));
}));

// --- Bias + search filtering (applies to both table rows and cards) ---
let curBias = 'all';
function setBias(btn){
  curBias = btn.dataset.bias;
  document.querySelectorAll('.fbtn').forEach(b=>b.classList.toggle('active', b===btn));
  applyFilter();
}
function applyFilter(){
  const q = (document.getElementById('search').value||'').toLowerCase();
  const match = el => {
    const b = el.dataset.bias, s = el.dataset.sym||'';
    const okBias = curBias==='all' || b===curBias;
    const okText = !q || s.includes(q);
    return okBias && okText;
  };
  document.querySelectorAll('#idx tbody tr').forEach(tr=>{
    if (tr.dataset.bias==='error'){ tr.classList.toggle('hidden', !!q && !(tr.dataset.sym||'').includes(q)); return; }
    tr.classList.toggle('hidden', !match(tr));
  });
  document.querySelectorAll('.card').forEach(c=> c.classList.toggle('hidden', !match(c)));
}
</script>
</body></html>"""


def render_report(results, period, out_path, embed_js):
    html = render_report_html(results, period, embed_js)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def render_report_html(results, period, embed_js) -> str:
    """Render the full report HTML to a string (shared by CLI + web front-end).

    Sorts results (bullish→neutral→bearish, errors last) and, when embed_js is
    True, inlines the ~4.8MB Plotly library into <head> so the report is fully
    self-contained; otherwise it loads from the CDN."""
    order = {"bullish": 0, "neutral": 1, "bearish": 2}
    results_sorted = sorted(
        results,
        key=lambda r: (0 if r.ok else 1, order.get(r.overall_bias, 1), -r.change_pct if r.ok else 0),
    )
    ok = [r for r in results_sorted if r.ok]
    # Pull the library source only when embedding (avoids the cost in CDN mode).
    plotly_js = ""
    if embed_js:
        from plotly.offline import get_plotlyjs
        plotly_js = get_plotlyjs()
    return Template(HTML_TEMPLATE).render(
        results=results_sorted,
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        period=period,
        ok_count=len(ok),
        total=len(results_sorted),
        n_bull=sum(1 for r in ok if r.overall_bias == "bullish"),
        n_neu=sum(1 for r in ok if r.overall_bias == "neutral"),
        n_bear=sum(1 for r in ok if r.overall_bias == "bearish"),
        embed_js=embed_js,
        plotly_js=plotly_js,
    )


# =============================================================================
# CLI
# =============================================================================
def load_symbols(path):
    syms = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                syms.append(line)
    return syms


def main():
    ap = argparse.ArgumentParser(description="Indian stock chart pattern analyzer → interactive HTML report")
    ap.add_argument("--stocks", default="stocks.txt", help="file with one ticker per line")
    ap.add_argument("--symbols", nargs="*", help="tickers directly (overrides --stocks)")
    ap.add_argument("--period", default="1mo",
                    choices=["1mo", "3mo", "6mo", "1y", "2y"], help="history window")
    ap.add_argument("--out", default="report.html", help="output HTML path")
    ap.add_argument("--cdn-charts", action="store_true",
                    help="load Plotly.js from a CDN instead of embedding it. "
                         "Smaller file, but charts need internet AND a real browser "
                         "(IDE/preview panes block external scripts → blank charts).")
    args = ap.parse_args()
    # Default: embed Plotly.js so the report is self-contained and renders anywhere.
    embed_js = not args.cdn_charts

    try:
        symbols = args.symbols if args.symbols else load_symbols(args.stocks)
    except FileNotFoundError:
        sys.exit(f"Stock list not found: {args.stocks}")
    if not symbols:
        sys.exit("No symbols to analyze.")

    # Plotly.js is loaded once in the report <head> (inline when embedding, else
    # from the CDN); individual charts never carry the library — see render_report.
    print(f"Analyzing {len(symbols)} symbols (period={args.period})...")
    results = []
    for i, sym in enumerate(symbols, 1):
        print(f"  [{i}/{len(symbols)}] {sym} ...", end=" ", flush=True)
        r = analyze_symbol(sym, args.period)
        print("OK" if r.ok else f"FAIL ({r.error})")
        results.append(r)

    render_report(results, args.period, args.out, embed_js=embed_js)
    ok = sum(1 for r in results if r.ok)
    print(f"\nDone. {ok}/{len(results)} analyzed → {args.out}")


if __name__ == "__main__":
    main()
