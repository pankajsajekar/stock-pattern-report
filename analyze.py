#!/usr/bin/env python3
"""
Indian Stock Chart Pattern Analyzer
-----------------------------------
Fetches NSE/BSE historical data, computes technical indicators, detects classic
chart patterns, and writes a single self-contained HTML report with candlestick
charts and per-stock trading-method suggestions.

Usage:
    python analyze.py                         # uses stocks.txt, 6mo period
    python analyze.py --stocks stocks.txt --period 6mo --out report.html
    python analyze.py --symbols RELIANCE.NS TCS.NS

Patterns detected (heuristic, see notes in report):
    Double Top / Double Bottom, Head & Shoulders (+ Inverse), Bull/Bear Flag,
    Cup & Handle, Golden/Death Cross, Support & Resistance levels.

NOTE: This is a technical-analysis tool for research/education. It is NOT
investment advice. Pattern detection is heuristic and can produce false signals.
"""

import argparse
import base64
import io
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")  # silence yfinance / mpl noise

# ---- Heavy / optional imports guarded so failures give a clear message --------
try:
    import yfinance as yf
    import mplfinance as mpf
    import matplotlib
    matplotlib.use("Agg")  # headless backend, no display needed
    from scipy.signal import find_peaks
    from jinja2 import Template
except ImportError as e:  # pragma: no cover
    sys.exit(
        f"Missing dependency: {e.name}. "
        "Install with: pip install yfinance pandas numpy scipy mplfinance jinja2"
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
    chart_b64: str = ""


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
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def bollinger(close: pd.Series, period=20, std=2):
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std()
    return mid + std * sd, mid, mid - std * sd


def compute_indicators(df: pd.DataFrame) -> dict:
    close = df["Close"]
    macd_line, macd_sig, macd_hist = macd(close)
    bb_up, bb_mid, bb_low = bollinger(close)
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    vol_avg = df["Volume"].rolling(20).mean()

    def last(s):
        v = s.dropna()
        return float(v.iloc[-1]) if len(v) else float("nan")

    return {
        "rsi": last(rsi(close)),
        "macd": last(macd_line),
        "macd_signal": last(macd_sig),
        "macd_hist": last(macd_hist),
        "bb_upper": last(bb_up),
        "bb_lower": last(bb_low),
        "sma20": last(sma20),
        "sma50": last(sma50),
        "sma200": last(sma200),
        "vol_ratio": (last(df["Volume"]) / last(vol_avg)) if last(vol_avg) else float("nan"),
        # series kept for cross detection
        "_sma50_series": sma50,
        "_sma200_series": sma200,
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
    # Double Top: two recent peaks of similar height with a trough between
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
    # Double Bottom: two recent troughs of similar depth with a peak between
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
    # Head & Shoulders: 3 peaks, middle highest, shoulders ~equal
    if len(peaks) >= 3:
        l, h, r = peaks[-3], peaks[-2], peaks[-1]
        hl, hh, hr = close[l], close[h], close[r]
        if hh > hl and hh > hr and abs(hl - hr) / max(hl, hr) < 0.05:
            hits.append(PatternHit(
                "Head & Shoulders", "bearish", 0.7,
                f"Head {hh:.1f} above shoulders {hl:.1f}/{hr:.1f} — reversal top"))
    # Inverse H&S: 3 troughs, middle lowest, shoulders ~equal
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
    if flag_range < 0.06:  # tight consolidation
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
    # bottom roughly centered, rims near equal, handle is a shallow dip below right rim
    centered = 0.25 * len(cup) < bottom_idx < 0.75 * len(cup)
    rims_equal = abs(left - right) / max(left, right) < 0.07
    depth = (max(left, right) - bottom) / max(left, right)
    handle_dip = (right - handle.min()) / right
    if centered and rims_equal and 0.1 < depth < 0.5 and 0 < handle_dip < 0.12:
        hits.append(PatternHit("Cup & Handle", "bullish", 0.6,
                               f"U-base depth {depth*100:.0f}% + shallow handle — bullish continuation"))
    return hits


def detect_crosses(ind: dict):
    """Golden/Death cross from SMA50 vs SMA200 over the last ~10 sessions."""
    hits = []
    s50 = ind.get("_sma50_series")
    s200 = ind.get("_sma200_series")
    if s50 is None or s200 is None:
        return hits
    df = pd.concat([s50, s200], axis=1).dropna()
    if len(df) < 11:
        return hits
    a = df.iloc[:, 0].values  # sma50
    b = df.iloc[:, 1].values  # sma200
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
    # resistance = levels above price; support = levels below
    res = sorted([r for r in res if r >= last])[:n_levels]
    sup = sorted([s for s in sup if s <= last], reverse=True)[:n_levels]
    return [round(float(s), 2) for s in sup], [round(float(r), 2) for r in res]


def detect_patterns(df: pd.DataFrame, ind: dict):
    close = df["Close"].values.astype(float)
    peaks, troughs = _peaks_troughs(close)
    hits = []
    hits += detect_double_top_bottom(close, peaks, troughs)
    hits += detect_head_shoulders(close, peaks, troughs)
    hits += detect_flag(close)
    hits += detect_cup_handle(close)
    hits += detect_crosses(ind)
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

    # Trading method text
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
# Charting
# =============================================================================
def make_chart_b64(df: pd.DataFrame, symbol: str, sup, res) -> str:
    """Render a candlestick chart with MAs + volume, return base64 PNG."""
    try:
        plot_df = df.copy()
        plot_df.index = pd.to_datetime(plot_df.index)
        addplots = []
        hlines = dict(hlines=[*sup, *res],
                      colors=["g"] * len(sup) + ["r"] * len(res),
                      linestyle="--", linewidths=0.7) if (sup or res) else None
        mav = (20, 50)
        buf = io.BytesIO()
        kwargs = dict(
            type="candle", style="yahoo", mav=mav, volume=True,
            title=f"\n{symbol}", figsize=(11, 6), tight_layout=True,
            savefig=dict(fname=buf, dpi=90, bbox_inches="tight"),
        )
        if hlines:
            kwargs["hlines"] = hlines
        mpf.plot(plot_df, **kwargs)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")
    except Exception as e:
        print(f"  [chart] {symbol}: {e}", file=sys.stderr)
        return ""


# =============================================================================
# Analysis driver
# =============================================================================
def analyze_symbol(symbol: str, period: str) -> StockResult:
    res = StockResult(symbol=symbol)
    try:
        df = fetch_history(symbol, period)
        ind = compute_indicators(df)
        patterns, sup, resist = detect_patterns(df, ind)
        close = df["Close"]
        res.last_price = float(close.iloc[-1])
        res.change_pct = float((close.iloc[-1] / close.iloc[0] - 1) * 100)
        res.indicators = {k: v for k, v in ind.items() if not k.startswith("_")}
        res.patterns = patterns
        res.support, res.resistance = sup, resist
        bias, suggestion = build_suggestion(ind, patterns, sup, resist, res.last_price)
        res.overall_bias = bias
        res.suggestion = suggestion
        res.chart_b64 = make_chart_b64(df, symbol, sup, resist)
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
<style>
  :root { --bull:#0a8f4e; --bear:#d62828; --neu:#7a7a7a; --bg:#0f1117; --card:#1a1d27; --txt:#e6e6e6; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
         background:var(--bg); color:var(--txt); }
  header { padding:24px 32px; border-bottom:1px solid #2a2e3a; }
  header h1 { margin:0 0 4px; font-size:22px; }
  header p { margin:0; color:#9aa0ac; font-size:13px; }
  .summary { padding:16px 32px; }
  table.idx { width:100%; border-collapse:collapse; font-size:13px; }
  table.idx th, table.idx td { padding:7px 10px; text-align:left; border-bottom:1px solid #262a36; }
  table.idx th { color:#9aa0ac; font-weight:600; cursor:pointer; }
  .pill { display:inline-block; padding:2px 9px; border-radius:11px; font-size:11px; font-weight:700; color:#fff; }
  .bullish { background:var(--bull);} .bearish { background:var(--bear);} .neutral { background:var(--neu);}
  .cards { padding:8px 32px 48px; }
  .card { background:var(--card); border:1px solid #262a36; border-radius:12px; margin:18px 0; padding:18px 22px; }
  .card h2 { margin:0 0 2px; font-size:18px; }
  .row { display:flex; flex-wrap:wrap; gap:24px; align-items:flex-start; }
  .col-chart { flex:2 1 540px; }
  .col-meta { flex:1 1 300px; }
  .chart img { width:100%; border-radius:8px; background:#fff; }
  .kv { font-size:13px; line-height:1.9; }
  .kv b { color:#9aa0ac; font-weight:600; display:inline-block; min-width:96px; }
  .pat { font-size:13px; margin:4px 0; padding:6px 10px; border-radius:7px; background:#22273300; border-left:3px solid var(--neu);}
  .pat.bullish { border-left-color:var(--bull);} .pat.bearish { border-left-color:var(--bear);}
  .sugg { margin-top:12px; font-size:13.5px; line-height:1.6; background:#161922; padding:12px 14px; border-radius:8px; border:1px solid #262a36;}
  .err { color:var(--bear); }
  .disclaimer { padding:0 32px 32px; color:#6b7180; font-size:12px; }
  a.anchor { color:inherit; text-decoration:none; }
  code { color:#9ad; }
</style></head><body>
<header>
  <h1>📈 Indian Stock Chart Pattern Analysis</h1>
  <p>Generated {{ generated }} · Period {{ period }} · {{ ok_count }}/{{ total }} stocks analyzed</p>
</header>

<div class="summary">
  <table class="idx" id="idx">
    <thead><tr>
      <th>#</th><th>Symbol</th><th>Last (₹)</th><th>Chg %</th><th>Bias</th>
      <th>RSI</th><th>Patterns</th>
    </tr></thead>
    <tbody>
    {% for r in results %}
      <tr>
        <td>{{ loop.index }}</td>
        <td><a class="anchor" href="#{{ r.symbol }}"><b>{{ r.symbol }}</b></a></td>
        {% if r.ok %}
        <td>{{ "%.2f"|format(r.last_price) }}</td>
        <td style="color:{{ 'var(--bull)' if r.change_pct>=0 else 'var(--bear)' }}">{{ "%+.1f"|format(r.change_pct) }}</td>
        <td><span class="pill {{ r.overall_bias }}">{{ r.overall_bias }}</span></td>
        <td>{{ "%.0f"|format(r.indicators.rsi) }}</td>
        <td>{{ r.patterns|map(attribute='name')|join(', ') if r.patterns else '—' }}</td>
        {% else %}
        <td colspan="5" class="err">ERROR: {{ r.error }}</td>
        {% endif %}
      </tr>
    {% endfor %}
    </tbody>
  </table>
</div>

<div class="cards">
{% for r in results %}{% if r.ok %}
  <div class="card" id="{{ r.symbol }}">
    <h2>{{ r.symbol }} <span class="pill {{ r.overall_bias }}">{{ r.overall_bias }}</span></h2>
    <div class="row">
      <div class="col-chart chart">
        {% if r.chart_b64 %}<img src="data:image/png;base64,{{ r.chart_b64 }}" alt="{{ r.symbol }} chart">
        {% else %}<p class="err">chart unavailable</p>{% endif %}
      </div>
      <div class="col-meta">
        <div class="kv">
          <div><b>Last price</b> ₹{{ "%.2f"|format(r.last_price) }} ({{ "%+.1f"|format(r.change_pct) }}% over period)</div>
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
{% endif %}{% endfor %}
</div>

<p class="disclaimer">
  ⚠️ <b>Disclaimer:</b> This report is generated by heuristic pattern-detection algorithms for
  educational/research purposes only. It is <b>not</b> investment advice. Pattern signals can be
  false; data may be delayed or inaccurate. Always do your own research and consult a SEBI-registered
  advisor before trading. Data source: Yahoo Finance via yfinance.
</p>

<script>
// Click a column header to sort the index table.
document.querySelectorAll('#idx th').forEach((th, col) => th.addEventListener('click', () => {
  const tb = document.querySelector('#idx tbody');
  const rows = [...tb.rows];
  const num = col >= 2 && col <= 5;
  rows.sort((a,b) => {
    const x=a.cells[col]?.innerText||'', y=b.cells[col]?.innerText||'';
    return num ? (parseFloat(x)||0)-(parseFloat(y)||0) : x.localeCompare(y);
  });
  if (th.dataset.asc==='1'){ rows.reverse(); th.dataset.asc='0'; } else th.dataset.asc='1';
  rows.forEach(r=>tb.appendChild(r));
}));
</script>
</body></html>"""


def render_report(results, period, out_path):
    ok = [r for r in results if r.ok]
    # Sort report: strongest bullish first, then neutral, then bearish; errors last
    order = {"bullish": 0, "neutral": 1, "bearish": 2}
    results_sorted = sorted(
        results,
        key=lambda r: (0 if r.ok else 1, order.get(r.overall_bias, 1), -r.change_pct if r.ok else 0),
    )
    html = Template(HTML_TEMPLATE).render(
        results=results_sorted,
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        period=period,
        ok_count=len(ok),
        total=len(results),
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


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
    ap = argparse.ArgumentParser(description="Indian stock chart pattern analyzer → HTML report")
    ap.add_argument("--stocks", default="stocks.txt", help="file with one ticker per line")
    ap.add_argument("--symbols", nargs="*", help="tickers directly (overrides --stocks)")
    ap.add_argument("--period", default="6mo",
                    choices=["1mo", "3mo", "6mo", "1y", "2y"], help="history window")
    ap.add_argument("--out", default="report.html", help="output HTML path")
    args = ap.parse_args()

    try:
        symbols = args.symbols if args.symbols else load_symbols(args.stocks)
    except FileNotFoundError:
        sys.exit(f"Stock list not found: {args.stocks}")
    if not symbols:
        sys.exit("No symbols to analyze.")

    print(f"Analyzing {len(symbols)} symbols (period={args.period})...")
    results = []
    for i, sym in enumerate(symbols, 1):
        print(f"  [{i}/{len(symbols)}] {sym} ...", end=" ", flush=True)
        r = analyze_symbol(sym, args.period)
        print("OK" if r.ok else f"FAIL ({r.error})")
        results.append(r)

    render_report(results, args.period, args.out)
    ok = sum(1 for r in results if r.ok)
    print(f"\nDone. {ok}/{len(results)} analyzed → {args.out}")


if __name__ == "__main__":
    main()
