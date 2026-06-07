#!/usr/bin/env python3
"""
Web front-end for the Indian Stock Chart Pattern Analyzer.

A tiny Flask app that wraps analyze.py: shows a form, lets a visitor enter
symbols + a period, runs the SAME analysis pipeline on demand, and returns the
interactive HTML report.

Local run:   python app.py            (http://localhost:8000)
Production:  gunicorn app:app         (Render/HF use this via Procfile)
"""

import os

from flask import Flask, request, Response, render_template_string

# Reuse the analysis pipeline verbatim — no logic duplicated here.
import analyze

app = Flask(__name__)

# Hard cap so a single request can't run forever on a small free instance
# (each symbol = a network fetch + indicators + chart build).
MAX_SYMBOLS = 12

# Embed Plotly.js inline in the report (loaded once in <head>, not per chart).
# Self-contained = charts always render regardless of CDN availability/version.
# Adds ~4.8MB to each report response, which is fine for occasional on-demand use.
# Set to False to load Plotly from the CDN instead (smaller pages, needs the CDN).
EMBED_JS = True

FORM_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stock Pattern Analyzer</title>
<style>
  body { margin:0; font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;
         background:#0d0f15; color:#e6e6e6; display:flex; min-height:100vh;
         align-items:center; justify-content:center; }
  .box { background:#1a1d27; border:1px solid #262a36; border-radius:16px;
         padding:32px 36px; width:min(520px,92vw); }
  h1 { margin:0 0 4px; font-size:22px; }
  p.sub { margin:0 0 22px; color:#8b92a1; font-size:13px; line-height:1.5; }
  label { display:block; font-size:13px; color:#8b92a1; margin:16px 0 6px; }
  textarea, select { width:100%; background:#11141c; border:1px solid #262a36;
         color:#e6e6e6; border-radius:9px; padding:10px 12px; font-size:14px;
         font-family:inherit; box-sizing:border-box; }
  textarea { height:96px; resize:vertical; }
  button { margin-top:22px; width:100%; padding:12px; border:0; border-radius:9px;
           background:#26a69a; color:#0d0f15; font-size:15px; font-weight:700;
           cursor:pointer; }
  button:hover { background:#2bbbad; }
  .note { margin-top:16px; font-size:11.5px; color:#5f6573; line-height:1.6; }
</style></head><body>
  <div class="box">
    <h1>📈 Stock Pattern Analyzer</h1>
    <p class="sub">Enter NSE/BSE tickers (e.g. <code>RELIANCE.NS</code>,
       <code>TCS.NS</code>). Up to {{ max_symbols }} at a time. The analysis runs
       live and may take ~10–40s.</p>
    <form action="/run" method="get">
      <label>Symbols (comma or newline separated)</label>
      <textarea name="symbols" placeholder="RELIANCE.NS, TCS.NS, INFY.NS">{{ default_symbols }}</textarea>
      <label>History period</label>
      <select name="period">
        <option value="1mo">1 month</option>
        <option value="3mo">3 months</option>
        <option value="6mo" selected>6 months</option>
        <option value="1y">1 year</option>
        <option value="2y">2 years</option>
      </select>
      <button type="submit">Run analysis →</button>
    </form>
    <p class="note">⚠️ Educational/research tool — not investment advice. Data via
       Yahoo Finance may be delayed, and cloud hosts can occasionally be
       rate-limited (some symbols may return errors).</p>
  </div>
</body></html>"""


def _parse_symbols(raw: str):
    """Split the textarea on commas/newlines/spaces, dedupe, cap, uppercase."""
    parts = raw.replace("\n", ",").replace(" ", ",").split(",")
    seen, out = set(), []
    for p in parts:
        sym = p.strip().upper()
        if sym and sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out[:MAX_SYMBOLS]


@app.route("/")
def index():
    return render_template_string(
        FORM_PAGE, max_symbols=MAX_SYMBOLS,
        default_symbols="RELIANCE.NS, TCS.NS, INFY.NS",
    )


@app.route("/run")
def run():
    symbols = _parse_symbols(request.args.get("symbols", ""))
    period = request.args.get("period", "6mo")
    if period not in {"1mo", "3mo", "6mo", "1y", "2y"}:
        period = "6mo"
    if not symbols:
        return Response("<p style='color:#ef5350;font-family:sans-serif'>"
                        "No symbols provided. <a href='/'>Go back</a>.</p>",
                        mimetype="text/html", status=400)

    results = []
    for sym in symbols:
        try:
            results.append(analyze.analyze_symbol(sym, period))
        except Exception as e:  # never let one bad symbol 500 the whole page
            r = analyze.StockResult(symbol=sym, ok=False, error=str(e))
            results.append(r)

    try:
        html = analyze.render_report_html(results, period, embed_js=EMBED_JS)
    except Exception as e:
        return Response(f"<p style='color:#ef5350'>Report render failed: {e}</p>",
                        mimetype="text/html", status=500)
    return Response(html, mimetype="text/html")


@app.route("/healthz")
def healthz():
    return "ok"


if __name__ == "__main__":
    # Local dev server. In production, gunicorn (see Procfile) serves app:app.
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
