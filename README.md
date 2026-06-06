# Indian Stock Chart Pattern Analyzer

Fetches NSE/BSE historical data, computes technical indicators, detects classic
chart patterns, and produces a single self-contained **HTML report** with
candlestick charts and per-stock trading-method suggestions.

## Quick start

```bash
./run.sh                      # analyze stocks.txt, 6-month window -> report.html
```

First run auto-creates a virtualenv and installs dependencies. Open `report.html`
in any browser (it's fully self-contained — charts are embedded, no server needed).

Manual setup (e.g. after `git clone`):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python analyze.py
```

## Usage

```bash
./run.sh --period 1y                       # different window
./run.sh --symbols RELIANCE.NS TCS.NS      # ad-hoc symbols, ignore stocks.txt
./run.sh --stocks mylist.txt --out out.html
```

Or call Python directly after setup: `.venv/bin/python analyze.py --help`

| Flag        | Default      | Options                              |
|-------------|--------------|--------------------------------------|
| `--stocks`  | `stocks.txt` | path to ticker list                  |
| `--symbols` | —            | tickers inline (overrides `--stocks`)|
| `--period`  | `6mo`        | `1mo 3mo 6mo 1y 2y`                  |
| `--out`     | `report.html`| output path                          |
| `--cdn-charts` | off      | load Plotly.js from a CDN instead of embedding (smaller file) |

### Self-contained (default) vs CDN charts

By default the report **embeds Plotly.js**, so it's fully self-contained
(~6.5 MB for 50 stocks) and renders **anywhere** — IDE preview panes, a browser,
or with no internet at all. This is the safe default.

Pass `--cdn-charts` for a much smaller file that loads Plotly from a CDN —
but then charts **only render in a real browser with internet**. IDE/preview
panes block external scripts, so a CDN report shows **blank charts** there.

## The stock list (`stocks.txt`)

One ticker per line. `#` lines and blanks are ignored. Use `.NS` for NSE,
`.BO` for BSE. Edit freely. Ships with the Nifty 50.

> Tata Motors demerged (2025) → `TMPV.NS` (passenger) + `TMCV.NS` (commercial).
> `LTIM.NS` had no Yahoo data at build time — re-check its ticker if you need it.

## What it computes

- **Indicators:** RSI(14), MACD, Bollinger Bands, SMA 20/50/200, volume vs 20-day avg.
- **Patterns:** Double Top / Double Bottom, Head & Shoulders (+ Inverse),
  Bull / Bear Flag, Cup & Handle, Golden / Death Cross, Support & Resistance.
- **Per stock:** a bias (bullish / bearish / neutral) with a plain-language
  trading method — entry, target, stop-loss — derived from patterns + indicators.

### The report

- **Interactive charts** (Plotly): 3 panels per stock — candlesticks with
  SMA 20/50/200 overlays and support/resistance lines, a volume sub-panel, and
  an RSI sub-panel with 30/70 bands. Drag to zoom, double-click to reset, hover
  for values, click legend items to toggle series.
- **Sortable index table** — click any header to sort.
- **Live filters** — search box + Bullish/Neutral/Bearish buttons filter both
  the table and the detail cards instantly.

## How it works (data flow)

```
stocks.txt --> yfinance (OHLCV) --> indicators + pattern detection
           --> Plotly interactive chart --> Jinja2 HTML report
```

## Important notes

- **Not investment advice.** Pattern detection is *heuristic* and produces false
  signals; data from Yahoo Finance may be delayed or inaccurate. Research/education only.
- Failed tickers (renamed/delisted) are reported per-row and never abort the run.
- Patterns are detected on the close-price series with prominence-filtered
  peaks/troughs (SciPy `find_peaks`); thresholds are tunable in `analyze.py`.
