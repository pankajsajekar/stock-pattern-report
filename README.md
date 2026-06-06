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

The report's index table is sortable (click a header) and each symbol links to
its detail card.

## How it works (data flow)

```
stocks.txt --> yfinance (OHLCV) --> indicators + pattern detection
           --> mplfinance candlestick (base64 PNG) --> Jinja2 HTML report
```

## Important notes

- **Not investment advice.** Pattern detection is *heuristic* and produces false
  signals; data from Yahoo Finance may be delayed or inaccurate. Research/education only.
- Failed tickers (renamed/delisted) are reported per-row and never abort the run.
- Patterns are detected on the close-price series with prominence-filtered
  peaks/troughs (SciPy `find_peaks`); thresholds are tunable in `analyze.py`.
