# Gold Desk (cloud)

A phone dashboard for GoldBot-style gold signals that doesn't need the laptop.
GitHub Actions runs `engine.py` every hour (Sun–Fri), pulls hourly gold
candles from Yahoo Finance (`GC=F`), computes the signal and paper trades,
and writes `docs/data.json`. GitHub Pages serves `docs/index.html`.

## What it computes
- EMA50 / EMA200 trend, Supertrend (10, 3), MACD histogram, RSI(14)
- GJR-GARCH(1,1) volatility (t-dist), refit every 24 bars on 1000 bars
- Volatility band filter (10th–95th percentile) and blocked hours (21–23 UTC)
- Entry on a new setup at the next bar's open; SL = 2σ, TP = 4σ (GARCH, in price)
- 1% risk per trade on a $10,000 paper balance, $0.30 spread cost

All settings are in `config.json`. Edit it on GitHub (also from the phone),
and the next run uses the new values.

## Differences from the laptop GoldBot
- Data is COMEX gold futures (GC=F), not Exness XAUUSDm spot, so prices differ by a few dollars.
- No live spread, volume or news filters (no broker feed in the cloud).
- Signal rules are a re-implementation; port the exact rules from `bot/` to match.
- Paper only, so no orders are sent anywhere.

## Run it manually
GitHub → **Actions** → *Update Gold Desk* → **Run workflow**.

## Telegram alerts (optional)
Add repo secrets `TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID`
(Settings → Secrets and variables → Actions). You'll get a message when a
paper trade opens or closes.

## Local test
```
pip install -r requirements.txt
python engine.py --sample   # offline synthetic data
python engine.py            # live data
cd docs && python -m http.server 8000
```
