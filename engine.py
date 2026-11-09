"""Gold Desk cloud engine.

Fetches hourly gold candles, runs the GoldBot-style signal
(EMA50/200 + Supertrend + MACD + RSI, GJR-GARCH volatility filter and
SL/TP), simulates paper trades, and writes docs/data.json for the
phone dashboard. Runs hourly on GitHub Actions; laptop not needed.

Usage:
    python engine.py            # live data from Yahoo Finance
    python engine.py --sample   # synthetic data (offline test)
"""
import argparse
import json
import math
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "docs", "data.json")


def load_config():
    with open(os.path.join(HERE, "config.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------- data
def fetch_yahoo(symbol, period, interval="1h"):
    import yfinance as yf

    df = yf.download(symbol, period=period, interval=interval,
                     auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.dropna()
    return df[~df.index.duplicated(keep="last")]


def sample_data(n=8000, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("h"),
                        periods=n, freq="h")
    vol = np.empty(n)
    v = 0.002
    r = np.empty(n)
    for i in range(n):
        v = math.sqrt(1e-7 + 0.05 * (r[i - 1] if i else 0) ** 2 + 0.93 * v * v)
        r[i] = rng.normal(0.00003, v)
        vol[i] = v
    close = 1900 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]]
    spread = close * vol * rng.uniform(0.5, 1.5, n)
    high = np.maximum(open_, close) + spread / 2
    low = np.minimum(open_, close) - spread / 2
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close,
                         "volume": rng.integers(500, 5000, n)}, index=idx)


# ---------------------------------------------------------- indicators
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df, n):
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def supertrend(df, n=10, mult=3.0):
    a = atr(df, n).values
    hl2 = ((df["high"] + df["low"]) / 2).values
    c = df["close"].values
    ub, lb = hl2 + mult * a, hl2 - mult * a
    fu, fl = ub.copy(), lb.copy()
    d = np.ones(len(c), dtype=int)
    line = np.full(len(c), np.nan)
    for i in range(1, len(c)):
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lb[i] if (lb[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        if d[i - 1] == 1:
            d[i] = -1 if c[i] < fl[i] else 1
        else:
            d[i] = 1 if c[i] > fu[i] else -1
        line[i] = fl[i] if d[i] == 1 else fu[i]
    return pd.Series(d, df.index), pd.Series(line, df.index)


def gjr_garch_sigma(ret_pct, window, refit_every):
    """One-step-ahead conditional sigma (in % per bar) with GJR-GARCH(1,1).

    Parameters are refit every `refit_every` bars on a trailing window;
    between refits the variance recursion is propagated with the last
    fitted parameters, so each value only uses past data.
    """
    from arch import arch_model

    r = ret_pct.values
    n = len(r)
    sig = np.full(n, np.nan)
    params = None
    s2 = np.nanvar(r[:window]) if n > window else np.nanvar(r)
    for t in range(window, n):
        if params is None or (t - window) % refit_every == 0:
            try:
                res = arch_model(r[t - window:t], mean="Constant", vol="GARCH",
                                 p=1, o=1, q=1, dist="t", rescale=False
                                 ).fit(disp="off", show_warning=False)
                p = res.params
                params = (p["mu"], p["omega"], p["alpha[1]"],
                          p["gamma[1]"], p["beta[1]"])
                s2 = float(res.conditional_volatility[-1] ** 2)
            except Exception:
                if params is None:
                    continue
        mu, w, a, g, b = params
        e = r[t - 1] - mu
        s2 = w + (a + g * (e < 0)) * e * e + b * s2
        sig[t] = math.sqrt(max(s2, 1e-12))
    return pd.Series(sig, ret_pct.index)


def build(df, cfg):
    s = cfg["strategy"]
    df = df.copy()
    df["ema_fast"] = ema(df["close"], s["ema_fast"])
    df["ema_slow"] = ema(df["close"], s["ema_slow"])
    m = ema(df["close"], 12) - ema(df["close"], 26)
    df["macd_hist"] = m - ema(m, 9)
    df["rsi"] = rsi(df["close"], 14)
    df["st_dir"], df["st_line"] = supertrend(df, s["st_period"], s["st_mult"])
    ret = 100 * np.log(df["close"]).diff().fillna(0)
    df["sigma_pct"] = gjr_garch_sigma(ret, s["garch_window"], s["refit_every"])
    q = df["sigma_pct"].rolling(500, min_periods=100)
    df["vol_lo"] = q.quantile(s["vol_low_q"])
    df["vol_hi"] = q.quantile(s["vol_high_q"])
    v = df["volume"].replace(0, np.nan)
    df["vol_ratio"] = (v / v.rolling(20, min_periods=5).mean()).fillna(1.0)
    return df


def checks(row, cfg):
    """Return (long_checks, short_checks) as ordered dicts of bools."""
    s = cfg["strategy"]
    hr = row.name.hour
    common = {
        "Volatility in range": bool(row.vol_lo <= row.sigma_pct <= row.vol_hi),
        "Trading hour": hr not in s["blocked_hours_utc"],
        "Volume active": bool(row.vol_ratio >= s.get("min_volume_ratio", 0)),
    }
    lg = {
        "EMA50 above EMA200": bool(row.ema_fast > row.ema_slow),
        "Price above EMA200": bool(row.close > row.ema_slow),
        "Supertrend up": bool(row.st_dir == 1),
        "MACD histogram > 0": bool(row.macd_hist > 0),
        f"RSI {s['rsi_long'][0]}-{s['rsi_long'][1]}":
            bool(s["rsi_long"][0] < row.rsi < s["rsi_long"][1]),
        **common,
    }
    sh = {
        "EMA50 below EMA200": bool(row.ema_fast < row.ema_slow),
        "Price below EMA200": bool(row.close < row.ema_slow),
        "Supertrend down": bool(row.st_dir == -1),
        "MACD histogram < 0": bool(row.macd_hist < 0),
        f"RSI {s['rsi_short'][0]}-{s['rsi_short'][1]}":
            bool(s["rsi_short"][0] < row.rsi < s["rsi_short"][1]),
        **common,
    }
    return lg, sh


# ------------------------------------------------------- multi-timeframe
def closed(df, bar):
    """Drop the still-forming last candle."""
    now = pd.Timestamp.now(tz="UTC")
    return df[df.index + bar <= now]


def tf_summary(df, n_candles, bar):
    d = df.copy()
    d["ema_fast"], d["ema_slow"] = ema(d["close"], 50), ema(d["close"], 200)
    m = ema(d["close"], 12) - ema(d["close"], 26)
    d["macd_hist"] = m - ema(m, 9)
    d["rsi"] = rsi(d["close"], 14)
    d["st_dir"], d["st_line"] = supertrend(d, 10, 3.0)
    dc = closed(d, bar)
    r = dc.iloc[-1] if len(dc) else d.iloc[-1]
    votes = {"EMA50>200": bool(r.ema_fast > r.ema_slow),
             "Price>EMA200": bool(r.close > r.ema_slow),
             "Supertrend up": bool(r.st_dir == 1),
             "MACD>0": bool(r.macd_hist > 0)}
    up = sum(votes.values())
    bias = "Bullish" if up >= 3 else "Bearish" if up <= 1 else "Mixed"
    rnd = lambda x: None if pd.isna(x) else round(float(x), 2)
    return {
        "bias": bias, "score": up, "votes": votes, "rsi": rnd(r.rsi),
        "bar_sec": int(bar.total_seconds()),
        "forming": bool(len(dc) < len(d)),
        "candles": [[int(t.timestamp()), rnd(x.open), rnd(x.high), rnd(x.low),
                     rnd(x.close), rnd(x.ema_fast), rnd(x.ema_slow),
                     rnd(x.st_line), int(x.st_dir)]
                    for t, x in d.tail(n_candles).iterrows()],
    }


def resample(df, rule):
    o = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last",
         "volume": "sum"})
    return o.dropna()


def fetch_news(cfg):
    """High-impact events this week (ForexFactory public feed)."""
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "golddesk"})
        with urllib.request.urlopen(req, timeout=20) as r:
            ev = json.load(r)
    except Exception as e:
        print("News fetch failed:", e)
        return None
    cur = set(cfg["news"]["currencies"])
    out = []
    for e in ev:
        if e.get("impact") == "High" and e.get("country") in cur:
            t = pd.Timestamp(e["date"]).tz_convert("UTC")
            out.append({"time": t.isoformat(), "title": e["title"],
                        "country": e["country"],
                        "forecast": e.get("forecast") or "",
                        "previous": e.get("previous") or ""})
    return sorted(out, key=lambda x: x["time"])


def fetch_spot():
    try:
        req = urllib.request.Request("https://api.gold-api.com/price/XAU",
                                     headers={"User-Agent": "golddesk"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return float(json.load(r)["price"])
    except Exception as e:
        print("Spot fetch failed:", e)
        return None


# ------------------------------------------------------------ backtest
def simulate(df, cfg):
    s, r = cfg["strategy"], cfg["risk"]
    eq = r["start_balance"]
    trades, curve = [], []
    pos = None
    idx = df.index
    rows = list(df.itertuples())
    start = s["garch_window"] + 1
    for i in range(start, len(df)):
        bar = rows[i]
        # manage open position on this bar (conservative: SL first)
        if pos:
            hit = None
            if pos["side"] == 1:
                if bar.low <= pos["sl"]:
                    hit = ("SL", pos["sl"])
                elif bar.high >= pos["tp"]:
                    hit = ("TP", pos["tp"])
            else:
                if bar.high >= pos["sl"]:
                    hit = ("SL", pos["sl"])
                elif bar.low <= pos["tp"]:
                    hit = ("TP", pos["tp"])
            if hit:
                px = hit[1]
                risk_pts = abs(pos["entry"] - pos["sl0"])
                R = (pos["side"] * (px - pos["entry"]) - r["spread_usd"]) / risk_pts
                pnl = pos["risk_usd"] * R
                eq += pnl
                pos.update(exit_time=idx[i].isoformat(), exit=round(px, 2),
                           reason=hit[0], R=round(R, 2), pnl=round(pnl, 2),
                           balance=round(eq, 2))
                trades.append(pos)
                pos = None
        # new entry decided on previous closed bar, filled at this open
        if pos is None:
            prev = df.iloc[i - 1]
            if not np.isnan(prev.sigma_pct) and not np.isnan(prev.vol_hi):
                lg, sh = checks(prev, cfg)
                side = 1 if all(lg.values()) else -1 if all(sh.values()) else 0
                if side and s.get("fresh_signal_only"):
                    # only enter when the setup is new (not on the bar before)
                    lg2, sh2 = checks(df.iloc[i - 2], cfg)
                    if all((lg2 if side == 1 else sh2).values()):
                        side = 0
                if side:
                    entry = bar.open
                    sig_px = entry * prev.sigma_pct / 100
                    sl = entry - side * s["sl_sigma"] * sig_px
                    tp = entry + side * s["tp_sigma"] * sig_px
                    pos = {"id": len(trades) + 1,
                           "side": "BUY" if side == 1 else "SELL",
                           "time": idx[i].isoformat(), "entry": round(entry, 2),
                           "sl": round(sl, 2), "sl0": sl, "tp": round(tp, 2),
                           "risk_usd": round(eq * r["risk_pct"] / 100, 2)}
                    pos["side"] = side
        if i % 6 == 0 or i == len(df) - 1:
            curve.append([idx[i].isoformat(), round(eq, 2)])
    for t in trades:
        t["side"] = "BUY" if t["side"] == 1 else "SELL"
        t.pop("sl0", None)
    if pos:
        last = rows[-1].close
        pos["unrealized"] = round(
            pos["risk_usd"] * (pos["side"] * (last - pos["entry"]))
            / abs(pos["entry"] - pos["sl0"]), 2)
        pos["side"] = "BUY" if pos["side"] == 1 else "SELL"
        pos.pop("sl0", None)
    return trades, pos, curve, eq


def stats(trades, curve, start_bal, final):
    if not trades:
        return {"trades": 0}
    pnl = np.array([t["pnl"] for t in trades])
    gw, gl = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    eqs = np.array([c[1] for c in curve])
    dd = (eqs / np.maximum.accumulate(eqs) - 1).min()
    rets = np.diff(eqs) / eqs[:-1]
    sharpe = (rets.mean() / rets.std() * math.sqrt(365 * 4)) if rets.std() else 0
    return {
        "trades": int(len(trades)),
        "return_pct": round(float(final / start_bal - 1) * 100, 1),
        "win_rate": round(float((pnl > 0).mean()) * 100, 1),
        "profit_factor": round(float(gw / gl), 2) if gl else None,
        "max_dd_pct": round(float(dd) * 100, 1),
        "sharpe": round(float(sharpe), 2),
        "avg_R": round(float(np.mean([t["R"] for t in trades])), 2),
    }


# ------------------------------------------------------------ telegram
def telegram(text):
    tok, chat = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not (tok and chat):
        return
    data = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
    try:
        urllib.request.urlopen(
            f"https://api.telegram.org/bot{tok}/sendMessage", data, timeout=15)
    except Exception as e:
        print("Telegram failed:", e)


def notify(prev, out):
    if not prev:
        return
    old_open = (prev.get("open_trade") or {}).get("time")
    new_open = (out.get("open_trade") or {}).get("time")
    if new_open and new_open != old_open:
        t = out["open_trade"]
        telegram(f"Gold Desk: NEW {t['side']} @ {t['entry']}\n"
                 f"SL {t['sl']}  TP {t['tp']}")
    if old_open and new_open != old_open:
        done = [t for t in out["trades"] if t["time"] == old_open]
        if done:
            t = done[0]
            telegram(f"Gold Desk: {t['side']} closed by {t['reason']} "
                     f"@ {t['exit']}  ({t['R']:+}R, ${t['pnl']:+})")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    a = ap.parse_args()
    cfg = load_config()

    sym = cfg["data"]["symbol"]
    if a.sample:
        df, source = sample_data(), "Synthetic sample data (test)"
        m15 = df.tail(2000)
        d1 = resample(df, "1D")
    else:
        df = fetch_yahoo(sym, cfg["data"]["period"], "1h")
        m15 = fetch_yahoo(sym, "60d", "15m")
        d1 = fetch_yahoo(sym, "3y", "1d")
        source = f"Yahoo Finance {sym}"
    H = pd.Timedelta
    h4 = resample(df, "4h")
    n = cfg["dashboard"]["chart_bars"]
    tfs = {"M15": tf_summary(m15, n, H(minutes=15)),
           "H1": tf_summary(df, n, H(hours=1)),
           "H4": tf_summary(h4, n, H(hours=4)),
           "D1": tf_summary(d1, n, H(days=1))}
    live_fut = float(m15["close"].iloc[-1]) if len(m15) else float(df["close"].iloc[-1])
    df = closed(df, H(hours=1))       # strategy uses closed H1 bars only
    news = None if a.sample else fetch_news(cfg)
    spot = None if a.sample else fetch_spot()

    df = build(df, cfg)
    trades, open_trade, curve, final = simulate(df, cfg)
    st = stats(trades, curve, cfg["risk"]["start_balance"], final)
    last = df.iloc[-1]
    lg, sh = checks(last, cfg)
    now = pd.Timestamp.now(tz="UTC")
    win = pd.Timedelta(minutes=cfg["news"]["block_minutes"])
    near = [e for e in (news or [])
            if abs(pd.Timestamp(e["time"]) - now) <= win]
    lg["No high-impact news"] = sh["No high-impact news"] = not near
    bias = ("BUY setup" if all(lg.values()) else
            "SELL setup" if all(sh.values()) else "No setup")

    tail = df.tail(n)
    rnd = lambda x: None if pd.isna(x) else round(float(x), 2)
    out = {
        "updated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "last_bar": df.index[-1].isoformat(),
        "source": source,
        "price": rnd(live_fut),
        "change_24h_pct": rnd((last.close / df["close"].iloc[-25] - 1) * 100),
        "bias": bias,
        "indicators": {
            "EMA50": rnd(last.ema_fast), "EMA200": rnd(last.ema_slow),
            "RSI": rnd(last.rsi), "MACD hist": rnd(last.macd_hist),
            "Supertrend": "Up" if last.st_dir == 1 else "Down",
            "GARCH σ/hr %": round(float(last.sigma_pct), 3),
        },
        "checks": {"long": lg, "short": sh},
        "open_trade": open_trade,
        "trades": trades[-cfg["dashboard"]["trade_rows"]:][::-1],
        "stats": st,
        "equity": curve,
        "test_from": curve[0][0] if curve else None,
        "candles": [[int(t.timestamp()), rnd(r.open), rnd(r.high), rnd(r.low),
                     rnd(r.close), rnd(r.ema_fast), rnd(r.ema_slow),
                     rnd(r.st_line), int(r.st_dir)]
                    for t, r in tail.iterrows()],
        "timeframes": tfs,
        "news": news or [],
        "news_ok": news is not None,
        "spot": spot,
        "basis": rnd(live_fut - spot) if spot else 0,
        "futures_last": rnd(live_fut),
        "settings": cfg["strategy"] | cfg["risk"] | {"news_block_min": cfg["news"]["block_minutes"]},
    }

    prev = None
    if os.path.exists(OUT):
        try:
            with open(OUT) as f:
                prev = json.load(f)
        except Exception:
            pass
    # Yahoo futures are ~10 min delayed, so one spot-vs-futures reading is
    # noisy. Keep recent readings and use their median as the basis.
    hist = list((prev or {}).get("basis_hist", []))
    if spot:
        hist = (hist + [round(live_fut - spot, 2)])[-16:]
    out["basis_hist"] = hist
    out["basis"] = round(float(np.median(hist)), 2) if hist else 0.0
    notify(prev, out)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"{out['updated']}  price {out['price']}  {bias}  "
          f"open={bool(open_trade)}  stats={st}")


if __name__ == "__main__":
    sys.exit(main())
