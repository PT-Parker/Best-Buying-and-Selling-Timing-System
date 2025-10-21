import os, sys, time, json, signal, math, threading, queue, argparse
from datetime import datetime, timedelta
from dateutil import tz
import yaml
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
import pandas as pd

# ---- Data source: twstock (free intraday snapshot for TWSE/TPEx)
import twstock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATE_DIR = os.path.join(ROOT, "state")
LOG_DIR   = os.path.join(ROOT, "logs")
METRICS_DIR = os.path.join(ROOT, "metrics")
os.makedirs(STATE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

PID_FILE = os.path.join(STATE_DIR, "monitor.pid")
POSITIONS_FILE = os.path.join(STATE_DIR, "positions.json")
ALERTS_CSV = os.path.join(LOG_DIR, "alerts.csv")
TRADES_CSV = os.path.join(METRICS_DIR, "live_trades.csv")
EQUITY_CSV = os.path.join(METRICS_DIR, "equity_timeseries.csv")

def load_yaml(path):
  with open(path, "r", encoding="utf-8") as f:
    return yaml.safe_load(f)

CFG = load_yaml(os.path.join(ROOT, "config", "watchlist.yaml"))
TAIPEI = tz.gettz(CFG["market"]["tz"])

def now_tpe():
  return datetime.now(tz=TAIPEI)

def in_trading_window(t):
  start_h, start_m = map(int, CFG["market"]["start"].split(":"))
  end_h, end_m = map(int, CFG["market"]["end"].split(":"))
  start_dt = t.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
  end_dt   = t.replace(hour=end_h,   minute=end_m,   second=0, microsecond=0)
  # 週一週五
  is_weekday = t.weekday() < 5
  return is_weekday and (start_dt <= t <= end_dt)


def write_pid():
  with open(PID_FILE, "w") as f:
    f.write(str(os.getpid()))

def remove_pid():
  try:
    if os.path.exists(PID_FILE):
      os.remove(PID_FILE)
  except Exception:
    pass


def read_positions():
  if not os.path.exists(POSITIONS_FILE):
    return {}
  with open(POSITIONS_FILE, "r", encoding="utf-8") as f:
    return json.load(f)


def save_positions(obj):
  with open(POSITIONS_FILE, "w", encoding="utf-8") as f:
    json.dump(obj, f, ensure_ascii=False, indent=2)


def append_csv(path, row_dict, header=None):
  df = pd.DataFrame([row_dict])
  exists = os.path.exists(path)
  df.to_csv(path, mode="a", header=(not exists and header is not None), index=False, encoding="utf-8-sig")


# ---- Notifiers
def notify_line(text: str):
  token_env = CFG["notify"].get("line_token_env", "LINE_NOTIFY_TOKEN")
  token = os.environ.get(token_env, "")
  if not token:
    return
  try:
    requests.post(
      "https://notify-api.line.me/api/notify",
      headers={"Authorization": f"Bearer {token}"},
      data={"message": text},
      timeout=5,
    )
  except Exception:
    pass


def notify_discord(text: str):
  wh_env = CFG["notify"].get("discord_webhook_env", "DISCORD_WEBHOOK_URL")
  url = os.environ.get(wh_env, "")
  if not url:
    return
  try:
    requests.post(url, json={"content": text}, timeout=5)
  except Exception:
    pass


def notify_gas(payload: dict):
  wh_env = CFG["notify"].get("gas_webhook_env", "GAS_WEBHOOK_URL")
  secret_env = CFG["notify"].get("gas_secret_env", "GAS_SECRET")
  url = os.environ.get(wh_env, "")
  secret = os.environ.get(secret_env, "")
  if not url or not secret:
    print("GAS_WEBHOOK_URL or GAS_SECRET not set", file=sys.stderr)
    return

  payload_with_secret = {**payload, "secret": secret}

  session = requests.Session()
  retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504]) # Increased total retries
  session.mount('https://', HTTPAdapter(max_retries=retries))

  try:
    # Increased timeout, added verify=False (for testing), added User-Agent
    response = session.post(url, json=payload_with_secret, timeout=30, verify=False, headers={'User-Agent': 'Python-Script/1.0'})
    response.raise_for_status()
  except requests.exceptions.RequestException as e:
    print(f"Error sending to Google Apps Script: {e}", file=sys.stderr)


def notify(text, payload=None):
  mode = CFG["notify"].get("mode", "line")
  if mode == "discord":
    notify_discord(text)
  elif mode == "gas":
    notify_gas(payload or {"message": text})
  elif mode == "webhook":
    # This is the old n8n webhook, which will be removed.
    # For now, I will leave it as a fallback.
    wh_env = CFG["notify"].get("generic_webhook_env", "N8N_WEBHOOK_URL")
    url = os.environ.get(wh_env, "")
    if not url:
        return
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception:
        pass
  else:
    notify_line(text)


# ---- Minute-bar aggregator & indicators
class BarStore:
  def __init__(self):
    # symbol -> pd.DataFrame columns=[time, open, high, low, close, volume, cum_vol]
    self.map = {}

  def _init_df(self):
    return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume", "cum_vol"])

  @staticmethod
  def minute_key(dt: datetime):
    return dt.replace(second=0, microsecond=0)

  def update(self, symbol: str, dt: datetime, last_price: float, acc_vol: int):
    df = self.map.get(symbol)
    if df is None:
      df = self._init_df()
    mk = self.minute_key(dt)
    if len(df) > 0 and df["time"].iloc[-1] == mk:
      # update last bar
      df.at[df.index[-1], "close"] = last_price
      df.at[df.index[-1], "high"] = max(df["high"].iloc[-1], last_price)
      df.at[df.index[-1], "low"] = min(df["low"].iloc[-1], last_price)
      prev_cum = df["cum_vol"].iloc[-2] if len(df) > 1 else 0
      vol = max(0, acc_vol - prev_cum)
      df.at[df.index[-1], "volume"] = vol
      df.at[df.index[-1], "cum_vol"] = acc_vol
    else:
      # new bar
      prev_cum = df["cum_vol"].iloc[-1] if len(df) > 0 else 0
      vol = max(0, acc_vol - prev_cum)
      new_row = {
        "time": mk,
        "open": last_price,
        "high": last_price,
        "low": last_price,
        "close": last_price,
        "volume": vol,
        "cum_vol": acc_vol,
      }
      df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # 只保留最近 500 根
    if len(df) > 500:
      df = df.iloc[-500:].reset_index(drop=True)

    self.map[symbol] = df
    return df

  def get(self, symbol: str) -> pd.DataFrame:
    return self.map.get(symbol, self._init_df())


def ema(series: pd.Series, span: int):
  return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period=14):
  delta = close.diff()
  up = delta.clip(lower=0)
  down = -1 * delta.clip(upper=0)
  ma_up = up.ewm(com=period - 1, adjust=True, min_periods=period).mean()
  ma_down = down.ewm(com=period - 1, adjust=True, min_periods=period).mean()
  rs = ma_up / (ma_down + 1e-12)
  return 100 - (100 / (1 + rs))


# ---- Signal engine
class Engine:
  def __init__(self, barstore: BarStore):
    self.bs = barstore
    self.cooldown = {}  # symbol -> last_signal_time
    self.pos = read_positions()  # symbol -> {side, entry_price, entry_time}

  def save(self):
    save_positions(self.pos)

  def gen_signals(self, symbol, cfg):
    df = self.bs.map.get(symbol)
    if df is None or len(df) < cfg["strategy"]["min_bars_for_signal"]:
      return None
    c = df["close"].astype(float)
    ema_fast = ema(c, cfg["strategy"]["ema_fast"])
    ema_slow = ema(c, cfg["strategy"]["ema_slow"])
    r = rsi(c, cfg["strategy"]["rsi_period"])
    # cross detection
    cross_up = (ema_fast.iloc[-2] <= ema_slow.iloc[-2]) and (ema_fast.iloc[-1] > ema_slow.iloc[-1])
    cross_down = (ema_fast.iloc[-2] >= ema_slow.iloc[-2]) and (ema_fast.iloc[-1] < ema_slow.iloc[-1])
    last_r = r.iloc[-1]
    nowt = now_tpe()
    # cooldown
    cd_min = cfg["strategy"]["cooldown_minutes"]
    last_t = self.cooldown.get(symbol)
    if last_t and (nowt - last_t).total_seconds() < cd_min * 60:
      return None
    # TP/SL
    if symbol in self.pos:
      ep = float(self.pos[symbol]["entry_price"])
      px = float(c.iloc[-1])
      if cfg["strategy"]["take_profit_pct"] > 0 and (px - ep) / ep >= cfg["strategy"]["take_profit_pct"]:
        return {"action": "SELL", "reason": "TAKE_PROFIT", "price": px}
      if cfg["strategy"]["stop_loss_pct"] > 0 and (ep - px) / ep >= cfg["strategy"]["stop_loss_pct"]:
        return {"action": "SELL", "reason": "STOP_LOSS", "price": px}
    # rules
    if (cross_up and last_r >= cfg["strategy"]["buy_rsi_min"]):
      if (symbol not in self.pos) or cfg["trading"]["allow_multi_entry"]:
        return {"action": "BUY", "reason": "EMA5↑EMA20 & RSI>=min", "price": float(c.iloc[-1])}
    if (cross_down or last_r <= cfg["strategy"]["sell_rsi_max"]):
      if symbol in self.pos:
        return {"action": "SELL", "reason": "EMA5↓EMA20 or RSI<=max", "price": float(c.iloc[-1])}
    return None

  def handle_signal(self, symbol, sig):
    t = now_tpe()
    act = sig["action"]
    px = float(sig["price"])
    # alert log
    alert_row = {
      "time": t.strftime("%Y-%m-%d %H:%M:%S"),
      "symbol": symbol,
      "action": act,
      "price": px,
      "reason": sig.get("reason", ""),
    }
    append_csv(ALERTS_CSV, alert_row, header=["time", "symbol", "action", "price", "reason"])
    # push
    msg = f"{t.strftime('%H:%M')} {symbol} {px:.2f} {('買' if act=='BUY' else '賣')}"
    notify(msg, payload=alert_row)
    # position & trade pnl
    if act == "BUY":
      if symbol not in self.pos:
        self.pos[symbol] = {
          "side": "LONG",
          "entry_price": px,
          "entry_time": t.strftime("%Y-%m-%d %H:%M:%S"),
        }
    elif act == "SELL":
      if symbol in self.pos:
        ep = float(self.pos[symbol]["entry_price"])
        pnl = (px - ep) * CFG["trading"]["position_size_shares"]
        trade = {
          "close_time": t.strftime("%Y-%m-%d %H:%M:%S"),
          "symbol": symbol,
          "entry_price": ep,
          "exit_price": px,
          "shares": CFG["trading"]["position_size_shares"],
          "pnl": round(pnl, 2),
          "reason": sig.get("reason", ""),
        }
        append_csv(
          TRADES_CSV,
          trade,
          header=["close_time", "symbol", "entry_price", "exit_price", "shares", "pnl", "reason"],
        )
        # update equity curve
        eq = pd.read_csv(EQUITY_CSV) if os.path.exists(EQUITY_CSV) else pd.DataFrame(columns=["time", "equity"])
        equity = (eq["equity"].iloc[-1] if len(eq) > 0 else 0.0) + pnl
        append_csv(EQUITY_CSV, {"time": trade["close_time"], "equity": round(equity, 2)}, header=["time", "equity"])
        del self.pos[symbol]
    self.save()
    self.cooldown[symbol] = now_tpe()


def graceful_exit(signum, frame):
  print("Stopping monitor...")
  remove_pid()
  sys.exit(0)


signal.signal(signal.SIGINT, graceful_exit)
signal.signal(signal.SIGTERM, graceful_exit)


def batched(lst, n):
  for i in range(0, len(lst), n):
    yield lst[i:i + n]


def run(force: bool = False, max_seconds: int | None = None):
  write_pid()
  bs = BarStore()
  eng = Engine(bs)
  syms = list(CFG["symbols"])
  poll = int(CFG["market"]["poll_seconds"])
  batch_size = int(CFG["market"]["batch_size"])
  print(f"[LIVE] {len(syms)} symbols, poll={poll}s, batch={batch_size}, tz={CFG['market']['tz']}, force={force}, max_seconds={max_seconds}")
  # twstock 代碼表更新（第一次使用建議更新一次）
  try:
    twstock.__update_codes()
  except Exception:
    pass

  start_ts = time.time()
  while True:
    # Respect optional max run duration
    if max_seconds is not None and (time.time() - start_ts) >= max_seconds:
      print("Max seconds reached; exiting.")
      break

    t = now_tpe()
    if not in_trading_window(t) and not force:
      # 非交易時段：睡到下一分鐘，降低 API 壓力
      nxt = t.replace(second=0, microsecond=0) + timedelta(minutes=1)
      slp = max(1.0, (nxt - t).total_seconds())
      time.sleep(slp)
      if max_seconds and (time.time() - start_ts) >= max_seconds:
        print("Max seconds reached, exiting.")
        return
      continue

    t_loop_start = time.time()
    groups = list(batched(syms, batch_size))
    per_group_pause = max(0.1, (poll * 0.3) / max(1, len(groups)))  # 依交易所節流建議做間隔
    for idx, group in enumerate(groups):
      try:
        data = twstock.realtime.get(group)
      except Exception:
        data = {"success": False}
      if not data or not data.get("success", False):
        time.sleep(1)
        continue
      for symbol in group:
        info = data.get(symbol, {})
        if not info or not info.get("success", False):
          continue
        # twstock 結構：info.time (str), realtime.latest_trade_price (str), realtime.accumulate_trade_volume (str)
        rt = info.get("realtime", {})
        latest = rt.get("latest_trade_price")
        accv  = rt.get("accumulate_trade_volume")
        timestr = info.get("info", {}).get("time") or data.get("info",{}).get("time")
        if not latest or latest in ("-", "—"):
          continue
        try:
          px = float(latest)
          acc = int(accv) if accv not in (None, "", "-") else 0
        except Exception:
          continue
        try:
          dt = datetime.strptime(timestr, "%Y-%m-%d %H:%M:%S").replace(tzinfo=TAIPEI)
        except Exception:
          dt = now_tpe()
        df = bs.update(symbol, dt, px, acc)
        sig = eng.gen_signals(symbol, CFG)
        if sig:
          eng.handle_signal(symbol, sig)
      # 每組之間小幅休眠，讓總體 poll 接近設定值
      if idx < len(groups) - 1:
        time.sleep(per_group_pause)

    # 主要輪詢間隔
    time.sleep(1)
    if max_seconds and (time.time() - start_ts) >= max_seconds:
      print("Max seconds reached, exiting.")
      return

    # 主要輪詢間隔（依群組數平均分攤輪詢頻率）
    time.sleep(max(1, poll // max(1, len(groups))))

  # Clean up on normal exit
  remove_pid()


if __name__ == "__main__":
  import argparse, os
  ap = argparse.ArgumentParser()
  ap.add_argument("--force", action="store_true", help="忽略交易時段限制立即執行")
  ap.add_argument("--max-seconds", type=int, default=None, help="執行最長秒數，超過自動結束")
  args = ap.parse_args()
  try:
    run(force=args.force, max_seconds=args.max_seconds)
  finally:
    # 結束時移除 PID，避免殘留
    remove_pid()
