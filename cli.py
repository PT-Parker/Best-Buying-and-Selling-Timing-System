import os
import sys
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import yaml
import subprocess

from scripts.build_slides import build as build_slides
from scripts.package_deliverables import main as package_deliverables

from backtest.run_backtest import generate_signals, backtest, plot_equity


def _flatten_yf_columns(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        cols = []
        for col in df.columns:
            if isinstance(col, tuple):
                parts = [c for c in col if c and c != symbol]
                name = "_".join(parts) or (col[-1] if col else "Close")
                cols.append(name)
            else:
                cols.append(col)
        df.columns = cols
        df.columns = [c.split("_")[0] if "_" in c else c for c in df.columns]
    return df


def _download_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, auto_adjust=True)
    df = _flatten_yf_columns(df, symbol)
    df = df.rename_axis('Date').reset_index().set_index('Date')
    return df


def read_rules() -> dict:
    with open("config/rules.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def backtest_one(symbol: str, start: str, end: str, commission: float = 0.001425, slippage: float = 0.0005) -> int:
    cmd = [sys.executable, "backtest/run_backtest.py",
           "--symbol", symbol, "--start", start, "--end", end,
           "--commission", str(commission), "--slippage", str(slippage)]
    return subprocess.call(cmd)


def compute_signal(close: pd.Series):
    df = pd.DataFrame({"Close": close})
    df["EMA5"] = close.ewm(span=5, adjust=False).mean()
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    # RSI
    d = close.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    ru = up.ewm(alpha=1/14, adjust=False).mean()
    rd = dn.ewm(alpha=1/14, adjust=False).mean()
    rs = ru / (rd + 1e-12)
    df["RSI"] = 100 - (100 / (1 + rs))
    # Bollinger
    mid = close.rolling(20).mean(); sd = close.rolling(20).std()
    df["UP"], df["LO"] = mid + 2.0*sd, mid - 2.0*sd

    c0, c1 = df["Close"].iloc[-1], df["Close"].iloc[-2]
    lo0, lo1 = df["LO"].iloc[-1], df["LO"].iloc[-2]
    up0, up1 = df["UP"].iloc[-1], df["UP"].iloc[-2]
    r0 = df["RSI"].iloc[-1]
    cross_up_lo = (c1 < lo1) and (c0 >= lo0)
    cross_dn_up = (c1 > up1) and (c0 <= up0)
    sig = "HOLD"; why = []
    if r0 < 30 and cross_up_lo:
        sig = "BUY"; why.append("RSI<30 且上穿下軌")
    if r0 > 70 and cross_dn_up:
        sig = "SELL"; why.append("RSI>70 且跌回上軌下")
    conf = "Weak"
    if sig == "BUY" and df["EMA5"].iloc[-1] > df["EMA20"].iloc[-1]:
        conf = "Strong"
    if sig == "SELL" and df["EMA5"].iloc[-1] < df["EMA20"].iloc[-1]:
        conf = "Strong"
    if sig != "HOLD" and conf == "Weak":
        conf = "Normal"
    return sig, conf, (";".join(why) if why else "無觸發")


def cmd_backtest_one(args: argparse.Namespace) -> int:
    df = _download_data(args.symbol, args.start, args.end)
    df = generate_signals(df)
    bt_df, metrics, _ = backtest(df, commission=args.commission, slippage=args.slippage)
    os.makedirs('backtest_out', exist_ok=True)
    out_png = f"backtest_out/{args.symbol}_equity.png"
    plot_equity(bt_df, out_png)
    print(f"Annual Return: {metrics['annual_return']:.2%}")
    print(f"Max Drawdown : {metrics['max_drawdown']:.2%}")
    print(f"Chart saved to {out_png}")
    return 0


def _load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def cmd_backtest_all(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    symbols: List[str] = cfg.get('symbols') or []
    period: Tuple[str, str] = tuple(cfg.get('backtest_period') or (None, None))  # type: ignore
    fees = cfg.get('fees') or {}
    if not symbols or not period[0] or not period[1]:
        print("Invalid config: require symbols and backtest_period [start,end]", file=sys.stderr)
        return 2

    start, end = period
    commission = float(fees.get('commission', args.commission))
    slippage = float(fees.get('slippage', args.slippage))

    os.makedirs('backtest_out', exist_ok=True)
    summary = []
    for sym in symbols:
        try:
            df = _download_data(sym, start, end)
            df = generate_signals(df)
            bt_df, metrics, trades = backtest(df, commission=commission, slippage=slippage)
            out_png = f"backtest_out/{sym}_equity.png"
            out_csv = f"backtest_out/{sym}_trades.csv"
            plot_equity(bt_df, out_png)
            trades.to_csv(out_csv, index=True, encoding='utf-8-sig')
            summary.append((sym, metrics['annual_return'], metrics['max_drawdown']))
            print(f"[OK] {sym} -> {out_png}, {out_csv}")
        except Exception as e:
            print(f"[ERR] {sym}: {e}", file=sys.stderr)

    if summary:
        print("\nSummary (Annual, MDD):")
        for sym, ret, mdd in summary:
            print(f"- {sym}: {ret:.2%}, {mdd:.2%}")
        return 0
    return 1


def cmd_signals(args: argparse.Namespace) -> int:
    df = _download_data(args.symbol, args.start, args.end)
    df = generate_signals(df)
    sigs = df[df['signal'] != 'HOLD'][['signal', 'confidence', 'Close']].copy()
    sigs = sigs.reset_index()
    if sigs.empty:
        print("No signals in period.")
    else:
        print(sigs.head(10).to_string(index=False))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        sigs.to_csv(args.out, index=False, encoding='utf-8-sig')
        print(f"Signals saved to {args.out}")
    return 0


def cmd_backtest_all_simple(args: argparse.Namespace) -> int:
    cfg = read_rules()
    start, end = cfg.get("backtest_period", [None, None])
    if not start or not end:
        print("Invalid config: backtest_period missing", file=sys.stderr)
        return 2
    for sym in cfg.get("symbols", []):
        print(f"== Backtest {sym} {start}~{end}")
        rc = backtest_one(sym, start, end)
        if rc != 0:
            return rc
    return 0


def cmd_backtest_one_simple(args: argparse.Namespace) -> int:
    return backtest_one(args.symbol, args.start, args.end)


def cmd_signals_today(args: argparse.Namespace) -> int:
    cfg = read_rules()
    rows = []
    for sym in cfg.get("symbols", []):
        df = yf.download(sym, period="6mo", interval="1d", auto_adjust=True)
        df = _flatten_yf_columns(df, sym)
        if not df.empty:
            df = df.rename_axis('Date').reset_index().set_index('Date')
        if df.empty or 'Close' not in df.columns or df['Close'].dropna().shape[0] < 25:
            rows.append({"symbol": sym, "signal": "HOLD", "confidence": "Weak", "note": "no data"})
            continue
        series = pd.to_numeric(df["Close"], errors='coerce').dropna().tail(60)
        if series.empty:
            rows.append({"symbol": sym, "signal": "HOLD", "confidence": "Weak", "note": "no close series"})
            continue
        sig, conf, why = compute_signal(series)
        rows.append({
            "symbol": sym,
            "date": df.index[-1].date().isoformat(),
            "close": float(df["Close"].iloc[-1]),
            "signal": sig,
            "confidence": conf,
            "reason": why,
        })
    out = "signals_today.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print("Wrote", out)
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    df = _download_data(args.symbol, args.start, args.end)
    df = generate_signals(df)
    bt_df, metrics, _ = backtest(df, commission=args.commission, slippage=args.slippage)
    os.makedirs('backtest_out', exist_ok=True)
    out_png = f"backtest_out/{args.symbol}_equity.png"
    plot_equity(bt_df, out_png)
    print(f"Replotted equity to {out_png}")
    return 0


def cmd_slides(args: argparse.Namespace) -> int:
    # Reuse scripts.build_slides.build which expects a similar args object
    return int(build_slides(args)) if build_slides(args) is not None else 0


def cmd_package(args: argparse.Namespace) -> int:
    # Package deliverables into a zip under deliverables/
    package_deliverables()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='cli.py', description='Unified CLI for backtesting, signals, and plotting')
    sub = p.add_subparsers(dest='cmd', required=True)

    # backtest group
    p_bt = sub.add_parser('backtest', help='Backtesting commands')
    sub_bt = p_bt.add_subparsers(dest='subcmd', required=True)

    p_bt_one = sub_bt.add_parser('one', help='Run backtest for a single symbol')
    p_bt_one.add_argument('--symbol', required=True)
    p_bt_one.add_argument('--start', required=True)
    p_bt_one.add_argument('--end', required=True)
    p_bt_one.add_argument('--commission', type=float, default=0.001425)
    p_bt_one.add_argument('--slippage', type=float, default=0.0005)
    p_bt_one.set_defaults(func=cmd_backtest_one)

    p_bt_all = sub_bt.add_parser('all', help='Run backtest for symbols from config')
    p_bt_all.add_argument('--config', default='config/rules.yaml')
    p_bt_all.add_argument('--commission', type=float, default=0.001425)
    p_bt_all.add_argument('--slippage', type=float, default=0.0005)
    p_bt_all.set_defaults(func=cmd_backtest_all)

    # signals
    p_sig = sub.add_parser('signals', help='Show or export trading signals')
    p_sig.add_argument('--symbol', required=True)
    p_sig.add_argument('--start', required=True)
    p_sig.add_argument('--end', required=True)
    p_sig.add_argument('--out', default=None, help='Optional CSV path')
    p_sig.set_defaults(func=cmd_signals)

    # plot
    p_plot = sub.add_parser('plot', help='Re-plot equity curve for a symbol and period')
    p_plot.add_argument('--symbol', required=True)
    p_plot.add_argument('--start', required=True)
    p_plot.add_argument('--end', required=True)
    p_plot.add_argument('--commission', type=float, default=0.001425)
    p_plot.add_argument('--slippage', type=float, default=0.0005)
    p_plot.set_defaults(func=cmd_plot)

    # slides
    p_slides = sub.add_parser('slides', help='Build PPTX slides from backtest outputs')
    p_slides.add_argument('--symbols', nargs='*', help='Symbols to include (space-separated)')
    p_slides.add_argument('--config', default='config/rules.yaml', help='Config file to read symbols from')
    p_slides.add_argument('--title', default='回測成果簡報')
    p_slides.add_argument('--out', default=os.path.join('slides', 'final_20pages.pptx'))
    p_slides.set_defaults(func=cmd_slides)

    # package
    p_pkg = sub.add_parser('package', help='Package slides, backtests, and configs into a zip')
    p_pkg.set_defaults(func=cmd_package)

    # Compatibility subcommands matching provided run-block
    p_bt_all2 = sub.add_parser('backtest-all', help='Batch backtest using config/rules.yaml (simple subprocess)')
    p_bt_all2.set_defaults(func=cmd_backtest_all_simple)

    p_bt_one2 = sub.add_parser('backtest-one', help='Backtest a single symbol via existing script')
    p_bt_one2.add_argument('--symbol', required=True)
    p_bt_one2.add_argument('--start', required=True)
    p_bt_one2.add_argument('--end', required=True)
    p_bt_one2.set_defaults(func=cmd_backtest_one_simple)

    p_sig_today = sub.add_parser('signals-today', help='Generate today signal summary for symbols from config')
    p_sig_today.set_defaults(func=cmd_signals_today)

    return p


def main(argv: List[str] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
