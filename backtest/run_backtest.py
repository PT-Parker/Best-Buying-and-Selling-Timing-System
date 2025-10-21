# backtest/run_backtest.py
# 需求：pip install pandas numpy matplotlib yfinance
import os, math, argparse
import numpy as np, pandas as pd, yfinance as yf
import matplotlib.pyplot as plt


def ema(series, span): return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0); down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def bollinger(series, period=20, std=2.0):
    ma = series.rolling(period).mean()
    sigma = series.rolling(period).std()
    upper = ma + std*sigma
    lower = ma - std*sigma
    return ma, upper, lower

def generate_signals(df, conf_bonus=True):
    df = df.copy()
    df['EMA_FAST'] = ema(df['Close'], 5)
    df['EMA_SLOW'] = ema(df['Close'], 20)
    df['RSI'] = rsi(df['Close'], 14)
    mid, up, lo = bollinger(df['Close'], 20, 2.0)
    df['BB_MID'], df['BB_UP'], df['BB_LO'] = mid, up, lo
    df['cross_up_lo'] = (df['Close'].shift(1) < df['BB_LO'].shift(1)) & (df['Close'] >= df['BB_LO'])
    df['cross_dn_up'] = (df['Close'].shift(1) > df['BB_UP'].shift(1)) & (df['Close'] <= df['BB_UP'])
    cond_buy  = (df['RSI'] < 30) & df['cross_up_lo']
    cond_sell = (df['RSI'] > 70) & df['cross_dn_up']
    df['signal'] = 'HOLD'
    df.loc[cond_buy,  'signal'] = 'BUY'
    df.loc[cond_sell, 'signal'] = 'SELL'
    conf = ['Weak']*len(df)
    if conf_bonus:
        bonus_buy  = (df['EMA_FAST'] > df['EMA_SLOW']) & cond_buy
        bonus_sell = (df['EMA_FAST'] < df['EMA_SLOW']) & cond_sell
        for i in df.index:
            if cond_buy.loc[i]:  conf[df.index.get_loc(i)]  = 'Normal'
            if cond_sell.loc[i]: conf[df.index.get_loc(i)]  = 'Normal'
        for i in df.index:
            if df.at[i,'signal']=='BUY'  and bonus_buy.loc[i]:  conf[df.index.get_loc(i)]='Strong'
            if df.at[i,'signal']=='SELL' and bonus_sell.loc[i]: conf[df.index.get_loc(i)]='Strong'
    df['confidence'] = conf
    return df

def backtest(df, commission=0.001425, slippage=0.0005, initial=1_000_000):
    df = df.copy()
    df['position'] = 0
    pos = 0; cash = initial; shares = 0
    equity = []
    for i in range(len(df)):
        sig = df.iloc[i]['signal']; px = df.iloc[i]['Close']
        if pos == 0 and sig == 'BUY':
            buy_px = px*(1+slippage)
            shares = math.floor((cash*(1-commission))/buy_px)
            cost = shares*buy_px; fee = cost*commission
            cash -= (cost + fee); pos = 1
        elif pos == 1 and sig == 'SELL':
            sell_px = px*(1-slippage)
            proceeds = shares*sell_px; fee = proceeds*commission
            cash += (proceeds - fee); pos = 0; shares = 0
        equity.append(cash + (shares*px if shares>0 else 0))
    df['equity'] = equity
    roll_max = df['equity'].cummax()
    drawdown = df['equity']/roll_max - 1
    metrics = {
        'annual_return': (df['equity'].iloc[-1]/df['equity'].iloc[0])**(252/len(df)) - 1,
        'max_drawdown' : float(drawdown.min())
    }
    trades = df[df['signal'].isin(['BUY','SELL'])][['signal','Close']]
    return df, metrics, trades

def plot_equity(df, out_png):
    plt.figure(figsize=(10,4))
    plt.plot(df.index, df['equity'])
    plt.title('Equity Curve'); plt.xlabel('Date'); plt.ylabel('TWD')
    plt.tight_layout(); plt.savefig(out_png, dpi=160)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='2330.TW')
    parser.add_argument('--start', default='2023-01-01')
    parser.add_argument('--end',   default='2025-09-30')
    parser.add_argument('--commission', type=float, default=0.001425)
    parser.add_argument('--slippage',   type=float, default=0.0005)
    args = parser.parse_args()
    df = yf.download(args.symbol, start=args.start, end=args.end, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([c for c in col if c and c != args.symbol]) or col[-1]
            if isinstance(col, tuple) else col
            for col in df.columns
        ]
        # 若仍存在重複欄位，預設取第一層名稱
        df.columns = [col.split("_")[0] if "_" in col else col for col in df.columns]
    df = df.rename_axis('Date').reset_index().set_index('Date')
    df = generate_signals(df)
    bt_df, metrics, trades = backtest(df, commission=args.commission, slippage=args.slippage)
    os.makedirs('backtest_out', exist_ok=True)
    plot_equity(bt_df, f'backtest_out/{args.symbol}_equity.png')
    trades.to_csv(f'backtest_out/{args.symbol}_trades.csv', index=True, encoding='utf-8-sig')
    print(f"Annual Return: {metrics['annual_return']:.2%}")
    print(f"Max Drawdown : {metrics['max_drawdown']:.2%}")
    print(f"Signals saved to backtest_out/{args.symbol}_trades.csv")
    print(f"Chart   saved to backtest_out/{args.symbol}_equity.png")

if __name__ == '__main__':
    main()
