from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go
import xgboost as xgb
import streamlit as st
from sklearn.metrics import precision_recall_fscore_support

from core.backtest import FeesConfig, StrategyConfig
from core.features import FeatureConfig, build_features
from core import inference as core_inference
from core import labeling
from services import backtest as backtest_service
from services import data_source, registry, signals as signal_service
from services.data_source import DataSourceMode


st.set_page_config(page_title="Best Buying & Selling Timing System", layout="wide")


def _ensure_state():
    st.session_state.setdefault("pending_orders", [])
    st.session_state.setdefault("op_logs", [])
    st.session_state.setdefault("last_refresh_ts", None)


def _init_configs() -> Tuple[StrategyConfig, FeesConfig]:
    strategy_cfg, fees_cfg = data_source.load_strategy_config()
    return strategy_cfg, fees_cfg


def _strategy_from_sidebar(default: StrategyConfig) -> StrategyConfig:
    ema_fast = st.sidebar.slider("EMA 快線", 3, 20, default.ema_fast)
    ema_slow = st.sidebar.slider("EMA 慢線", 10, 60, default.ema_slow)
    rsi_buy = st.sidebar.slider("RSI 買進閾值", 10, 50, int(default.rsi_buy_lt))
    rsi_sell = st.sidebar.slider("RSI 賣出閾值", 50, 90, int(default.rsi_sell_gt))
    bb_period = st.sidebar.slider("布林通道期間", 10, 40, default.bollinger_period)
    bb_std = st.sidebar.slider("布林通道標準差", 1.0, 3.0, float(default.bollinger_std), 0.1)
    take_profit = st.sidebar.slider("停利 %", 0.01, 0.10, float(default.take_profit_pct), 0.01)
    stop_loss = st.sidebar.slider("停損 %", 0.005, 0.08, float(default.stop_loss_pct), 0.005)
    return StrategyConfig(
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        rsi_period=default.rsi_period,
        rsi_buy_lt=float(rsi_buy),
        rsi_sell_gt=float(rsi_sell),
        bollinger_period=bb_period,
        bollinger_std=float(bb_std),
        take_profit_pct=take_profit,
        stop_loss_pct=stop_loss,
        cooldown_days=default.cooldown_days,
    )


def _load_signal_summary(
    symbol: str,
    start: str,
    end: str,
    strategy: StrategyConfig,
    mode: DataSourceMode,
    model_ctx: signal_service.ModelContext | None,
):
    try:
        return signal_service.summarize_signals(
            symbols=[symbol],
            start=start,
            end=end,
            strategy=strategy,
            mode=mode,
            model=model_ctx,
        )
    except Exception as exc:  # pragma: no cover - UI error path
        st.error(f"載入訊號摘要失敗：{exc}")
        return {"rows": [], "metadata": {"anomalies": []}}


def _render_signal_section(summary: dict, strategy: StrategyConfig, prob_threshold: float, label_cfg: labeling.LabelConfig, fees: FeesConfig):
    st.subheader("今日決策卡")
    rows = summary.get("rows") or []
    anomalies = summary.get("metadata", {}).get("anomalies") or []
    if not rows:
        st.info("尚無可用訊號，請確認資料期間與網路連線。")
        if anomalies:
            st.warning(" / ".join(anomalies))
        return

    row = rows[0]
    score = row.get("score")
    signal = row.get("signal")
    action = "需先訓練模型" if score is None else ("考慮進場" if score >= prob_threshold else "觀望")
    price = float(row.get("close", 0) or 0)
    tp_px = price * (1 + label_cfg.take_profit_pct)
    sl_px = price * (1 + label_cfg.stop_loss_pct)
    score_val = score or 0.0
    expected_return = score_val * label_cfg.take_profit_pct + (1 - score_val) * label_cfg.stop_loss_pct
    if score is None:
        action = "需先訓練模型"
    elif score_val < prob_threshold or expected_return <= 0:
        action = "觀望"
    else:
        action = "考慮進場"
    risk_per_share = price * abs(label_cfg.stop_loss_pct) if label_cfg.stop_loss_pct != 0 else 0
    risk_budget = fees.initial_capital * 0.01  # 1% 風險預算
    shares_suggest = int(risk_budget // risk_per_share) if (risk_per_share > 0 and action == "考慮進場") else 0

    cols = st.columns(5)
    cols[0].metric("收盤價", f"{price:,.2f}")
    cols[1].metric("模型機率", f"{score_val:.2f}")
    cols[2].metric("期望報酬", f"{expected_return:.2%}")
    cols[3].metric("技術訊號", signal or "無")
    cols[4].metric("建議", action)
    if action == "考慮進場":
        st.caption(
            f"進場價 ≈ {price:,.2f} ｜ 停利 {tp_px:,.2f} (+{label_cfg.take_profit_pct:.2%}) ｜ 停損 {sl_px:,.2f} ({label_cfg.stop_loss_pct:.2%}) ｜ 預估持有 {label_cfg.horizon_days} 天"
        )
        st.caption(
            f"風險預算 1%：建議部位約 {shares_suggest} 股（風險={risk_budget:,.0f} / 每股風險={risk_per_share:,.2f}），門檻 {prob_threshold:.2f}，資料日期 {row.get('as_of', 'N/A')}"
        )
    else:
        st.caption(
            f"預測方向為下或期望報酬為負，暫不建議進場；參考價位：停利 {tp_px:,.2f} / 停損 {sl_px:,.2f}，資料日期 {row.get('as_of', 'N/A')}"
        )
    direction = "上" if expected_return > 0 else "下"
    st.caption(f"預測方向：{direction}，期望報酬 {expected_return:.2%}（若為負則建議觀望）")
    if anomalies:
        st.warning(" / ".join(anomalies))
    if action == "考慮進場":
        if st.button("加入待執行清單", key=f"add_order_{row.get('as_of', '')}"):
            _add_pending_order(
                symbol=row.get("symbol", ""),
                price=price,
                tp=tp_px,
                sl=sl_px,
                shares=shares_suggest,
                horizon_days=label_cfg.horizon_days,
                score=score_val,
                expected_return=expected_return,
            )


def _render_backtest_section(result: backtest_service.BacktestResult, title: str = "回測績效", initial_capital: float | None = None):
    st.subheader(title)
    cols = st.columns(3)
    cols[0].metric("年化報酬", f"{result.metrics['annual_return']:.2%}")
    cols[1].metric("最大回撤", f"{result.metrics['max_drawdown']:.2%}")
    cols[2].metric("勝率", f"{result.metrics['win_rate']:.2%}")
    if initial_capital is None and not result.equity_curve.empty:
        initial_capital = float(result.equity_curve["equity"].iloc[0])
    if initial_capital is not None and not result.equity_curve.empty:
        ending = float(result.equity_curve["equity"].iloc[-1])
        pnl = ending - initial_capital
        pnl_cols = st.columns(2)
        pnl_cols[0].metric("期末資金", f"{ending:,.0f}")
        pnl_cols[1].metric("總損益", f"{pnl:,.0f}")

    st.caption("資金曲線")
    equity_df = result.equity_curve.rename_axis("Date").reset_index()
    if not equity_df.empty:
        y_min = equity_df["equity"].min()
        y_max = equity_df["equity"].max()
        padding = max((y_max - y_min) * 0.05, 1)
        fig = go.Figure(
            go.Scatter(
                x=equity_df["Date"],
                y=equity_df["equity"],
                mode="lines+markers+text",
                text=[f"{val:,.0f}" for val in equity_df["equity"]],
                textposition="top center",
                textfont=dict(size=12, color="#d7e3ff"),
                fill="tozeroy",
                line=dict(color="#7fa3ff", width=2),
                marker=dict(size=6, color="#5c7cff"),
                name="Equity",
            )
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            height=260,
            yaxis=dict(range=[y_min - padding, y_max + padding]),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption("交易紀錄（含損益）")
    trades = result.trades.copy() if not result.trades.empty else pd.DataFrame(columns=["entry_date", "exit_date", "pnl"])
    if not trades.empty:
        trades = trades.sort_values("entry_date").reset_index(drop=True)
        trades.insert(0, "trade_id", range(1, len(trades) + 1))
        trades["cumulative_pnl"] = trades["pnl"].cumsum()
    display_trades = trades.rename(
        columns={
            "trade_id": "交易序號",
            "entry_date": "進場日",
            "exit_date": "出場日",
            "entry_price": "進場價",
            "exit_price": "出場價",
            "shares": "股數",
            "holding_days": "持有天數",
            "pnl": "損益",
            "cumulative_pnl": "累計損益",
        }
    )
    st.dataframe(display_trades, use_container_width=True)
    if not trades.empty:
        st.download_button(
            "下載交易紀錄",
            trades.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{result.symbol}_trades.csv",
        )


def _render_validation_chart(prices: pd.DataFrame, trades: pd.DataFrame, title: str):
    if prices.empty:
        return
    fig = go.Figure(
        data=[
            go.Scatter(
                x=prices["date"],
                y=prices["close"],
                mode="lines+markers+text",
                text=[f"{val:,.2f}" for val in prices["close"]],
                textposition="top center",
                textfont=dict(size=11, color="#b0c4de"),
                line=dict(color="#5c7080", width=2),
                marker=dict(size=6, color="#4f6479"),
                fill="tozeroy",
                fillcolor="rgba(124,144,169,0.2)",
                name="收盤價",
            )
        ]
    )
    if trades is not None and not trades.empty:
        fig.add_trace(
            go.Scatter(
                x=trades["entry_date"],
                y=trades["entry_price"],
                mode="markers",
                marker=dict(color="#4CAF50", symbol="triangle-up", size=10),
                name="進場",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=trades["exit_date"],
                y=trades["exit_price"],
                mode="markers",
                marker=dict(color="#F44336", symbol="triangle-down", size=10),
                name="出場",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="日期",
        yaxis_title="價格",
        margin=dict(l=0, r=0, t=30, b=0),
        height=320,
        yaxis=dict(
            range=[
                prices["close"].min() - max((prices["close"].max() - prices["close"].min()) * 0.05, 0.5),
                prices["close"].max() + max((prices["close"].max() - prices["close"].min()) * 0.05, 0.5),
            ]
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


def _load_latest_model():
    latest = registry.latest()
    if not latest:
        return None
    model_path = latest.get("model_path")
    feature_cols = latest.get("feature_columns")
    if not model_path or not feature_cols:
        return None
    booster = core_inference.load_booster(model_path)
    return booster, feature_cols


def _forecast_price_with_model(
    prices: pd.DataFrame,
    days: int,
    booster,
    feature_cols: Sequence[str],
    label_cfg: labeling.LabelConfig,
) -> pd.DataFrame:
    if prices.empty:
        raise ValueError("無可用價格資料")
    prices = prices.copy()
    numeric_base = ["open", "high", "low", "close", "volume"]
    for col in numeric_base:
        if col in prices.columns:
            prices[col] = pd.to_numeric(prices[col], errors="coerce")
    feature_cfg = FeatureConfig()
    feats = build_features(prices, feature_cfg)
    numeric_cols = set(feature_cols) | set(numeric_base)
    for col in numeric_cols:
        if col in feats.columns:
            feats[col] = pd.to_numeric(feats[col], errors="coerce")
    feats = feats.dropna(subset=feature_cols)
    if feats.empty:
        raise ValueError("特徵不足，無法預測")
    last_row = feats.iloc[-1]
    vector = (
        last_row[feature_cols]
        .astype(float, errors="ignore")
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .to_numpy()
        .reshape(1, -1)
    )
    dmatrix = xgb.DMatrix(vector, feature_names=list(feature_cols))
    prob = float(booster.predict(dmatrix)[0])
    expected_return = prob * label_cfg.take_profit_pct + (1 - prob) * label_cfg.stop_loss_pct

    last_price = prices.sort_values("date")["close"].iloc[-1]
    start_date = prices["date"].max()
    future_dates = pd.bdate_range(start=start_date + pd.Timedelta(days=1), periods=days)
    forecasts = []
    price = last_price
    for dt in future_dates:
        price *= 1 + expected_return
        forecasts.append({"date": dt, "forecast_price": price})
    return pd.DataFrame(forecasts)


def _log_event(event: str, detail: dict):
    entry = {
        "time": datetime.utcnow().isoformat() + "Z",
        "event": event,
        "detail": detail,
    }
    st.session_state["op_logs"].append(entry)


def _add_pending_order(symbol: str, price: float, tp: float, sl: float, shares: int, horizon_days: int, score: float, expected_return: float):
    order = {
        "id": int(datetime.utcnow().timestamp() * 1000),
        "symbol": symbol,
        "price": price,
        "tp": tp,
        "sl": sl,
        "shares": shares,
        "horizon_days": horizon_days,
        "score": score,
        "expected_return": expected_return,
        "status": "pending",
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    st.session_state["pending_orders"].append(order)
    _log_event("add_order", order)
    st.success("已加入待執行清單")


def _render_model_section(symbol: str, data_mode: DataSourceMode, strategy: StrategyConfig, fees: FeesConfig, prob_threshold: float):
    st.subheader("模型版本 / 績效")
    latest = registry.latest()
    if latest:
        metrics = latest.get("metrics", {})
        st.write(
            f"最新模型：`{Path(latest.get('model_path', 'model')).name}` "
            f"(精確度 {metrics.get('precision', 0):.2f}, 召回率 {metrics.get('recall', 0):.2f})"
        )
    else:
        st.info("尚未有模型版本紀錄。")

    st.caption(f"模型進場機率門檻：{prob_threshold:.2f}")
    train_col1, train_col2 = st.columns([1, 3])
    trigger_train = train_col1.button("以目前資料重新訓練模型")
    progress_bar = train_col2.empty()
    if trigger_train:
        try:
            progress_bar = progress_bar.progress(5, text="初始化模型訓練...")
            (
                artifacts,
                validation_metrics,
                windows,
                validation_backtest,
                validation_prices,
            ) = _train_model_with_windows(
                symbol,
                data_mode,
                strategy,
                fees,
                prob_threshold,
                validation_days=st.session_state.get("validation_days_selected", 120),
                validation_end=st.session_state.get("validation_end_selected", date.today()),
            )
            progress_bar.progress(35, text="訓練完成，寫入 registry...")
            st.session_state.pop("model_ctx", None)
            registry.append(
                {
                    "model_path": str(artifacts.model_path),
                    "metrics": artifacts.metrics,
                    "feature_columns": artifacts.feature_columns,
                    "trained_at": datetime.utcnow().isoformat() + "Z",
                    "symbol": symbol,
                    "window_train": windows["train"],
                    "window_validation": windows["validation"],
                    "validation_metrics": validation_metrics or {},
                    "prob_threshold": prob_threshold,
                }
            )
            if artifacts.model is not None:
                st.session_state["model_ctx"] = signal_service.ModelContext(
                    booster=artifacts.model.get_booster(),
                    feature_columns=artifacts.feature_columns,
                    threshold=prob_threshold,
                )
            st.session_state.pop("latest_decision_summary", None)
            st.session_state.pop("pending_orders", None)
            st.session_state.pop("op_logs", None)
            st.session_state.pop("latest_decision_summary", None)
            st.session_state["decision_needs_refresh"] = True
            progress_bar.progress(80, text="驗證回測計算中...")
            st.success(
                f"模型已重新訓練並寫入 registry。訓練期間 {windows['train']}, 驗證期間 {windows['validation']}。"
            )
            st.write("訓練指標", artifacts.metrics)
            if validation_metrics:
                st.write("驗證指標", validation_metrics)
            else:
                st.info("驗證期間資料不足，無法計算指標。")
            st.session_state["latest_validation_bt"] = validation_backtest
            st.session_state["latest_validation_window"] = windows["validation"]
            st.session_state["latest_validation_prices"] = validation_prices
            progress_bar.progress(100, text="完成！")
        except Exception as exc:  # pragma: no cover - UI error path
            st.error(f"模型訓練失敗：{exc}")
            progress_bar.empty()


def _render_feature_preview(prices: pd.DataFrame):
    st.subheader("特徵預覽 (最近 10 筆)")
    feats = build_features(prices, FeatureConfig())
    st.dataframe(feats.tail(10), use_container_width=True)


def _update_order_status(order_id: int, status: str, fill_price: float | None = None):
    orders = st.session_state.get("pending_orders", [])
    for od in orders:
        if od["id"] == order_id:
            od["status"] = status
            od["updated_at"] = datetime.utcnow().isoformat() + "Z"
            if fill_price is not None:
                od["fill_price"] = fill_price
            _log_event(f"order_{status}", od)
            return od
    return None


def _render_pending_orders(price_history: pd.DataFrame):
    st.subheader("待執行清單 / 模擬下單")
    orders = st.session_state.get("pending_orders", [])
    if not orders:
        st.info("目前沒有待執行單。")
        return
    latest_price = None
    if not price_history.empty:
        latest_price = float(price_history.sort_values("date")["close"].iloc[-1])
    for od in list(orders):
        cols = st.columns([3, 2, 2, 2, 2])
        status = od.get("status", "pending")
        cols[0].write(f"{od.get('symbol')} ｜狀態：{status}")
        cols[1].write(f"價格 {od.get('price', 0):,.2f}")
        cols[2].write(f"股數 {od.get('shares', 0)}")
        cols[3].write(f"TP {od.get('tp', 0):,.2f} / SL {od.get('sl', 0):,.2f}")
        cols[4].write(f"期望報酬 {od.get('expected_return', 0):.2%}")
        if status == "pending":
            exec_price = latest_price or od.get("price", 0)
            if cols[0].button("模擬成交", key=f"exec_{od['id']}"):
                _update_order_status(od["id"], "executed", exec_price)
                st.success(f"模擬成交於 {exec_price:,.2f}")
            if cols[1].button("取消", key=f"cancel_{od['id']}"):
                _update_order_status(od["id"], "canceled")
                st.warning("已取消該筆待執行單")
    if st.button("清空已結束單"):
        st.session_state["pending_orders"] = [od for od in orders if od.get("status") == "pending"]
        st.success("已清空已結束的待執行單")


def _render_op_logs():
    st.subheader("操作日誌")
    logs = st.session_state.get("op_logs", [])
    if not logs:
        st.info("尚無操作日誌。")
        return
    df = pd.DataFrame(logs)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("下載日誌", df.to_csv(index=False).encode("utf-8-sig"), file_name="op_logs.csv")


def _train_model_with_windows(
    symbol: str,
    mode: DataSourceMode,
    strategy: StrategyConfig,
    fees: FeesConfig,
    prob_threshold: float,
    validation_days: int,
    validation_end: date,
):
    validation_end = validation_end or date.today()
    validation_start = validation_end - timedelta(days=validation_days)
    train_end = validation_start - timedelta(days=1)
    train_start = train_end - timedelta(days=365)

    feature_cfg = FeatureConfig()
    label_cfg = data_source.load_label_config()
    train_prices = data_source.load_price_history(
        [symbol],
        train_start.isoformat(),
        train_end.isoformat(),
        mode=mode,
    )
    if train_prices.empty:
        raise ValueError("訓練期間沒有可用資料")

    train_features = build_features(train_prices, feature_cfg)
    rand_seed = int(datetime.utcnow().timestamp())
    artifacts = core_inference.train_classifier(
        train_features,
        label_config=label_cfg,
        hyperparameters={"random_state": rand_seed},
    )

    validation_prices = data_source.load_price_history(
        [symbol],
        validation_start.isoformat(),
        validation_end.isoformat(),
        mode=mode,
    )
    validation_metrics = None
    validation_backtest = None
    val_features = pd.DataFrame()
    if not validation_prices.empty and artifacts.model is not None:
        val_features = build_features(validation_prices, feature_cfg)
        # 以指定門檻產生預測，若無交易則自動放寬
        X_val_all = (
            val_features[artifacts.feature_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
        )
        proba_all = artifacts.model.predict_proba(X_val_all)[:, 1]
        threshold = prob_threshold
        pred_full = (proba_all >= threshold).astype(int)
        if pred_full.sum() == 0:
            # 自動放寬門檻確保至少有機會產生交易
            threshold = max(0.20, min(0.5, max(proba_all.mean(), prob_threshold * 0.8)))
            pred_full = (proba_all >= threshold).astype(int)
        prediction_series = pd.Series(pred_full, index=val_features.index)
        val_labeled = labeling.assign_labels(val_features, label_cfg)
        val_labeled = val_labeled.dropna(subset=["label"])
        val_labeled = val_labeled[val_labeled["label"] != 0]
        val_labeled.loc[val_labeled["label"] == -1, "label"] = 0
        if not val_labeled.empty:
            y_true = val_labeled["label"]
            y_pred = prediction_series.loc[val_labeled.index]
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            validation_metrics = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
            prediction_series.loc[val_labeled.index] = y_pred
            validation_backtest = backtest_service.model_backtest_from_predictions(
                val_features,
                prediction_series.tolist(),
                label_cfg,
                fees,
                symbol,
            )

    windows = {
        "train": f"{train_start.isoformat()} ~ {train_end.isoformat()}",
        "validation": f"{validation_start.isoformat()} ~ {validation_end.isoformat()}",
    }

    return artifacts, validation_metrics, windows, validation_backtest, validation_prices


def main():
    _ensure_state()
    strategy_cfg, fees_cfg = _init_configs()
    watchlist = data_source.load_watchlist()

    st.title("股市進出場時機系統")
    st.caption("資料 → 特徵 → 推論/回測 → 訊號 → 視覺化（更新行情後生成決策卡）")

    end_default = date.today()
    start_default = end_default - timedelta(days=365)
    start_date, end_date = st.sidebar.date_input(
        "資料期間",
        (
            start_default,
            end_default,
        ),
    )
    prob_threshold = st.sidebar.slider("模型進場機率門檻", 0.1, 0.9, 0.4, 0.05)
    refresh_interval = st.sidebar.slider("自動刷新行情 (秒，0=關閉)", 0, 900, 0, 30)
    init_capital = st.sidebar.number_input("初始資金", min_value=1_000, max_value=10_000_000, step=1_000, value=1_500_000)
    fees_cfg = FeesConfig(
        commission=fees_cfg.commission,
        slippage=fees_cfg.slippage,
        initial_capital=float(init_capital),
    )
    symbol = st.sidebar.selectbox("監控股票", watchlist.symbols)
    data_mode = DataSourceMode.YFINANCE
    strategy = _strategy_from_sidebar(strategy_cfg)
    forecast_days = st.sidebar.select_slider("預測天數", options=list(range(1, 31)), value=5)
    show_debug = st.sidebar.checkbox("顯示除錯資訊", value=False)

    if refresh_interval > 0:
        last = st.session_state.get("last_refresh_ts")
        now = datetime.utcnow()
        if last is None:
            st.session_state["last_refresh_ts"] = now
        elif (now - last).total_seconds() >= refresh_interval:
            st.session_state["last_refresh_ts"] = now
            st.experimental_rerun()

    start_str = start_date.isoformat()
    end_str = end_date.isoformat()
    validation_days = st.sidebar.slider("驗證天數", 30, 240, 120, 10)
    validation_end_date = st.sidebar.date_input("驗證結束日", date.today(), key="val_end_date")
    st.session_state["validation_days_selected"] = validation_days
    st.session_state["validation_end_selected"] = validation_end_date

    price_history = data_source.load_price_history([symbol], start_str, end_str, mode=data_mode)

    _render_model_section(symbol, data_mode, strategy, fees_cfg, prob_threshold)

    model_ctx = st.session_state.get("model_ctx")
    if model_ctx is None:
        latest_model = _load_latest_model()
        if latest_model:
            booster, feature_cols = latest_model
            model_ctx = signal_service.ModelContext(booster=booster, feature_columns=feature_cols, threshold=prob_threshold)
            st.session_state["model_ctx"] = model_ctx

    if st.button("生成價格預測"):
        try:
            ctx = model_ctx or _load_latest_model()
            if not ctx:
                st.error("尚未有訓練好的模型，請先重新訓練。")
            else:
                booster, feature_cols = ctx if isinstance(ctx, tuple) else (ctx.booster, ctx.feature_columns)
                forecast_df = _forecast_price_with_model(
                    price_history,
                    forecast_days,
                    booster,
                    feature_cols,
                    data_source.load_label_config(),
                )
                st.subheader(f"未來 {forecast_days} 日價格預測")
                st.metric("預測股價", f"{forecast_df['forecast_price'].iloc[-1]:,.2f}")
                fig = go.Figure(
                    go.Scatter(
                        x=forecast_df["date"],
                        y=forecast_df["forecast_price"],
                        mode="lines+markers+text",
                        text=[f"{val:,.2f}" for val in forecast_df["forecast_price"]],
                        textposition="top center",
                        textfont=dict(size=11, color="#cfe7ff"),
                        fill="tozeroy",
                        line=dict(color="#9ad0f5", width=2),
                        marker=dict(size=6, color="#6a9eea"),
                        name="價格預測",
                    )
                )
                fig.update_layout(
                    margin=dict(l=0, r=0, t=30, b=0),
                    height=260,
                    yaxis_title="價格",
                    xaxis_title="日期",
                    yaxis=dict(
                        range=[
                            forecast_df["forecast_price"].min()
                            - max((forecast_df["forecast_price"].max() - forecast_df["forecast_price"].min()) * 0.05, 0.5),
                            forecast_df["forecast_price"].max()
                            + max((forecast_df["forecast_price"].max() - forecast_df["forecast_price"].min()) * 0.05, 0.5),
                        ]
                    ),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(forecast_df, hide_index=True, use_container_width=True)
        except Exception as exc:
            st.error(f"無法產生預測：{exc}")

    if model_ctx is not None:
        model_ctx.threshold = prob_threshold

    decision_btn = st.button("更新行情並生成今日決策卡")
    summary = st.session_state.get("latest_decision_summary")
    if st.session_state.pop("decision_needs_refresh", False):
        with st.spinner("套用最新模型並計算決策..."):
            summary = _load_signal_summary(symbol, start_str, end_str, strategy, data_mode, model_ctx)
            st.session_state["latest_decision_summary"] = summary
    if decision_btn:
        with st.spinner("計算決策..."):
            summary = _load_signal_summary(symbol, start_str, end_str, strategy, data_mode, model_ctx)
            st.session_state["latest_decision_summary"] = summary
    if summary:
        _render_signal_section(summary, strategy, prob_threshold, data_source.load_label_config(), fees_cfg)
    else:
        st.info("請先按「生成今日決策卡」。")

    validation_bt = st.session_state.get("latest_validation_bt")
    validation_window = st.session_state.get("latest_validation_window")
    validation_prices = st.session_state.get("latest_validation_prices")
    if validation_bt and validation_window:
        _render_backtest_section(validation_bt, title=f"回測績效（驗證期間 {validation_window}）", initial_capital=fees_cfg.initial_capital)
        if validation_prices is not None:
            _render_validation_chart(validation_prices, validation_bt.trades, title="驗證期間 K 線與訊號")

    _render_feature_preview(price_history)

    _render_pending_orders(price_history)
    _render_op_logs()

    bt_result = validation_bt
    if show_debug:
        with st.expander("除錯細節"):
            st.write("策略設定", strategy)
            st.write("資料模式", data_mode.value)
            st.write("原始訊號", summary)
            if bt_result:
                st.write("回測指標", bt_result.metrics)


if __name__ == "__main__":
    main()
