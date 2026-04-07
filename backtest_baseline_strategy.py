# backtest_baseline_strategy.py
import backtrader as bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from data_download_daily import load_or_download_daily
from feature_engineering import compute_features
from config import cfg


# =====================
# 1. ML 策略
# =====================
class MLStrategy(bt.Strategy):
    params = (('model_name', 'mlp'),)

    def __init__(self):
        self.order = None

    def next(self):
        # 直接使用预计算的 ml_signal line
        if self.datas[0].ml_signal[0] == 1:
            if not self.position:
                self.buy()
        elif self.datas[0].ml_signal[0] == -1:
            if self.position:
                self.sell()


# =====================
# 2. 规则策略
# =====================
class MaCrossStrategy(bt.Strategy):
    params = (('p_fast', 5), ('p_slow', 20),)

    def __init__(self):
        self.ma_fast = bt.indicators.SMA(self.data.close, period=self.p.p_fast)
        self.ma_slow = bt.indicators.SMA(self.data.close, period=self.p.p_slow)
        self.crossover = bt.indicators.CrossOver(self.ma_fast, self.ma_slow)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
        else:
            if self.crossover < 0:
                self.sell()


# =====================
# 3. 扩展 PandasData
# =====================
class PandasData_Extended(bt.feeds.PandasData):
    lines = ('ml_signal',)
    params = (('ml_signal', -1),)


# =====================
# 4. 运行器
# =====================
def run_backtest(symbol, strategy_class, plot=True, **kwargs):
    print(f"\n{'=' * 10} Backtesting {symbol} with {strategy_class.__name__} {'=' * 10}")
    df = load_or_download_daily(symbol)
    if df.empty:
        return

    if strategy_class == MLStrategy:
        df = prepare_ml_signal(df, **kwargs)
        if df.empty:
            print(f"[Backtest] No ML signal data for {symbol}, skip.")
            return

    df['date'] = pd.to_datetime(df['date'])
    df_indexed = df.set_index('date')

    data = PandasData_Extended(
        dataname=df_indexed,
        datetime=None,
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        ml_signal='ml_signal' if strategy_class == MLStrategy else None
    )

    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(strategy_class, **kwargs)
    cerebro.broker.setcash(100000.0)
    results = cerebro.run()
    final_value = cerebro.broker.getvalue()
    print(f"Final Value: {final_value:.2f}")
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], df['close'], label='Price')
        plt.title(f"{symbol} - {strategy_class.__name__} | Val: {final_value:.0f}")
        plt.show()


# =====================
# 5. ML 信号生成（核心修复）
# =====================
def prepare_ml_signal(df: pd.DataFrame,
                      model_name: str = 'mlp',
                      proba_threshold_buy: float = 0.38,
                      proba_threshold_sell: float = 0.38,
                      use_quantile_transform: bool = False,
                      random_state: int = 42) -> pd.DataFrame:
    """
    3 分类模型信号生成逻辑：
      - probs[:, 0] = P(弱)   → 用于卖出
      - probs[:, 1] = P(中)   → 持仓/忽略
      - probs[:, 2] = P(强)   → 用于买入

    买入条件：P(强) > proba_threshold_buy
    卖出条件：P(弱) > proba_threshold_sell
    """
    model = joblib.load(cfg.model_dir / f"{model_name}_model.pkl")
    scaler = joblib.load(cfg.model_dir / f"{model_name}_scaler.pkl")
    feats = joblib.load(cfg.model_dir / f"{model_name}_features.pkl")

    # 计算特征
    df_feat = compute_features(df)

    # 确保特征列都是数值
    df_feat = df_feat.copy()
    for c in feats:
        if c not in df_feat.columns:
            continue
        if pd.api.types.is_object_dtype(df_feat[c]):
            df_feat[c] = pd.to_numeric(df_feat[c], errors="coerce")

    X = df_feat[feats].values

    # 【修复1】用 np.asarray 转浮点，不用 astype(errors=...)
    X = np.asarray(X, dtype=np.float64)

    mask = ~np.isnan(X).any(axis=1)
    if not mask.all():
        print(f"[ML] drop {(~mask).sum()} rows due to NaN before scaling")
        X = X[mask]
        df_feat = df_feat.loc[mask].reset_index(drop=True)

    X_scaled = scaler.transform(X)

    # 【修复2】3 分类模型：用 probs[:, 2]（强）判断买入，probs[:, 0]（弱）判断卖出
    probs = model.predict_proba(X_scaled)
    n_classes = probs.shape[1]

    if n_classes == 3:
        p_weak = probs[:, 0]     # P(弱)
        p_strong = probs[:, 2]   # P(强)

        signals = np.zeros(len(probs))
        signals[p_strong > proba_threshold_buy] = 1    # 强 → 买入
        signals[p_weak > proba_threshold_sell] = -1     # 弱 → 卖出
    else:
        # 2 分类兜底
        p_positive = probs[:, 1]
        signals = np.zeros(len(probs))
        signals[p_positive > proba_threshold_buy] = 1
        signals[p_positive < (1 - proba_threshold_sell)] = -1

    # 打印信号分布，方便调参
    n_buy = (signals == 1).sum()
    n_sell = (signals == -1).sum()
    n_hold = (signals == 0).sum()
    print(f"[ML] Signal distribution: buy={n_buy}, hold={n_hold}, sell={n_sell} "
          f"(total={len(signals)})")

    df_feat['ml_signal'] = signals

    # 信号合并回原始 df
    df_ml = df.copy()
    df_ml['ml_signal'] = np.nan
    df_ml.iloc[:len(df_feat), df_ml.columns.get_loc('ml_signal')] = df_feat['ml_signal'].values

    # 用昨天的信号交易（shift 1），避免未来函数
    df_ml['ml_signal'] = df_ml['ml_signal'].shift(1)

    df_ml = df_ml.dropna(subset=['ml_signal'])
    return df_ml
