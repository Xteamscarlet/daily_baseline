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
    params = (('model_name', 'gbdt'),)   # 【改】默认用 GBDT

    def __init__(self):
        self.order = None

    def next(self):
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
    cerebro.broker.setcommission(commission=0.001)  # 【改】加入手续费 0.1%
    results = cerebro.run()
    final_value = cerebro.broker.getvalue()
    print(f"Final Value: {final_value:.2f}")
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], df['close'], label='Price')
        plt.title(f"{symbol} - {strategy_class.__name__} | Val: {final_value:.0f}")
        plt.show()


# =====================
# 5. ML 信号生成（核心重写）
# =====================
def prepare_ml_signal(df: pd.DataFrame,
                      model_name: str = 'gbdt',      # 【改】默认 GBDT
                      proba_threshold_buy: float = None,
                      proba_threshold_sell: float = None,
                      use_quantile_transform: bool = False,
                      random_state: int = 42) -> pd.DataFrame:
    """
    支持二分类和三分类模型，自动检测并生成信号。

    二分类模型:
      - probs[:, 1] = P(涨) > 阈值 → 买入
      - probs[:, 1] = P(涨) < (1-阈值) → 卖出

    三分类模型:
      - probs[:, 2] = P(强) > 阈值 → 买入
      - probs[:, 0] = P(弱) > 阈值 → 卖出

    阈值默认用分位数自适应确定。
    """
    model = joblib.load(cfg.model_dir / f"{model_name}_model.pkl")
    scaler = joblib.load(cfg.model_dir / f"{model_name}_scaler.pkl")
    feats = joblib.load(cfg.model_dir / f"{model_name}_features.pkl")

    # 尝试读取 n_classes，默认 3（兼容旧模型）
    n_classes_path = cfg.model_dir / f"{model_name}_n_classes.pkl"
    try:
        n_classes = joblib.load(n_classes_path)
    except Exception:
        n_classes = 3  # 旧模型默认三分类

    # 计算特征
    df_feat = compute_features(df)

    # 确保特征列是数值
    df_feat = df_feat.copy()
    for c in feats:
        if c not in df_feat.columns:
            continue
        if pd.api.types.is_object_dtype(df_feat[c]):
            df_feat[c] = pd.to_numeric(df_feat[c], errors="coerce")

    X = df_feat[feats].values
    X = np.asarray(X, dtype=np.float64)

    mask = ~np.isnan(X).any(axis=1)
    if not mask.all():
        print(f"[ML] drop {(~mask).sum()} rows due to NaN before scaling")
        X = X[mask]
        df_feat = df_feat.loc[mask].reset_index(drop=True)

    X_scaled = scaler.transform(X)

    # 预测概率
    probs = model.predict_proba(X_scaled)
    n_model_classes = probs.shape[1]

    # ==================== 信号生成 ====================

    if n_model_classes == 2:
        # ---- 二分类 ----
        p_up = probs[:, 1]   # P(涨)

        # 【改】自适应阈值：用概率分布的 60/40 分位数
        if proba_threshold_buy is None:
            proba_threshold_buy = np.quantile(p_up, 0.60)
        if proba_threshold_sell is None:
            proba_threshold_sell = 1.0 - np.quantile(p_up, 0.60)

        signals = np.zeros(len(p_up))
        signals[p_up > proba_threshold_buy] = 1       # 概率高于阈值 → 买入
        signals[p_up < proba_threshold_sell] = -1     # 概率低于阈值 → 卖出

        print(f"[ML] Binary model: buy_thresh={proba_threshold_buy:.3f}, "
              f"sell_thresh={proba_threshold_sell:.3f}")

    else:
        # ---- 三分类（兼容旧模型）----
        p_weak = probs[:, 0]
        p_strong = probs[:, 2]

        if proba_threshold_buy is None:
            proba_threshold_buy = np.quantile(p_strong, 0.65)
        if proba_threshold_sell is None:
            proba_threshold_sell = np.quantile(p_weak, 0.65)

        signals = np.zeros(len(probs))
        signals[p_strong > proba_threshold_buy] = 1
        signals[p_weak > proba_threshold_sell] = -1

        print(f"[ML] 3-class model: buy_thresh={proba_threshold_buy:.3f}, "
              f"sell_thresh={proba_threshold_sell:.3f}")

    # 打印信号分布
    n_buy = (signals == 1).sum()
    n_sell = (signals == -1).sum()
    n_hold = (signals == 0).sum()
    print(f"[ML] Signal distribution: buy={n_buy}, hold={n_hold}, sell={n_sell} "
          f"(total={len(signals)})")

    # 信号合并回 df
    df_feat['ml_signal'] = signals

    df_ml = df.copy()
    df_ml['ml_signal'] = np.nan
    n_assign = min(len(df_feat), len(df_ml))
    df_ml.iloc[:n_assign, df_ml.columns.get_loc('ml_signal')] = df_feat['ml_signal'].values[:n_assign]

    # shift(1) 避免未来函数
    df_ml['ml_signal'] = df_ml['ml_signal'].shift(1)
    df_ml = df_ml.dropna(subset=['ml_signal'])
    return df_ml
