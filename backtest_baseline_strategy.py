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
# 1. ML 策略 (新增)
# =====================
class MLStrategy(bt.Strategy):
    params = (('model_name', 'mlp'),)  # 'mlp' or 'gbdt'

    def __init__(self):
        # 这里只读 ml_signal，不再重复加载模型/scaler/特征列
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
# 3. 扩展 PandasData 以支持自定义列 (关键)
# =====================
class PandasData_Extended(bt.feeds.PandasData):
    # 增加一个 ml_signal 的 line
    lines = ('ml_signal',)
    params = (('ml_signal', -1),)  # -1 表示列索引，后面会指定列名

# =====================
# 4. 运行器
# =====================
def run_backtest(symbol, strategy_class, plot=True, **kwargs):
    print(f"\n{'=' * 10} Backtesting {symbol} with {strategy_class.__name__} {'=' * 10}")
    df = load_or_download_daily(symbol)
    if df.empty:
        return

    # 如果是 ML 策略，需要预计算信号
    if strategy_class == MLStrategy:
        df = prepare_ml_signal(df, **kwargs)  # 新封装的函数
        if df.empty:
            print(f"[Backtest] No ML signal data for {symbol}, skip.")
            return

    data = PandasData_Extended(
        dataname=df.set_index('date'),
        datetime=None,
        open='open', high='high', low='low', close='close', volume='volume',
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

def prepare_ml_signal(df: pd.DataFrame,
                      model_name: str = 'mlp',
                      proba_threshold_buy: float = 0.6,
                      proba_threshold_sell: float = 0.4,
                      use_quantile_transform: bool = False,
                      random_state: int = 42) -> pd.DataFrame:
    """
    预计算特征 -> 标准化 -> 模型预测 -> 生成 ml_signal -> 返回 df
    如 use_quantile_transform=True，则使用 QuantileTransformer（可选方案，避免对分布做太强假设）。
    """
    model = joblib.load(cfg.model_dir / f"{model_name}_model.pkl")
    scaler = joblib.load(cfg.model_dir / f"{model_name}_scaler.pkl")
    feats = joblib.load(cfg.model_dir / f"{model_name}_features.pkl")

    # 计算特征 (使用默认 flags)
    df_feat = compute_features(df)  # 这里可以传入 flags=...（暂保持简单）
    X = df_feat[feats].values

    # 如果启用 QuantileTransformer，可在此替换 scaler（需要额外保存/加载）
    X_scaled = scaler.transform(X)

    # 生成预测概率
    probs = model.predict_proba(X_scaled)[:, 1]

    # 生成信号: 概率 > proba_threshold_buy 买入, < proba_threshold_sell 卖出
    signals = np.zeros_like(probs)
    signals[probs > proba_threshold_buy] = 1
    signals[probs < proba_threshold_sell] = -1

    df['ml_signal'] = signals
    df['ml_signal'] = df['ml_signal'].shift(1)  # 避免未来函数，用昨天的信号交易
    df = df.dropna()
    return df
