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
        # 加载模型和特征列
        self.model = joblib.load(cfg.model_dir / f"{self.p.model_name}_model.pkl")
        self.scaler = joblib.load(cfg.model_dir / f"{self.p.model_name}_scaler.pkl")
        self.feature_cols = joblib.load(cfg.model_dir / f"{self.p.model_name}_features.pkl")

        # 标记是否已持仓
        self.order = None

    def next(self):
        # 获取当前数据的日期索引
        current_date = self.datas[0].datetime.date(0)

        # 我们需要在 next 中访问当前行的特征。
        # 最稳妥的方法是在回测前把预测结果算好，作为一个 'signal' line 传入。
        # 但为了演示通用性，这里假设我们在 dataframe 中预存了预测结果。
        # 这里我们采用一种简单的 trick：
        # 在 run_backtest 中预计算好 'ml_signal' 列，这里直接读取。

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
            if self.crossover > 0: self.buy()
        else:
            if self.crossover < 0: self.sell()


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
    if df.empty: return

    # 如果是 ML 策略，需要预计算信号
    if strategy_class == MLStrategy:
        model_name = kwargs.get('model_name', 'mlp')
        model = joblib.load(cfg.model_dir / f"{model_name}_model.pkl")
        scaler = joblib.load(cfg.model_dir / f"{model_name}_scaler.pkl")
        feats = joblib.load(cfg.model_dir / f"{model_name}_features.pkl")

        # 计算特征 (使用默认 flags)
        df_feat = compute_features(df)
        # 确保特征顺序一致
        X = df_feat[feats].values
        X_scaled = scaler.transform(X)

        # 生成预测概率
        probs = model.predict_proba(X_scaled)[:, 1]

        # 生成信号: 概率 > 0.6 买入, < 0.4 卖出 (简单示例)
        signals = np.zeros_like(probs)
        signals[probs > 0.6] = 1
        signals[probs < 0.4] = -1

        df['ml_signal'] = signals
        df['ml_signal'] = df['ml_signal'].shift(1)  # 避免未来函数，用昨天的信号交易
        df = df.dropna()

    data = PandasData_Extended(
        dataname=df.set_index('date'),
        datetime=None,
        open='open', high='high', low='low', close='close', volume='volume',
        ml_signal='ml_signal'  # 映射列名
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
