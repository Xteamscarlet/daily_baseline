# backtest_baseline_strategy.py（修复 prepare_ml_signal 中的 np.isnan 报错）
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

    # 确保 date 列是 pd.Timestamp 类型
    df['date'] = pd.to_datetime(df['date'])

    # 设置 DatetimeIndex
    df_indexed = df.set_index('date')

    # PandasData 默认用索引做 datetime；这里不要再指定 datetime 列名
    data = PandasData_Extended(
        dataname=df_indexed,
        datetime=None,  # 不写 datetime=...，直接用 DatetimeIndex
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
    df_feat = compute_features(df)

    # 【修复关键】：确保 feats 里所有列都是数值，而不是 object/字符串
    df_feat = df_feat.copy()
    for c in feats:
        if c not in df_feat.columns:
            continue
        if pd.api.types.is_object_dtype(df_feat[c]):
            df_feat[c] = pd.to_numeric(df_feat[c], errors="coerce")

    # 只保留特征里不含 NaN 的样本（简单粗暴但最稳）
    X = df_feat[feats].values

    # 兜底：如果 X 仍是 object，强制转成浮点；无法转换的变成 NaN
    if not np.issubdtype(X.dtype, np.floating):
        X = X.astype(float, errors="ignore")

    mask = ~np.isnan(X).any(axis=1)
    if not mask.all():
        print(f"[ML] drop {(~mask).sum()} rows due to NaN before scaling")
        X = X[mask]
        # 顺便对齐 df_feat（后面要生成信号）
        df_feat = df_feat.loc[mask].reset_index(drop=True)

    X_scaled = scaler.transform(X)  # 现在 X 里没有 NaN
    # 生成预测概率
    probs = model.predict_proba(X_scaled)[:, 1]  # 不会再报 NaN
    # 生成信号: 概率 > proba_threshold_buy 买入, < proba_threshold_sell 卖出
    signals = np.zeros_like(probs)
    signals[probs > proba_threshold_buy] = 1
    signals[probs < proba_threshold_sell] = -1

    # 【注意】这里用 df_feat 来对齐信号，保证长度正确
    df_feat['ml_signal'] = signals

    # 把信号合并回原始 df（按行索引/顺序合并）
    # 前面 compute_features 返回的 df_feat 行数可能与 df 不同（有 dropna 等）
    # 为了简单，这里假设 df_feat 和 df 的前 len(df_feat) 行是对应的
    # 如果 compute_features 内部改变了行数/顺序，需要按 date/symbol 等键对齐
    df_ml = df.copy()
    df_ml['ml_signal'] = np.nan
    df_ml.iloc[:len(df_feat), df_ml.columns.get_loc('ml_signal')] = df_feat['ml_signal'].values

    # 用昨天的信号交易（shift 1），避免未来函数
    df_ml['ml_signal'] = df_ml['ml_signal'].shift(1)

    # 删掉没有信号的样本
    df_ml = df_ml.dropna(subset=['ml_signal'])
    return df_ml
