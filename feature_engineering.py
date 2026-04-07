# feature_engineering.py
import pandas as pd
import numpy as np
from data_download_daily import load_or_download_daily
from config import cfg
from typing import Dict

# 默认配置
DEFAULT_FLAGS = {
    "ma": True,
    "rsi": True,
    "macd": True,
    "volatility": True
}


def compute_features(df: pd.DataFrame, flags: Dict = None) -> pd.DataFrame:
    """
    计算技术指标因子。
    :param flags: dict, 控制开启哪些因子
    """
    if flags is None:
        flags = DEFAULT_FLAGS

    df = df.copy()

    # MA因子
    if flags.get('ma', True):
        for w in [5, 10, 20]:
            df[f'ma_{w}'] = df['close'].rolling(w).mean()

    # RSI
    if flags.get('rsi', True):
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    if flags.get('macd', True):
        exp12 = df['close'].ewm(span=12).mean()
        exp26 = df['close'].ewm(span=26).mean()
        df['macd_dif'] = exp12 - exp26
        df['macd_dea'] = df['macd_dif'].ewm(span=9).mean()

    # 波动率
    if flags.get('volatility', True):
        df['volatility_20'] = df['close'].pct_change().rolling(20).std()

    return df


def prepare_dataset(flags: Dict = None):
    """
    准备全量数据集
    :param flags: 传递给 compute_features
    """
    all_dfs = []
    for sym in cfg.symbol_pool:
        df = load_or_download_daily(sym)
        if df.empty: continue
        df = compute_features(df, flags=flags)
        df['symbol'] = sym
        # 构造标签：未来5日收益排名
        df['fwd_ret_5'] = df.groupby('symbol')['close'].shift(-5) / df['close'] - 1
        # 只有在有多只股票时，截面排名才有意义；单只股票则是时间序列排名
        # 这里为了通用性，简单处理：
        df['label'] = df.groupby('date')['fwd_ret_5'].transform(
            lambda x: x.rank(pct=True) if len(x) > 1 else (1 if x.iloc[0] > 0 else 0)
        )
        all_dfs.append(df)

    full_df = pd.concat(all_dfs).dropna()
    return full_df
