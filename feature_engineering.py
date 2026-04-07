# feature_engineering.py
import pandas as pd
import numpy as np
from data_download_daily import load_or_download_daily
import config

# 因子开关配置
FEATURE_FLAGS = {
    "ma": True,
    "rsi": True,
    "macd": True,
    "volatility": True
}


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算所有技术指标因子"""
    df = df.copy()

    # MA因子
    if FEATURE_FLAGS['ma']:
        for w in [5, 10, 20]:
            df[f'ma_{w}'] = df['close'].rolling(w).mean()

    # RSI
    if FEATURE_FLAGS['rsi']:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    if FEATURE_FLAGS['macd']:
        exp12 = df['close'].ewm(span=12).mean()
        exp26 = df['close'].ewm(span=26).mean()
        df['macd_dif'] = exp12 - exp26
        df['macd_dea'] = df['macd_dif'].ewm(span=9).mean()

    # 波动率
    if FEATURE_FLAGS['volatility']:
        df['volatility_20'] = df['close'].pct_change().rolling(20).std()

    return df


def prepare_dataset():
    """准备全量数据集"""
    all_dfs = []
    for sym in config.SYMBOL_POOL:
        df = load_or_download_daily(sym)
        if df.empty: continue
        df = compute_features(df)
        df['symbol'] = sym
        # 构造标签：未来5日收益排名
        df['fwd_ret_5'] = df.groupby('symbol')['close'].shift(-5) / df['close'] - 1
        df['label'] = df.groupby('date')['fwd_ret_5'].transform(lambda x: x.rank(pct=True))
        all_dfs.append(df)

    full_df = pd.concat(all_dfs).dropna()
    return full_df


if __name__ == "__main__":
    df = prepare_dataset()
    print(df.head())
