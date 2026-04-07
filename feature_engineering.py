# feature_engineering.py
import pandas as pd
import numpy as np
from data_download_daily import load_or_download_daily
from config import cfg
from typing import Dict

# 默认配置字典
DEFAULT_FLAGS = {"ma": True, "rsi": True, "macd": True, "volatility": True}


def compute_features(df: pd.DataFrame, flags: Dict = None) -> pd.DataFrame:
    if flags is None: flags = DEFAULT_FLAGS
    df = df.copy()

    if flags.get('ma', True):
        for w in [5, 10, 20]: df[f'ma_{w}'] = df['close'].rolling(w).mean()

    if flags.get('rsi', True):
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))

    if flags.get('macd', True):
        exp12 = df['close'].ewm(span=12).mean()
        exp26 = df['close'].ewm(span=26).mean()
        df['macd_dif'] = exp12 - exp26
        df['macd_dea'] = df['macd_dif'].ewm(span=9).mean()

    if flags.get('volatility', True):
        df['volatility_20'] = df['close'].pct_change().rolling(20).std()

    return df


def prepare_dataset(flags: Dict = None):
    """[重构] 接受 flags 参数，真正控制特征生成"""
    all_dfs = []
    for sym in cfg.symbol_pool:
        df = load_or_download_daily(sym)
        if df.empty: continue

        # 核心：将 flags 传递进去
        df = compute_features(df, flags=flags)

        df['symbol'] = sym
        df['fwd_ret_5'] = df.groupby('symbol')['close'].shift(-5) / df['close'] - 1
        df['label'] = df.groupby('date')['fwd_ret_5'].transform(
            lambda x: x.rank(pct=True) if len(x) > 1 else 0.5
        )
        all_dfs.append(df)

    if not all_dfs: return pd.DataFrame()
    full_df = pd.concat(all_dfs).dropna()
    return full_df
