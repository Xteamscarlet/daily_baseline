# feature_engineering.py
import pandas as pd
import numpy as np
from data_download_daily import load_or_download_daily
from config import cfg
from typing import Dict

# 默认配置字典
DEFAULT_FLAGS = {"ma": True, "rsi": True, "macd": True, "volatility": True}
# feature_engineering.py（补充部分）
import hashlib
import json
from pathlib import Path

def _flags_to_key(flags: dict) -> str:
    """简单把 flags 转成字符串 key，用于缓存文件名"""
    return hashlib.md5(json.dumps(flags, sort_keys=True).encode()).hexdigest()

def get_feature_path(symbol: str, flags: dict = None) -> Path:
    key = _flags_to_key(flags or DEFAULT_FLAGS)
    return cfg.data_dir / f"daily_{symbol}_features_{key}.parquet"

def compute_and_cache_features(df: pd.DataFrame, symbol: str, flags: dict = None) -> pd.DataFrame:
    path = get_feature_path(symbol, flags)
    if path.exists():
        return pd.read_parquet(path)
    df_feat = compute_features(df, flags=flags)
    df_feat.to_parquet(path, index=False)
    return df_feat

def prepare_dataset(flags: dict = None):
    all_dfs = []
    for sym in cfg.symbol_pool:
        df = load_or_download_daily(sym)
        if df.empty:
            continue
        # 只计算/读取特征，不做 label
        df_feat = compute_and_cache_features(df, sym, flags=flags)
        df_feat['symbol'] = sym
        all_dfs.append(df_feat)

    if not all_dfs:
        return pd.DataFrame()
    full_df = pd.concat(all_dfs).dropna()

    # label 部分保持原逻辑
    full_df['fwd_ret_5'] = full_df.groupby('symbol')['close'].shift(-5) / full_df['close'] - 1
    full_df['label'] = full_df.groupby('date')['fwd_ret_5'].transform(
        lambda x: x.rank(pct=True) if len(x) > 1 else 0.5
    )
    return full_df


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

