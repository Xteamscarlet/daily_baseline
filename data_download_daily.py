# data_download_daily.py
import os  # [Fix] 必须在文件最开头
import pandas as pd
import akshare as ak
from config import cfg
from pathlib import Path


def load_or_download_daily(symbol: str) -> pd.DataFrame:
    """下载或加载日线数据 (Parquet缓存优先)"""
    file_path = cfg.data_dir / f"daily_{symbol}.parquet"

    # 1. 如果存在缓存，直接读取
    if file_path.exists():
        # print(f"[Daily] Loading cache for {symbol}...")
        return pd.read_parquet(file_path)

    # 2. 否则下载
    print(f"[Daily] Downloading {symbol} via Akshare...")
    try:
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            adjust="qfq"
        )
        # 重命名列
        df = df.rename(columns={
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume",
            "成交额": "amount", "振幅": "amplitude", "涨跌幅": "pct_chg",
            "涨跌额": "change", "换手率": "turnover"
        })
        df["symbol"] = symbol
        df.to_parquet(file_path, index=False)
        return df
    except Exception as e:
        print(f"Error downloading {symbol}: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    for sym in cfg.symbol_pool:
        load_or_download_daily(sym)
