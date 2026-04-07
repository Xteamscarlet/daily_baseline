# data_download_daily.py
import pandas as pd
import akshare as ak
from pathlib import Path
import config  # 导入配置


def load_or_download_daily(symbol: str) -> pd.DataFrame:
    """下载或加载日线数据 (Parquet缓存优先)"""
    file_path = config.DATA_DIR / f"daily_{symbol}.parquet"

    # 1. 如果存在缓存，直接读取
    if file_path.exists():
        print(f"[Daily] Loading cache for {symbol}...")
        return pd.read_parquet(file_path)

    # 2. 否则下载
    print(f"[Daily] Downloading {symbol} via Akshare (Proxy {'On' if 'HTTP_PROXY' in os.environ else 'Off'})...")
    try:
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=config.START_DATE,
            end_date=config.END_DATE,
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
    import os

    for sym in config.SYMBOL_POOL:
        load_or_download_daily(sym)
