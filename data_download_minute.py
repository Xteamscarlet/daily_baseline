# data_download_minute.py
import pandas as pd
import akshare as ak
import os
import config


def download_minute_data(symbol: str, period="5"):
    """下载分钟线数据"""
    file_path = config.DATA_DIR / f"minute_{symbol}.parquet"
    if file_path.exists():
        print(f"[Minute] Cache exists for {symbol}")
        return pd.read_parquet(file_path)

    print(f"[Minute] Downloading {symbol} {period}min...")
    try:
        # Akshare 分钟线需要带市场前缀
        market_symbol = f"sh{symbol}" if symbol.startswith("6") else f"sz{symbol}"
        df = ak.stock_zh_a_minute(symbol=market_symbol, period=period, adjust="")

        # 重命名
        df.columns = ['datetime', 'open', 'close', 'high', 'low', 'volume', 'amount']
        df['symbol'] = symbol

        # 简单质量检查
        check_data_quality(df)

        df.to_parquet(file_path, index=False)
        return df
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()


def check_data_quality(df):
    # 检查时间间隔
    dt = pd.to_datetime(df['datetime'])
    gaps = dt.diff().dropna().value_counts()
    print(f"Time gaps distribution:\n{gaps.head()}")


if __name__ == "__main__":
    download_minute_data("600519")
