# data_download_daily.py
import os
import time

import pandas as pd
import akshare as ak
from config import cfg
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    before_sleep=before_sleep_log(logger, logging.INFO),
)
def _download_with_retry(symbol: str) -> pd.DataFrame:
    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=cfg.start_date,
        adjust="qfq",
    )
    df = df.rename(columns={
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume",
        "成交额": "amount", "振幅": "amplitude", "涨跌幅": "pct_chg",
        "涨跌额": "change", "换手率": "turnover"
    })
    df["symbol"] = symbol
    return df

def load_or_download_daily(symbol: str) -> pd.DataFrame:
    file_path = cfg.data_dir / f"daily_{symbol}.parquet"
    if file_path.exists():
        return pd.read_parquet(file_path)

    logger.info("[Daily] Downloading %s...", symbol)
    try:
        df = _download_with_retry(symbol)
        time.sleep(5)
        df.to_parquet(file_path, index=False)
        return df
    except Exception as e:
        logger.exception("[Daily] Download failed for %s after retries", symbol)
        return pd.DataFrame()
