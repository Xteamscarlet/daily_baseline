# run_all.py
import logging
from config import cfg
from data_download_daily import load_or_download_daily
from ablation_features import run_ablation
from model_mlp_baseline import train_mlp
from backtest_baseline_strategy import run_backtest, MaCrossStrategy, MLStrategy

# 1) 配置日志（简单版：只打印到标准输出；项目可后续改文件日志）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    # Step 1: 下载数据
    logger.info("Step 1: Downloading Data...")
    try:
        for sym in cfg.symbol_pool:
            load_or_download_daily(sym)
    except Exception:
        logger.exception("Step 1 failed while downloading data")
        # 可以选择 return / raise，这里按 Grok 风格继续执行后续步骤

    # Step 2: Ablation 实验
    logger.info("Step 2: Running Ablation Study...")
    try:
        run_ablation()  # 这一步会保存 ablation_results.csv 到 data 目录
    except Exception:
        logger.exception("Step 2 failed during ablation study")

    # Step 3: 训练最终 MLP 模型
    logger.info("Step 3: Training Final MLP Model (All Factors)...")
    try:
        train_mlp()  # 默认 flags=None（使用全部因子）
    except Exception:
        logger.exception("Step 3 failed while training MLP")

    # Step 4: 回测（遍历所有 symbol）
    logger.info("Step 4: Running Backtests...")
    for sym in cfg.symbol_pool:
        try:
            # 规则策略
            run_backtest(sym, MaCrossStrategy, plot=False)
        except Exception:
            logger.exception("MaCrossStrategy backtest failed for %s", sym)
            # 单只股票失败，继续下一只

        try:
            # ML 策略（加载刚才训练好的模型）
            run_backtest(sym, MLStrategy, model_name="mlp", plot=True)
        except Exception:
            logger.exception("MLStrategy backtest failed for %s", sym)

if __name__ == "__main__":
    main()
