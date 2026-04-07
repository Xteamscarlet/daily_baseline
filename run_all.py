# run_all.py
from config import cfg
from data_download_daily import load_or_download_daily
from ablation_features import run_ablation
from model_mlp_baseline import train_mlp
from backtest_baseline_strategy import run_backtest, MaCrossStrategy, MLStrategy


def main():
    print("Step 1: Downloading Data...")
    for sym in cfg.symbol_pool:
        load_or_download_daily(sym)

    print("\nStep 2: Running Ablation Study...")
    # 这一步会保存 ablation_results.csv 到 data 目录
    run_ablation()

    print("\nStep 3: Training Final MLP Model (All Factors)...")
    # 训练一个最终版本用于回测
    train_mlp()

    print("\nStep 4: Running Backtests...")
    # 1. 规则策略对比
    run_backtest(cfg.symbol_pool[0], MaCrossStrategy, plot=False)

    # 2. ML 策略回测 (加载刚才训练好的模型)
    run_backtest(cfg.symbol_pool[0], MLStrategy, model_name='mlp', plot=True)


if __name__ == "__main__":
    main()
