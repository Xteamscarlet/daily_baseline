# run_all.py
import argparse
from config import cfg
from data_download_daily import load_or_download_daily
from ablation_features import run_ablation
from model_mlp_baseline import train_mlp
from backtest_baseline_strategy import run_backtest, MaCrossStrategy


def main():
    print("Step 1: Downloading Data...")
    for sym in cfg.symbol_pool:
        load_or_download_daily(sym)

    print("\nStep 2: Running Ablation Study (This may take a while)...")
    ablation_res = run_ablation()

    print("\nStep 3: Training Final MLP Model...")
    train_mlp()  # 训练一个全特征模型用于演示

    print("\nStep 4: Running Backtest Demo...")
    # 随机选一只股票演示策略
    run_backtest(cfg.symbol_pool[0], MaCrossStrategy, plot=True)


if __name__ == "__main__":
    main()
