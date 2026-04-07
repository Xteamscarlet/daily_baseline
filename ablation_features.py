# ablation_features.py
import pandas as pd
from model_mlp_baseline import train_mlp
from model_gbdt_baseline import train_gbdt
import time


def run_ablation():
    results = []

    # 定义消融配置
    configs = [
        {"name": "all_factors", "flags": {"ma": True, "rsi": True, "macd": True, "volatility": True}},
        {"name": "no_ma", "flags": {"ma": False, "rsi": True, "macd": True, "volatility": True}},
        {"name": "no_macd", "flags": {"ma": True, "rsi": True, "macd": False, "volatility": True}},
        {"name": "only_price", "flags": {"ma": False, "rsi": False, "macd": False, "volatility": False}}
    ]

    print(f"{'=' * 20} Starting Ablation Study {'=' * 20}")

    for conf in configs:
        print(f"\n>>> Running config: {conf['name']}")
        start_time = time.time()

        # 1. 训练 MLP
        _, _, mlp_acc, _ = train_mlp(flags=conf['flags'])

        # 2. 训练 GBDT
        _, _, gbdt_acc, _ = train_gbdt(flags=conf['flags'])

        duration = time.time() - start_time

        results.append({
            "config": conf['name'],
            "mlp_acc": mlp_acc,
            "gbdt_acc": gbdt_acc,
            "time_sec": round(duration, 2)
        })

    df_results = pd.DataFrame(results)
    print("\n" + "=" * 20 + " Ablation Results " + "=" * 20)
    print(df_results)

    # 保存结果
    df_results.to_csv(cfg.data_dir / "ablation_results.csv", index=False)
    return df_results


if __name__ == "__main__":
    from config import cfg  # 确保加载配置

    run_ablation()
