# ablation_features.py
import pandas as pd
from model_mlp_baseline import train_mlp
from model_gbdt_baseline import train_gbdt
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from config import cfg

def train_one_config(conf: dict):
    """每个配置的训练逻辑封装成单独函数，便于并行"""
    name = conf["name"]
    flags = conf["flags"]
    start_time = time.time()

    try:
        _, _, mlp_acc, _ = train_mlp(flags=flags)
    except Exception as e:
        mlp_acc = None
        print(f"[MLP] Config {name} failed: {e}")

    try:
        _, _, gbdt_acc, _ = train_gbdt(flags=flags)
    except Exception as e:
        gbdt_acc = None
        print(f"[GBDT] Config {name} failed: {e}")

    duration = time.time() - start_time
    return {
        "config": name,
        "mlp_acc": mlp_acc,
        "gbdt_acc": gbdt_acc,
        "time_sec": round(duration, 2),
    }

def run_ablation(max_workers: int = 2):
    configs = [
        {"name": "all_factors", "flags": {"ma": True, "rsi": True, "macd": True, "volatility": True}},
        {"name": "no_ma", "flags": {"ma": False, "rsi": True, "macd": True, "volatility": True}},
        {"name": "no_macd", "flags": {"ma": True, "rsi": True, "macd": False, "volatility": True}},
        {"name": "only_price", "flags": {"ma": False, "rsi": False, "macd": False, "volatility": False}},
    ]

    print(f"{'=' * 20} Starting Ablation Study (parallel, max_workers={max_workers}) {'=' * 20}")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(train_one_config, conf): conf for conf in configs}
        for future in as_completed(futures):
            conf = futures[future]
            try:
                res = future.result()
                results.append(res)
                print(f"Finished: {res['config']}")
            except Exception as e:
                print(f"Config {conf['name']} raised: {e}")

    df_results = pd.DataFrame(results)
    print("\n" + "=" * 20 + " Ablation Results " + "=" * 20)
    print(df_results)
    df_results.to_csv(cfg.data_dir / "ablation_results.csv", index=False)
    return df_results

if __name__ == "__main__":
    run_ablation()
