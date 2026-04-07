# model_mlp_baseline.py
import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from feature_engineering import prepare_dataset
from config import cfg


def train_mlp(flags: dict = None):
    # 1. 带着 flags 生成数据
    df = prepare_dataset(flags=flags)
    if df.empty: return None, None, 0, []

    # 2. 特征工程
    # [关键修复] 明确要排除的列（原始价格、元数据、标签）
    # 强烈建议不要把 raw_ohlcv 放入 MLP，除非你做了极其复杂的归一化
    exclude_cols = [
        'date', 'symbol', 'label', 'fwd_ret_5',
        'amplitude', 'pct_chg', 'change', 'turnover',  # AKShare 原始衍生
        'open', 'high', 'low', 'close', 'volume', 'amount'  # [重要] 剔除原始价格
    ]

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    # 增加 only_price 的安全检查
    if not feature_cols:
        print(f"[MLP] Warning: No features available for flags={flags}. Skipping training.")
        return None, None, 0, []

    X = df[feature_cols].values
    y = (df['label'] > 0.7).astype(int)

    # 3. 切分
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 4. 训练
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, early_stopping=True, random_state=42)
    model.fit(X_train_s, y_train)

    acc = model.score(X_test_s, y_test)

    # 5. 保存
    joblib.dump(model, cfg.model_dir / "mlp_model.pkl")
    joblib.dump(scaler, cfg.model_dir / "mlp_scaler.pkl")
    # 保存特征列名，保证回测时特征顺序一致
    joblib.dump(feature_cols, cfg.model_dir / "mlp_features.pkl")

    return model, scaler, acc, feature_cols
