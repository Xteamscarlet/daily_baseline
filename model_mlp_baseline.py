# model_mlp_baseline.py
import pandas as pd
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from feature_engineering import prepare_dataset
from config import cfg


def train_mlp(flags: dict = None):
    """
    训练 MLP 模型
    :param flags: 因子配置
    :return: model, scaler, accuracy
    """
    print("[MLP] Preparing dataset...")
    df = prepare_dataset(flags=flags)

    # 特征列排除非数值列
    feature_cols = [c for c in df.columns if
                    c not in ['date', 'symbol', 'label', 'fwd_ret_5', 'amplitude', 'pct_chg', 'change', 'turnover']]

    X = df[feature_cols].values
    y = (df['label'] > 0.7).astype(int)  # 选前30%作为正样本

    # 为了回测时的一致性，我们只按时间切分，不用随机切分
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"[MLP] Training on {X_train.shape[0]} samples...")
    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, early_stopping=True, random_state=42)
    model.fit(X_train_scaled, y_train)

    acc = model.score(X_test_scaled, y_test)
    print(f"[MLP] Test Accuracy: {acc:.4f}")

    # 保存模型和 Scaler
    model_path = cfg.model_dir / "mlp_model.pkl"
    scaler_path = cfg.model_dir / "mlp_scaler.pkl"
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"[MLP] Model saved to {model_path}")

    return model, scaler, acc, feature_cols
