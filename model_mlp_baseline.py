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
    feature_cols = [c for c in df.columns if c not in
                    ['date', 'symbol', 'label', 'fwd_ret_5', 'amplitude', 'pct_chg', 'change', 'turnover']]

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
