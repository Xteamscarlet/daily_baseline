# model_mlp_baseline.py（标签部分改法示例）
import joblib
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import classification_report, roc_auc_score
from feature_engineering import prepare_dataset
from config import cfg

def train_mlp(flags: dict = None,
              use_quantile_transform: bool = False,
              n_classes: int = 3,
              random_state: int = 42):
    # 1. 数据准备
    df = prepare_dataset(flags=flags)
    if df.empty:
        return None, None, 0, []

    # 2. 特征选择（保持原有逻辑）
    exclude_cols = [
        'date', 'symbol', 'label', 'fwd_ret_5', 'amplitude', 'pct_chg', 'change',
        'turnover', 'open', 'high', 'low', 'close', 'volume', 'amount'
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    if not feature_cols:
        print(f"[MLP] Warning: No features available for flags={flags}. Skipping training.")
        return None, None, 0, []

    X = df[feature_cols].values
    y_raw = df['label'].values  # 当前 label 是 rank 分位数 0~1

    # 3. 标签构造：3 分类
    if n_classes == 3:
        # 方案1：按分位数切分（避免 0.7 武断阈值）
        # 如果 label 已经是横截面分位数，可以直接按 0~0.33 / 0.33~0.67 / 0.67~1 划分
        y = np.zeros_like(y_raw, dtype=int)
        y[y_raw <= 0.33] = 0        # 弱
        y[(y_raw > 0.33) & (y_raw <= 0.67)] = 1  # 中
        y[y_raw > 0.67] = 2        # 强
    else:
        # 保留二分类，但阈值可配（比如通过 cfg 或参数传入）
        threshold = cfg.label_threshold_positive if hasattr(cfg, "label_threshold_positive") else 0.7
        y = (y_raw > threshold).astype(int)

    # 4. 时间序列切分
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    mask_train = ~np.isnan(X_train).any(axis=1)
    X_train = X_train[mask_train]
    y_train = y_train[mask_train]

    mask_test = ~np.isnan(X_test).any(axis=1)
    X_test = X_test[mask_test]
    y_test = y_test[mask_test]
    # 5. 标准化（可选 QuantileTransformer）
    if use_quantile_transform:
        scaler = QuantileTransformer(output_distribution='normal',
                                     random_state=random_state)
    else:
        scaler = StandardScaler()

    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 6. 训练 MLP（超参建议从 cfg 读取，见下一节）
    model = MLPClassifier(
        hidden_layer_sizes=cfg.mlp_hidden_layers,
        max_iter=cfg.mlp_max_iter,
        early_stopping=True,
        random_state=cfg.random_state if hasattr(cfg, "random_state") else 42,
    )
    model.fit(X_train_s, y_train)

    # 7. 指标：多分类 / 二分类通用
    y_pred = model.predict(X_test_s)
    print("\nClassification report (MLP):")
    print(classification_report(y_test, y_pred))

    if n_classes == 2:
        try:
            y_score = model.predict_proba(X_test_s)[:, 1]
            auc = roc_auc_score(y_test, y_score)
            print(f"ROC-AUC (binary): {auc:.4f}")
        except Exception as e:
            print("ROC-AUC calculation failed:", e)
    else:
        # 多分类 AUC（one-vs-rest）
        try:
            y_score = model.predict_proba(X_test_s)
            auc = roc_auc_score(y_test, y_score, multi_class='ovr')
            print(f"ROC-AUC (multi-class, ovr): {auc:.4f}")
        except Exception as e:
            print("ROC-AUC (multi-class) calculation failed:", e)

    acc = model.score(X_test_s, y_test)
    print(f"Accuracy: {acc:.4f}")

    # 8. 保存
    joblib.dump(model, cfg.model_dir / "mlp_model.pkl")
    joblib.dump(scaler, cfg.model_dir / "mlp_scaler.pkl")
    joblib.dump(feature_cols, cfg.model_dir / "mlp_features.pkl")

    return model, scaler, acc, feature_cols
