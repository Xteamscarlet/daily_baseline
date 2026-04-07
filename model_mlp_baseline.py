# model_mlp_baseline.py
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
              n_classes: int = 2,          # 【改】默认二分类
              random_state: int = 42):
    # 1. 数据准备
    df = prepare_dataset(flags=flags)
    if df.empty:
        return None, None, 0, []

    # 2. 特征选择
    exclude_cols = [
        'date', 'symbol', 'label', 'fwd_ret_5',
        'amplitude', 'pct_chg', 'change',
        'turnover', 'open', 'high', 'low', 'close', 'volume', 'amount'
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    if not feature_cols:
        print(f"[MLP] Warning: No features available for flags={flags}. Skipping.")
        return None, None, 0, []

    # 确保所有特征列是数值类型
    df = df.copy()
    for col in feature_cols:
        if pd.api.types.is_object_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    X = df[feature_cols].values
    y_raw = df['label'].values

    # 3. 标签构造
    if n_classes == 2:
        # 【改】二分类：label > 0.5 为正（横截面排名前50%），否则为负
        y = (y_raw > 0.5).astype(int)
        print(f"[MLP] Binary label: positive={y.sum()}, negative={len(y)-y.sum()}, "
              f"ratio={y.mean():.3f}")
    else:
        y = np.zeros_like(y_raw, dtype=int)
        y[y_raw <= 0.33] = 0
        y[(y_raw > 0.33) & (y_raw <= 0.67)] = 1
        y[y_raw > 0.67] = 2

    # 4. 时间序列切分（80/20）
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 兼容 object dtype
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)

    mask_train = ~np.isnan(X_train).any(axis=1)
    X_train = X_train[mask_train]
    y_train = y_train[mask_train]

    mask_test = ~np.isnan(X_test).any(axis=1)
    X_test = X_test[mask_test]
    y_test = y_test[mask_test]

    # 5. 标准化
    if use_quantile_transform:
        scaler = QuantileTransformer(output_distribution='normal',
                                     random_state=random_state)
    else:
        scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 6. 训练 MLP —— 【改】更大网络 + 更好正则化
    model = MLPClassifier(
        hidden_layer_sizes=cfg.mlp_hidden_layers if hasattr(cfg, 'mlp_hidden_layers') else (256, 128, 64),
        max_iter=cfg.mlp_max_iter if hasattr(cfg, 'mlp_max_iter') else 500,
        alpha=0.0001,                  # 【改】L2 正则化（默认0.0001，更强）
        learning_rate_init=0.001,      # 【改】降低初始学习率，更稳定
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,           # 【改】耐心值加大
        random_state=cfg.random_state if hasattr(cfg, "random_state") else 42,
    )
    model.fit(X_train_s, y_train)

    # 7. 评估
    y_pred = model.predict(X_test_s)
    print("\nClassification report (MLP):")
    print(classification_report(y_test, y_pred, zero_division=0))

    acc = model.score(X_test_s, y_test)
    print(f"Accuracy: {acc:.4f}")

    try:
        y_score = model.predict_proba(X_test_s)[:, 1] if n_classes == 2 else model.predict_proba(X_test_s)
        if n_classes == 2:
            auc = roc_auc_score(y_test, y_score)
            print(f"ROC-AUC: {auc:.4f}")
        else:
            auc = roc_auc_score(y_test, y_score, multi_class='ovr')
            print(f"ROC-AUC (ovr): {auc:.4f}")
    except Exception as e:
        print("AUC failed:", e)

    # 8. 保存
    joblib.dump(model, cfg.model_dir / "mlp_model.pkl")
    joblib.dump(scaler, cfg.model_dir / "mlp_scaler.pkl")
    joblib.dump(feature_cols, cfg.model_dir / "mlp_features.pkl")
    joblib.dump(n_classes, cfg.model_dir / "mlp_n_classes.pkl")  # 【改】保存类别数供回测使用
    return model, scaler, acc, feature_cols
