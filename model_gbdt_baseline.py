# model_gbdt_baseline.py
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from model_mlp_baseline import prepare_dataset, StandardScaler, cfg


def train_gbdt(flags: dict = None):
    print("[GBDT] Preparing dataset...")
    df = prepare_dataset(flags=flags)
    feature_cols = [c for c in df.columns if
                    c not in ['date', 'symbol', 'label', 'fwd_ret_5', 'amplitude', 'pct_chg', 'change', 'turnover']]

    X = df[feature_cols].values
    y = (df['label'] > 0.7).astype(int)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"[GBDT] Training on {X_train.shape[0]} samples...")
    model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_train_scaled, y_train)

    acc = model.score(X_test_scaled, y_test)
    print(f"[GBDT] Test Accuracy: {acc:.4f}")

    model_path = cfg.model_dir / "gbdt_model.pkl"
    scaler_path = cfg.model_dir / "gbdt_scaler.pkl"
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    return model, scaler, acc, feature_cols
