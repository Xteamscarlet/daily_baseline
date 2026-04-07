# model_gbdt_baseline.py
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import prepare_dataset
from model_mlp_baseline import StandardScaler  # 复用 scaler 逻辑
from config import cfg


def train_gbdt(flags: dict = None):
    df = prepare_dataset(flags=flags)
    if df.empty: return None, None, 0, []

    feature_cols = [c for c in df.columns if c not in
                    ['date', 'symbol', 'label', 'fwd_ret_5', 'amplitude', 'pct_chg', 'change', 'turnover']]

    X = df[feature_cols].values
    y = (df['label'] > 0.7).astype(int)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_train_s, y_train)

    acc = model.score(X_test_s, y_test)

    joblib.dump(model, cfg.model_dir / "gbdt_model.pkl")
    joblib.dump(scaler, cfg.model_dir / "gbdt_scaler.pkl")
    joblib.dump(feature_cols, cfg.model_dir / "gbdt_features.pkl")

    return model, scaler, acc, feature_cols
