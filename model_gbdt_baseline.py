# model_gbdt_baseline.py
from sklearn.ensemble import GradientBoostingClassifier
from model_mlp_baseline import train_mlp, prepare_dataset, StandardScaler, train_test_split


def train_gbdt():
    df = prepare_dataset()
    feature_cols = [c for c in df.columns if c not in ['date', 'symbol', 'label', 'fwd_ret_5']]

    X = df[feature_cols].values
    y = (df['label'] > 0.7).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = GradientBoostingClassifier(n_estimators=100, max_depth=4)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"GBDT Test Accuracy: {score:.4f}")


if __name__ == "__main__":
    train_gbdt()
