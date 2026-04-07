# model_mlp_baseline.py
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from feature_engineering import prepare_dataset


def train_mlp():
    df = prepare_dataset()
    feature_cols = [c for c in df.columns if c not in ['date', 'symbol', 'label', 'fwd_ret_5']]

    X = df[feature_cols].values
    y = (df['label'] > 0.7).astype(int)  # 选前30%作为正样本

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, early_stopping=True)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"MLP Test Accuracy: {score:.4f}")


if __name__ == "__main__":
    train_mlp()
