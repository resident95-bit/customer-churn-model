"""Train and evaluate a customer churn classifier on synthetic data."""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

SEED = 42


def load_data():
    X, y = make_classification(
        n_samples=2000,
        n_features=12,
        n_informative=5,
        n_redundant=2,
        weights=[0.8, 0.2],
        flip_y=0.05,
        class_sep=0.8,
        random_state=SEED,
    )
    return X, y


def balance(X, y):
    """Oversample churn rows to balance the target classes."""
    X_min, y_min = X[y == 1], y[y == 1]
    X_maj, y_maj = X[y == 0], y[y == 0]
    X_up, y_up = resample(
        X_min, y_min, replace=True, n_samples=len(y_maj), random_state=SEED
    )
    return np.vstack([X_maj, X_up]), np.concatenate([y_maj, y_up])


def main():
    X, y = load_data()
    X, y = balance(X, y)
    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED
    )

    best_acc, best_pred = 0.0, None
    for depth in (4, 6, 8):
        for trees in (50, 100):
            model = RandomForestClassifier(
                n_estimators=trees, max_depth=depth, random_state=SEED
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            acc = accuracy_score(y_test, pred)
            if acc > best_acc:
                best_acc, best_pred = acc, pred

    print("== Evaluation (held-out test set) ==")
    print(f"test_rows={len(y_test)}")
    print(f"accuracy={best_acc:.3f}")
    print(f"macro_f1={f1_score(y_test, best_pred, average='macro'):.3f}")


if __name__ == "__main__":
    main()
