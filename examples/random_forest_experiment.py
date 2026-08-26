import time
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from src.random_forest import RandomForestClassifier


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(y_true == y_pred)


def main():
    data = load_iris()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Dataset: Iris")
    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print()

    estimator_values = [5, 10, 50]
    results = []

    for n_estimators in estimator_values:
        forest = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

        start_train = time.perf_counter()
        forest.fit(X_train, y_train)
        train_time = time.perf_counter() - start_train

        start_predict = time.perf_counter()
        predictions = forest.predict(X_test)
        predict_time = time.perf_counter() - start_predict

        acc = accuracy_score(y_test, predictions)
        results.append((n_estimators, acc, train_time, predict_time))

        print(f"=== n_estimators = {n_estimators} ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"Training time: {train_time:.6f} seconds")
        print(f"Prediction time: {predict_time:.6f} seconds")
        print()

    print("=== Summary Table ===")
    print(f"{'n_estimators':<15}{'Accuracy':<12}{'Train Time (s)':<18}{'Predict Time (s)'}")
    for n_estimators, acc, train_time, predict_time in results:
        print(f"{n_estimators:<15}{acc:<12.4f}{train_time:<18.6f}{predict_time:.6f}")


if __name__ == "__main__":
    main()