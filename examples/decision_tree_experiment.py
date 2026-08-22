import time
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from src.decision_tree import DecisionTreeClassifier


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy: fraction of correct predictions."""
    return np.mean(y_true == y_pred)


def main():
    # 1. Load dataset
    data = load_iris()
    X, y = data.data, data.target

    # 2. Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Dataset: Iris")
    print(f"Total samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print()

    # 3. Train custom Decision Tree
    tree = DecisionTreeClassifier(max_depth=10, min_samples_split=2)

    start_train = time.perf_counter()
    tree.fit(X_train, y_train)
    train_time = time.perf_counter() - start_train

    # 4. Predict
    start_predict = time.perf_counter()
    predictions = tree.predict(X_test)
    predict_time = time.perf_counter() - start_predict

    # 5. Report actual results
    acc = accuracy_score(y_test, predictions)

    print("=== Custom Decision Tree Results ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Training time: {train_time:.6f} seconds")
    print(f"Prediction time: {predict_time:.6f} seconds")
    print()
    print("Sample predictions vs actual:")
    for i in range(5):
        print(f"  Predicted: {predictions[i]}, Actual: {y_test[i]}")


if __name__ == "__main__":
    main()