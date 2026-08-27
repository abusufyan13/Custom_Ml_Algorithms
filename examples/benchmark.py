import time
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForest
from sklearn.metrics import accuracy_score, precision_score, recall_score

from src.decision_tree import DecisionTreeClassifier as CustomDecisionTree
from src.random_forest import RandomForestClassifier as CustomRandomForest


def evaluate(name, model, X_train, y_train, X_test, y_test):
    start_train = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start_train

    start_predict = time.perf_counter()
    predictions = model.predict(X_test)
    predict_time = time.perf_counter() - start_predict

    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions)
    rec = recall_score(y_test, predictions)

    print(f"=== {name} ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"Training time:   {train_time:.6f} seconds")
    print(f"Prediction time: {predict_time:.6f} seconds")
    print()

    return {
        "name": name, "accuracy": acc, "precision": prec, "recall": rec,
        "train_time": train_time, "predict_time": predict_time,
    }


def main():
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Dataset: Breast Cancer")
    print(f"Total samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print()

    results = []

    # --- Decision Tree comparison ---
    results.append(evaluate(
        "Custom Decision Tree",
        CustomDecisionTree(max_depth=10),
        X_train, y_train, X_test, y_test,
    ))
    results.append(evaluate(
        "Sklearn Decision Tree",
        SklearnDecisionTree(max_depth=10, random_state=42),
        X_train, y_train, X_test, y_test,
    ))

    # --- Random Forest comparison ---
    results.append(evaluate(
        "Custom Random Forest (n=10)",
        CustomRandomForest(n_estimators=10, max_depth=10, random_state=42),
        X_train, y_train, X_test, y_test,
    ))
    results.append(evaluate(
        "Sklearn Random Forest (n=10)",
        SklearnRandomForest(n_estimators=10, max_depth=10, random_state=42),
        X_train, y_train, X_test, y_test,
    ))

    # --- Summary table ---
    print("=== Summary Table ===")
    header = f"{'Model':<32}{'Accuracy':<11}{'Precision':<11}{'Recall':<10}{'Train (s)':<12}{'Predict (s)'}"
    print(header)
    for r in results:
        print(f"{r['name']:<32}{r['accuracy']:<11.4f}{r['precision']:<11.4f}"
              f"{r['recall']:<10.4f}{r['train_time']:<12.6f}{r['predict_time']:.6f}")


if __name__ == "__main__":
    main()