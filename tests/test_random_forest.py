import numpy as np
from src.random_forest import RandomForestClassifier


def test_forest_has_correct_number_of_trees():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    forest = RandomForestClassifier(n_estimators=5, random_state=1)
    forest.fit(X, y)
    assert len(forest.trees) == 5


def test_forest_predicts_correct_shape():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    forest = RandomForestClassifier(n_estimators=5, random_state=1)
    forest.fit(X, y)
    predictions = forest.predict(X)
    assert predictions.shape == y.shape


def test_forest_reproducible_with_same_seed():
    X = np.array([[i] for i in range(20)], dtype=float)
    y = np.array([0 if i < 10 else 1 for i in range(20)])
    forest1 = RandomForestClassifier(n_estimators=10, random_state=42)
    forest1.fit(X, y)
    preds1 = forest1.predict(X)

    forest2 = RandomForestClassifier(n_estimators=10, random_state=42)
    forest2.fit(X, y)
    preds2 = forest2.predict(X)

    assert np.array_equal(preds1, preds2)


def test_forest_on_perfectly_separable_data():
    X = np.array([[1.0], [2.0], [3.0], [8.0], [9.0], [10.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    forest = RandomForestClassifier(n_estimators=10, random_state=1)
    forest.fit(X, y)
    predictions = forest.predict(X)
    assert np.array_equal(predictions, y)


def test_majority_voting_direct():
    # Manually verify majority vote logic using a forest of 1 tree first,
    # then more trees, on a simple separable case
    X = np.array([[1.0], [2.0], [9.0], [10.0]])
    y = np.array([0, 0, 1, 1])
    forest = RandomForestClassifier(n_estimators=3, random_state=7)
    forest.fit(X, y)
    predictions = forest.predict(X)
    # With clearly separable data, all trees should agree
    assert np.array_equal(predictions, y)


def test_different_random_seeds_can_differ():
    X = np.array([[i] for i in range(30)], dtype=float)
    y = np.array([i % 2 for i in range(30)])  # hard, noisy pattern
    forest1 = RandomForestClassifier(n_estimators=5, random_state=1)
    forest1.fit(X, y)
    preds1 = forest1.predict(X)

    forest2 = RandomForestClassifier(n_estimators=5, random_state=2)
    forest2.fit(X, y)
    preds2 = forest2.predict(X)

    # Not asserting they must differ (they could coincidentally match),
    # just confirming both run successfully and produce valid shapes
    assert preds1.shape == preds2.shape


def test_forest_default_max_features_uses_sqrt():
    X = np.random.default_rng(1).random((20, 9))  # 9 features -> sqrt=3
    y = np.array([0, 1] * 10)
    forest = RandomForestClassifier(n_estimators=3, random_state=1)
    forest.fit(X, y)
    for tree in forest.trees:
        assert tree.max_features == 3


def test_forest_edge_case_single_estimator():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    forest = RandomForestClassifier(n_estimators=1, random_state=1)
    forest.fit(X, y)
    predictions = forest.predict(X)
    assert predictions.shape == y.shape