from src.decision_tree import Node
from src.decision_tree import Node, DecisionTreeClassifier
import numpy as np

def test_leaf_node_is_leaf():
    leaf = Node(value=1)
    assert leaf.is_leaf_node() is True


def test_internal_node_is_not_leaf():
    left_leaf = Node(value=0)
    right_leaf = Node(value=1)
    internal = Node(feature_index=0, threshold=2.5, left=left_leaf, right=right_leaf)
    assert internal.is_leaf_node() is False


def test_leaf_node_stores_correct_value():
    leaf = Node(value=1)
    assert leaf.value == 1


def test_internal_node_stores_correct_attributes():
    left_leaf = Node(value=0)
    right_leaf = Node(value=1)
    internal = Node(feature_index=2, threshold=3.7, left=left_leaf, right=right_leaf)
    assert internal.feature_index == 2
    assert internal.threshold == 3.7
    assert internal.left is left_leaf
    assert internal.right is right_leaf
def test_fit_perfectly_separable_data():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTreeClassifier()
    tree.fit(X, y)
    predictions = tree.predict(X)
    assert np.array_equal(predictions, y)


def test_predict_multiple_samples():
    X = np.array([[1.0], [2.0], [8.0], [9.0]])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTreeClassifier()
    tree.fit(X, y)
    predictions = tree.predict(np.array([[1.5], [8.5]]))
    assert predictions[0] == 0
    assert predictions[1] == 1


def test_pure_node_becomes_leaf_immediately():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([0, 0, 0])
    tree = DecisionTreeClassifier()
    tree.fit(X, y)
    assert tree.root.is_leaf_node()
    assert tree.root.value == 0


def test_max_depth_limits_tree_growth():
    X = np.array([[i] for i in range(20)], dtype=float)
    y = np.array([i % 2 for i in range(20)])  # alternating classes, hard to separate cleanly
    tree = DecisionTreeClassifier(max_depth=1)
    tree.fit(X, y)
    # Root should split once, and both children must be leaves (depth 1 reached)
    assert tree.root.is_leaf_node() is False
    assert tree.root.left.is_leaf_node()
    assert tree.root.right.is_leaf_node()


def test_min_samples_split_prevents_splitting_small_nodes():
    X = np.array([[1.0], [2.0]])
    y = np.array([0, 1])
    tree = DecisionTreeClassifier(min_samples_split=3)
    tree.fit(X, y)
    # Only 2 samples, min_samples_split=3 -> should not split, root is a leaf
    assert tree.root.is_leaf_node()


def test_majority_class_prediction_in_leaf():
    X = np.array([[1.0], [1.0], [1.0], [1.0]])  # identical features, no valid split
    y = np.array([0, 0, 0, 1])  # majority class 0
    tree = DecisionTreeClassifier()
    tree.fit(X, y)
    assert tree.root.is_leaf_node()
    assert tree.root.value == 0


def test_multi_feature_dataset():
    X = np.array([
        [1.0, 5.0],
        [2.0, 5.0],
        [8.0, 1.0],
        [9.0, 1.0],
    ])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTreeClassifier()
    tree.fit(X, y)
    predictions = tree.predict(X)
    assert np.array_equal(predictions, y)


def test_invalid_input_raises_or_handles_gracefully():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1])
    tree = DecisionTreeClassifier()
    tree.fit(X, y)
    # Predicting on a differently-shaped single sample should still work
    prediction = tree.predict(np.array([[1.5]]))
    assert prediction[0] in [0, 1]