import numpy as np
from src.metrics import best_split
from src.utils import random_feature_subset


class Node:
    """
    A single node in a Decision Tree.

    An internal (decision) node has `feature_index`, `threshold`, `left`,
    and `right` set, and `value` is None.

    A leaf node has `value` set (the predicted class), and everything
    else is None.
    """

    def __init__(self, feature_index=None, threshold=None, left=None,
                 right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self) -> bool:
        """Return True if this node is a leaf (holds a prediction)."""
        return self.value is not None
class DecisionTreeClassifier:
    """
    A Decision Tree classifier built from scratch using Gini Impurity
    and Information Gain for split selection.

    Parameters
    ----------
    max_depth : int, default=10
        Maximum depth the tree is allowed to grow to.
    min_samples_split : int, default=2
        Minimum number of samples required at a node to attempt a split.
    """


    def __init__(self, max_depth: int = 10, min_samples_split: int = 2,
                 max_features: int = None, random_state: int = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.root = None
        self._rng = np.random.default_rng(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Build the decision tree from training data."""
        X = np.asarray(X)
        y = np.asarray(y)
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """Recursively build the tree, returning the root Node of this subtree."""
        n_samples, n_features = X.shape
        n_unique_labels = len(np.unique(y))

        if (depth >= self.max_depth
                or n_unique_labels == 1
                or n_samples < self.min_samples_split):
            leaf_value = self._majority_class(y)
            return Node(value=leaf_value)

        if self.max_features is not None:
            feature_indices = random_feature_subset(
                n_features, self.max_features, random_state=self._rng.integers(0, 1_000_000)
            )
        else:
            feature_indices = None

        split = best_split(X, y, feature_indices=feature_indices)

        if split is None:
            leaf_value = self._majority_class(y)
            return Node(value=leaf_value)

        feature_index = split["feature_index"]
        threshold = split["threshold"]

        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature_index=feature_index, threshold=threshold,
                    left=left_subtree, right=right_subtree)

    @staticmethod
    def _majority_class(y: np.ndarray):
        """Return the most frequent class label in y."""
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X."""
        X = np.asarray(X)
        return np.array([self._predict_sample(sample, self.root) for sample in X])

    def _predict_sample(self, sample: np.ndarray, node: Node):
        """Traverse the tree for a single sample and return its predicted class."""
        if node.is_leaf_node():
            return node.value

        if sample[node.feature_index] <= node.threshold:
            return self._predict_sample(sample, node.left)
        else:
            return self._predict_sample(sample, node.right)