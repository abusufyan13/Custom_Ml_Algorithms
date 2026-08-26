import numpy as np
from src.decision_tree import DecisionTreeClassifier
from src.utils import bootstrap_sample


class RandomForestClassifier:
    """
    A Random Forest classifier built from scratch: an ensemble of
    DecisionTreeClassifier instances trained on bootstrap samples with
    random feature selection at each split, combined via majority voting.

    Parameters
    ----------
    n_estimators : int, default=10
        Number of trees in the forest.
    max_depth : int, default=10
        Maximum depth of each individual tree.
    min_samples_split : int, default=2
        Minimum samples required at a node to attempt a split.
    max_features : int, optional
        Number of features each split considers. If None, defaults to
        sqrt(n_features), rounded down (minimum 1).
    random_state : int, optional
        Seed for reproducibility across bootstrap sampling and trees.
    """

    def __init__(self, n_estimators: int = 10, max_depth: int = 10,
                 min_samples_split: int = 2, max_features: int = None,
                 random_state: int = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the forest: build n_estimators trees on bootstrap samples."""
        X = np.asarray(X)
        y = np.asarray(y)
        n_features = X.shape[1]

        if self.max_features is None:
            tree_max_features = max(1, int(np.sqrt(n_features)))
        else:
            tree_max_features = self.max_features

        rng = np.random.default_rng(self.random_state)
        self.trees = []

        for i in range(self.n_estimators):
            tree_seed = int(rng.integers(0, 1_000_000))

            X_sample, y_sample = bootstrap_sample(X, y, random_state=tree_seed)

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=tree_max_features,
                random_state=tree_seed,
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels via majority vote across all trees."""
        X = np.asarray(X)

        # Shape: (n_estimators, n_samples) -- each row is one tree's predictions
        all_predictions = np.array([tree.predict(X) for tree in self.trees])

        # For each sample (column), find the majority-voted class
        n_samples = X.shape[0]
        final_predictions = np.empty(n_samples, dtype=all_predictions.dtype)

        for i in range(n_samples):
            sample_votes = all_predictions[:, i]
            values, counts = np.unique(sample_votes, return_counts=True)
            final_predictions[i] = values[np.argmax(counts)]

        return final_predictions