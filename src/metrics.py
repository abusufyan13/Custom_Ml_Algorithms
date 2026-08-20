import numpy as np


def gini_impurity(y: np.ndarray) -> float:
    """
    Compute Gini Impurity for a set of class labels.

    Gini = 1 - sum(p_i^2) for each class i present in y.

    Parameters
    ----------
    y : np.ndarray
        1D array of class labels (any hashable type, typically int).

    Returns
    -------
    float
        Gini impurity, in range [0, 1 - 1/C] for C classes.
        Returns 0.0 for an empty array (no impurity to measure).
    """
    y = np.asarray(y)

    if y.size == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / y.size
    return 1.0 - np.sum(probabilities ** 2)




def information_gain(parent: np.ndarray, left_child: np.ndarray, right_child: np.ndarray) -> float:
    """
    Compute Information Gain from splitting `parent` into `left_child` and `right_child`.

    Gain = Gini(parent) - WeightedGini(children)

    Parameters
    ----------
    parent : np.ndarray
        Class labels of all samples before the split.
    left_child : np.ndarray
        Class labels of samples routed left by the split.
    right_child : np.ndarray
        Class labels of samples routed right by the split.

    Returns
    -------
    float
        Information gain. Returns 0.0 if either child is empty
        (an empty-child split provides no useful separation).
    """
    n_parent = len(parent)

    if n_parent == 0 or len(left_child) == 0 or len(right_child) == 0:
        return 0.0

    weight_left = len(left_child) / n_parent
    weight_right = len(right_child) / n_parent

    weighted_child_gini = (
        weight_left * gini_impurity(left_child)
        + weight_right * gini_impurity(right_child)
    )

    return gini_impurity(parent) - weighted_child_gini







def best_split(X: np.ndarray, y: np.ndarray):
    """
    Search over all features and thresholds to find the split with the
    highest Information Gain.

    Parameters
    ----------
    X : np.ndarray
        2D array of shape (n_samples, n_features).
    y : np.ndarray
        1D array of class labels, shape (n_samples,).

    Returns
    -------
    dict or None
        Dictionary with keys 'feature_index', 'threshold', 'gain' for the
        best split found. Returns None if no valid split exists (e.g. all
        samples identical across every feature, or fewer than 2 samples).
    """
    n_samples, n_features = X.shape

    if n_samples < 2:
        return None

    best_gain = 0.0
    best_feature_index = None
    best_threshold = None

    for feature_index in range(n_features):
        feature_values = X[:, feature_index]
        unique_values = np.unique(feature_values)

        if unique_values.size < 2:
            # All samples identical on this feature -> no valid threshold
            continue

        # Candidate thresholds: midpoints between consecutive unique values
        thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0

        for threshold in thresholds:
            left_mask = feature_values <= threshold
            right_mask = ~left_mask

            left_y = y[left_mask]
            right_y = y[right_mask]

            gain = information_gain(y, left_y, right_y)

            if gain > best_gain:
                best_gain = gain
                best_feature_index = feature_index
                best_threshold = threshold

    if best_feature_index is None:
        return None

    return {
        "feature_index": best_feature_index,
        "threshold": best_threshold,
        "gain": best_gain,
    }