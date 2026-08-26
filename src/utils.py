import numpy as np


def bootstrap_sample(X: np.ndarray, y: np.ndarray, random_state: int = None):
    """
    Generate a bootstrap sample (sampling with replacement) from X and y.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_samples, n_features).
    y : np.ndarray
        Labels, shape (n_samples,).
    random_state : int, optional
        Seed for reproducibility. If None, sampling is non-deterministic.

    Returns
    -------
    X_sample : np.ndarray
        Bootstrap-sampled feature matrix, same shape as X.
    y_sample : np.ndarray
        Corresponding bootstrap-sampled labels, same shape as y.
    """
    n_samples = X.shape[0]
    rng = np.random.default_rng(random_state)
    indices = rng.integers(low=0, high=n_samples, size=n_samples)
    return X[indices], y[indices]



def random_feature_subset(n_features: int, max_features: int, random_state: int = None) -> np.ndarray:
    """
    Randomly select a subset of feature indices without replacement.

    Parameters
    ----------
    n_features : int
        Total number of features available.
    max_features : int
        Number of features to randomly select.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of randomly chosen, unique feature indices, length max_features.
    """
    rng = np.random.default_rng(random_state)
    max_features = min(max_features, n_features)
    return rng.choice(n_features, size=max_features, replace=False)