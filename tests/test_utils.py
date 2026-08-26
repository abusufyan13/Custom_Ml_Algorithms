import numpy as np
from src.utils import bootstrap_sample
from src.utils import bootstrap_sample, random_feature_subset


def test_bootstrap_sample_shape_preserved():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 1, 1, 1])
    X_sample, y_sample = bootstrap_sample(X, y, random_state=42)
    assert X_sample.shape == X.shape
    assert y_sample.shape == y.shape


def test_bootstrap_sample_reproducible_with_same_seed():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 1, 1, 1])
    X_sample1, y_sample1 = bootstrap_sample(X, y, random_state=42)
    X_sample2, y_sample2 = bootstrap_sample(X, y, random_state=42)
    assert np.array_equal(X_sample1, X_sample2)
    assert np.array_equal(y_sample1, y_sample2)


def test_bootstrap_sample_different_with_different_seeds():
    X = np.array([[i] for i in range(20)], dtype=float)
    y = np.array([i % 2 for i in range(20)])
    X_sample1, _ = bootstrap_sample(X, y, random_state=1)
    X_sample2, _ = bootstrap_sample(X, y, random_state=2)
    # With 20 samples and different seeds, extremely unlikely to be identical
    assert not np.array_equal(X_sample1, X_sample2)


def test_bootstrap_sample_contains_only_original_rows():
    X = np.array([[10.0], [20.0], [30.0]])
    y = np.array([0, 1, 2])
    X_sample, y_sample = bootstrap_sample(X, y, random_state=7)
    for value in X_sample.flatten():
        assert value in X.flatten()


def test_bootstrap_sample_x_y_correspondence_preserved():
    # Ensure each sampled X row still matches its corresponding y label
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([10, 20, 30, 40, 50])  # y = X * 10, easy to verify pairing
    X_sample, y_sample = bootstrap_sample(X, y, random_state=42)
    for x_val, y_val in zip(X_sample.flatten(), y_sample):
        assert y_val == x_val * 10


def test_bootstrap_sample_allows_duplicates():
    # With enough samples and draws, duplicates should appear (sampling with replacement)
    X = np.array([[i] for i in range(10)], dtype=float)
    y = np.array([i for i in range(10)])
    X_sample, _ = bootstrap_sample(X, y, random_state=1)
    unique_count = len(np.unique(X_sample))
    # Sampling 10 with replacement from 10 should almost never give all 10 unique
    assert unique_count < 10



def test_random_feature_subset_correct_size():
    subset = random_feature_subset(n_features=10, max_features=3, random_state=1)
    assert len(subset) == 3


def test_random_feature_subset_no_duplicates():
    subset = random_feature_subset(n_features=10, max_features=5, random_state=1)
    assert len(set(subset)) == len(subset)


def test_random_feature_subset_reproducible():
    subset1 = random_feature_subset(n_features=10, max_features=4, random_state=42)
    subset2 = random_feature_subset(n_features=10, max_features=4, random_state=42)
    assert np.array_equal(subset1, subset2)


def test_random_feature_subset_within_valid_range():
    subset = random_feature_subset(n_features=6, max_features=3, random_state=1)
    assert all(0 <= idx < 6 for idx in subset)


def test_random_feature_subset_capped_at_n_features():
    # Requesting more features than exist should just return all of them
    subset = random_feature_subset(n_features=3, max_features=10, random_state=1)
    assert len(subset) == 3