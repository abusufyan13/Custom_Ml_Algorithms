import numpy as np
import pytest
from src.metrics import gini_impurity, information_gain, best_split

def test_pure_node_single_class():
    y = np.array([0, 0, 0, 0, 0])
    assert gini_impurity(y) == 0.0


def test_pure_node_all_class_one():
    y = np.array([1, 1, 1])
    assert gini_impurity(y) == 0.0


def test_binary_perfectly_mixed():
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    assert gini_impurity(y) == pytest.approx(0.5)


def test_binary_uneven_split():
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 2])
    assert gini_impurity(y) == pytest.approx(0.54)


def test_empty_array_returns_zero():
    y = np.array([])
    assert gini_impurity(y) == 0.0


def test_three_class_uniform():
    y = np.array([0, 0, 1, 1, 2, 2])
    assert gini_impurity(y) == pytest.approx(2 / 3)






def test_information_gain_perfect_split():
    parent = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    left = np.array([0, 0, 0, 0, 0])
    right = np.array([1, 1, 1, 1, 1])
    assert information_gain(parent, left, right) == pytest.approx(0.5)


def test_information_gain_bad_split():
    parent = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    left = np.array([0, 0, 0, 1, 1])
    right = np.array([0, 0, 1, 1, 1])
    assert information_gain(parent, left, right) == pytest.approx(0.02)


def test_information_gain_no_split_needed():
    parent = np.array([0, 0, 0, 0])
    left = np.array([0, 0])
    right = np.array([0, 0])
    assert information_gain(parent, left, right) == pytest.approx(0.0)


def test_information_gain_empty_left_child():
    parent = np.array([0, 0, 1, 1])
    left = np.array([])
    right = np.array([0, 0, 1, 1])
    assert information_gain(parent, left, right) == 0.0


def test_information_gain_empty_right_child():
    parent = np.array([0, 0, 1, 1])
    left = np.array([0, 0, 1, 1])
    right = np.array([])
    assert information_gain(parent, left, right) == 0.0


def test_information_gain_empty_parent():
    parent = np.array([])
    left = np.array([])
    right = np.array([])
    assert information_gain(parent, left, right) == 0.0




def test_best_split_simple_separable_data():
    # Feature 0 perfectly separates the two classes at threshold 2.5
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    result = best_split(X, y)
    assert result is not None
    assert result["feature_index"] == 0
    assert result["threshold"] == pytest.approx(2.5)
    assert result["gain"] == pytest.approx(0.5)

def test_best_split_picks_correct_feature():
    # Feature 0 is random noise, feature 1 perfectly separates classes
    X = np.array([
        [5.0, 1.0],
        [1.0, 1.0],
        [3.0, 4.0],
        [9.0, 4.0],
    ])
    y = np.array([0, 0, 1, 1])
    result = best_split(X, y)
    assert result is not None
    assert result["feature_index"] == 1
    assert result["threshold"] == pytest.approx(2.5)


def test_best_split_no_valid_split_identical_features():
    # All samples have identical feature values -> no valid split
    X = np.array([[1.0], [1.0], [1.0], [1.0]])
    y = np.array([0, 1, 0, 1])
    result = best_split(X, y)
    assert result is None


def test_best_split_pure_labels_no_gain():
    # All labels already the same -> no split improves anything, gain stays 0
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([0, 0, 0])
    result = best_split(X, y)
    assert result is None


def test_best_split_single_sample():
    X = np.array([[1.0]])
    y = np.array([0])
    result = best_split(X, y)
    assert result is None


def test_best_split_rejects_empty_split_in_favor_of_real_one():
    # Ensure it doesn't get fooled by degenerate splits
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 0, 1, 1])
    result = best_split(X, y)
    assert result is not None
    assert result["gain"] > 0.0