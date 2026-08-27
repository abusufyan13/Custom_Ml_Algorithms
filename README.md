# Custom Decision Tree & Random Forest from Scratch

## Project Overview

This project implements a Decision Tree Classifier and Random Forest Ensemble entirely from scratch using Python and NumPy, without relying on Scikit-Learn's modeling classes. It was built as part of a Machine Learning internship to demonstrate a working understanding of how these algorithms operate internally — from impurity metrics up to full ensemble prediction.

## Internship Information

- **Organization:** Devnexes Digital Solutions Pvt. Ltd.
- **Domain:** Machine Learning
- **Duration:** 1 Month (4 Weeks)
- **Author:** Abu Sufyan

## Objectives

- Implement Gini Impurity and Information Gain from first principles
- Build a recursive Decision Tree classifier with configurable stopping criteria
- Extend to a Random Forest using bootstrap sampling and random feature selection
- Benchmark the custom implementation against Scikit-Learn for correctness and performance

## Features

- Gini Impurity and Information Gain computed with vectorized NumPy operations
- Exhaustive best-split search across features and thresholds
- Recursive Decision Tree with `max_depth`, `min_samples_split`, and optional `max_features`
- Random Forest with bootstrap sampling, per-split random feature selection, and majority voting
- Full unit test suite (pytest) covering metrics, tree, forest, and utility functions
- Benchmarking and visualization scripts comparing results against Scikit-Learn

## Mathematical Foundation

### Gini Impurity

Measures how mixed the class labels are within a node:
custom-ml-algorithms/
├── src/
│ ├── metrics.py # Gini Impurity, Information Gain, Best Split
│ ├── decision_tree.py # Node, DecisionTreeClassifier
│ ├── random_forest.py # RandomForestClassifier
│ └── utils.py # Bootstrap sampling, random feature selection
├── tests/
│ ├── test_metrics.py
│ ├── test_decision_tree.py
│ ├── test_random_forest.py
│ └── test_utils.py
├── examples/
│ ├── decision_tree_experiment.py
│ ├── random_forest_experiment.py
│ ├── benchmark.py
│ └── visualizations.py
├── results/
│ └── figures/ # Generated comparison charts
├── README.md
├── requirements.txt
└── pytest.ini


## Technologies Used

- Python 3.12
- NumPy — core numerical operations
- Matplotlib — visualizations
- pytest — unit testing
- Scikit-Learn — dataset loading, train/test splitting, and benchmarking only (not used in the custom algorithm implementation)

## Limitations

- `best_split` has `O(d·n²)` complexity per node due to recomputing impurity for every candidate threshold, rather than using incremental updates — this is the main source of the ~200x training-time gap versus Scikit-Learn
- No support for missing values or categorical features (numerical features only)
- No pruning implemented beyond the basic stopping criteria (`max_depth`, `min_samples_split`)
- Out-of-bag (OOB) error estimation was not implemented, though the bootstrap sampling mechanism that enables it is in place

## Future Improvements

- Optimize `best_split` using presorted features and incremental impurity updates to close the performance gap with Scikit-Learn
- Add feature importance calculation
- Add support for regression trees (currently classification only)
- Implement OOB score estimation using the bootstrap sampling already in place

## Author

**Abu Sufyan**
Machine Learning Intern, Devnexes Digital Solutions Pvt. Ltd.
GitHub: [abusufyan13](https://github.com/abusufyan13)