import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure output directory exists
os.makedirs("results/figures", exist_ok=True)


def plot_accuracy_comparison():
    """Bar chart: Accuracy of Custom vs Sklearn, Tree vs Forest (Breast Cancer results)."""
    models = ["Custom\nTree", "Sklearn\nTree", "Custom\nForest", "Sklearn\nForest"]
    accuracies = [0.9386, 0.9474, 0.9649, 0.9561]
    colors = ["#4C72B0", "#DD8452", "#4C72B0", "#DD8452"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(models, accuracies, color=colors)

    ax.set_title("Accuracy Comparison: Custom vs Scikit-Learn (Breast Cancer Dataset)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.85, 1.0)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                 f"{acc:.4f}", ha="center", fontsize=9)

    custom_patch = plt.Rectangle((0, 0), 1, 1, color="#4C72B0", label="Custom")
    sklearn_patch = plt.Rectangle((0, 0), 1, 1, color="#DD8452", label="Sklearn")
    ax.legend(handles=[custom_patch, sklearn_patch])

    fig.tight_layout()
    fig.savefig("results/figures/accuracy_comparison.png", dpi=150)
    plt.close(fig)
    print("Saved: results/figures/accuracy_comparison.png")


def plot_training_time_comparison():
    """Bar chart (log scale): Training time of Custom vs Sklearn (Breast Cancer results)."""
    models = ["Custom\nTree", "Sklearn\nTree", "Custom\nForest", "Sklearn\nForest"]
    times = [5.897669, 0.027529, 6.267104, 0.029779]
    colors = ["#4C72B0", "#DD8452", "#4C72B0", "#DD8452"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(models, times, color=colors)
    ax.set_yscale("log")

    ax.set_title("Training Time Comparison: Custom vs Scikit-Learn (Breast Cancer Dataset)")
    ax.set_ylabel("Training Time (seconds, log scale)")

    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
                 f"{t:.4f}s", ha="center", fontsize=9)

    custom_patch = plt.Rectangle((0, 0), 1, 1, color="#4C72B0", label="Custom")
    sklearn_patch = plt.Rectangle((0, 0), 1, 1, color="#DD8452", label="Sklearn")
    ax.legend(handles=[custom_patch, sklearn_patch])

    fig.tight_layout()
    fig.savefig("results/figures/training_time_comparison.png", dpi=150)
    plt.close(fig)
    print("Saved: results/figures/training_time_comparison.png")


def plot_n_estimators_vs_metrics():
    """Line charts: n_estimators vs accuracy AND vs training time (custom Random Forest, Iris results)."""
    n_estimators_values = [5, 10, 50]
    accuracies = [1.0000, 1.0000, 1.0000]
    train_times = [0.092296, 0.223231, 0.889477]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(n_estimators_values, accuracies, marker="o", color="#4C72B0")
    ax1.set_title("n_estimators vs Accuracy\n(Custom Random Forest, Iris Dataset)")
    ax1.set_xlabel("Number of Trees (n_estimators)")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0.9, 1.05)

    ax2.plot(n_estimators_values, train_times, marker="o", color="#DD8452")
    ax2.set_title("n_estimators vs Training Time\n(Custom Random Forest, Iris Dataset)")
    ax2.set_xlabel("Number of Trees (n_estimators)")
    ax2.set_ylabel("Training Time (seconds)")

    fig.tight_layout()
    fig.savefig("results/figures/n_estimators_vs_metrics.png", dpi=150)
    plt.close(fig)
    print("Saved: results/figures/n_estimators_vs_metrics.png")


def main():
    plot_accuracy_comparison()
    plot_training_time_comparison()
    plot_n_estimators_vs_metrics()
    print("\nAll figures generated in results/figures/")


if __name__ == "__main__":
    main()