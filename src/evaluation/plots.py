"""Standard plots for all experiments."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

CLASSES = ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC"]


def plot_confusion_matrix(cm: np.ndarray, title: str, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_per_class_f1(f1_scores: dict[str, list], save_path: str) -> None:
    """Bar chart comparing per-class F1 across multiple experiments."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(CLASSES))
    width = 0.8 / len(f1_scores)
    for i, (exp_name, scores) in enumerate(f1_scores.items()):
        ax.bar(x + i * width, scores, width, label=exp_name)
    ax.set_xticks(x + width * (len(f1_scores) - 1) / 2)
    ax.set_xticklabels(CLASSES, rotation=15)
    ax.set_ylabel("F1 Score")
    ax.legend()
    ax.set_title("Per-Class F1 by Experiment")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_accuracy_vs_dimension(results_df, save_path: str) -> None:
    """Scatter plot: feature_dim (x) vs macro_f1 (y) coloured by experiment."""
    import pandas as pd
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in results_df.iterrows():
        ax.scatter(row["feature_dim"], row["macro_f1"], s=100, label=row["experiment"])
        ax.annotate(row["experiment"], (row["feature_dim"], row["macro_f1"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Feature Dimension")
    ax.set_ylabel("Macro F1")
    ax.set_title("Accuracy vs Feature Dimension Trade-off")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
