from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

def plot_confusion_matrix(
        confusion_matrix: np.ndarray,
        class_name: List[str],
) -> None:
    """
    Plot and save the confusion matrix.
    """

    Path("plots").mkdir(exist_ok=True)

    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=class_name,
    )

    display.plot(cmap="Blues")

    plt.title("Confusion Matrix")
    plt.tight_layout()

    plt.savefig("plots/confusion_matrix.png")

    plt.show()

def plot_accuracy_vs_k(
        accuracies: List[float],
) -> None:
    """
    Plot Accuracy against k values.
    """

    Path("plots").mkdir(exist_ok=True)

    k_values = list(range(1, len(accuracies) + 1))

    plt.figure(figsize=(8, 5))

    plt.plot(
        k_values,
        accuracies,
        marker="o"
    )

    plt.title("Accuracy vs K")
    plt.xlabel("Number of Neighbors (K)")
    plt.ylabel("Accuracy")
    plt.xticks(k_values)
    plt.grid(True)

    plt.tight_layout()

    plt.savefig("plots/accuracy_vs_k.png")

    plt.show()

def plot_error_rate_vs_k(
        error_rates: List[float],
) -> None:
    """
    Plot Error Rate against K values.
    """

    Path("plots").mkdir(exist_ok=True)

    k_values = list(range(1, len(error_rates) + 1))

    plt.figure(figsize=(8, 5))

    plt.plot(
        k_values,
        error_rates,
        marker="o"
    )

    plt.title("Error Rate vs K")
    plt.xlabel("Number of Neighbors (K)")
    plt.ylabel("Error Rate")
    plt.xticks(k_values)
    plt.grid(True)

    plt.tight_layout()

    plt.savefig("plots/error_rate_vs_k.png")

    plt.show()