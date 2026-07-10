from typing import List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def train_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
) -> KNeighborsClassifier:
    """
    Train a KNN classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.
        k: Number of nearest neighbors.

    Returns:
        Trained KNN classifier.
    """

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    return model


def predict(
    model: KNeighborsClassifier,
    X_test: np.ndarray,
) -> np.ndarray:
    """
    Predict labels for the test dataset.
    """

    return model.predict(X_test)


def find_best_k(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_k: int = 15,
) -> Tuple[int, List[float], List[float]]:
    """
    Evaluate K values from 1 to max_k.

    Returns:
        best_k
        accuracies
        error_rates
    """

    accuracies: List[float] = []
    error_rates: List[float] = []

    best_accuracy = 0.0
    best_k = 1

    for k in range(1, max_k + 1):

        model = train_knn(X_train, y_train, k)

        predictions = predict(model, X_test)

        accuracy = accuracy_score(y_test, predictions)
        error = 1 - accuracy

        accuracies.append(accuracy)
        error_rates.append(error)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k

    return best_k, accuracies, error_rates