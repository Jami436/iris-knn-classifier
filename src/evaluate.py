from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

import numpy as np

from src.schemas import EvaluationResult


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> EvaluationResult:
    """
    Evaluate a trained classifier.
    """

    return EvaluationResult(
        confusion_matrix=confusion_matrix(
            y_true,
            y_pred,
        ),
        classification_report=classification_report(
            y_true,
            y_pred,
            target_names=[
                "setosa",
                "versicolor",
                "virginica",
            ],
        ),
        macro_f1=f1_score(
            y_true,
            y_pred,
            average="macro",
        ),
        weighted_f1=f1_score(
            y_true,
            y_pred,
            average="weighted",
        ),
    )