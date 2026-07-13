from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class EvaluationResult:
    """
    Stores evaluation metrics for a classification model.
    """

    confusion_matrix: np.ndarray
    classification_report: str
    macro_f1: float
    weighted_f1: float