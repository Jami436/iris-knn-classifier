from typing import Dict

import numpy as np
from sklearn.metrics import(
    classification_report,
    confusion_matrix,
    f1_score,
)

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, object]:
    """
    Evaluate the KNN model using multiple metrics.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        
        Returns:
            Dictionary containing evaluation metrics.
        """
    
    results = {
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(
            y_true,
            y_pred,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
        ),
    }

    return results