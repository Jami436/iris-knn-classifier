from typing import Tuple 

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split_dataset(
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.3,
        random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Shuffle and split the dataset into training and testing sets.
    
    Args:
    X: Feature matrix.
    y: Target labels.
    test_size: proportion of the dataset to include in the split.
    random_state: Random seed for reproducibility.
    
    Returns:
    X_train, X_test, y_train, y_test
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=True,
    )

    return X_train, X_test, y_train, y_test

def scale_features(
        X_train: np.ndarray,
        X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Standardize the training and testing features.
    
    The scalar is fitted only on the training data.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler