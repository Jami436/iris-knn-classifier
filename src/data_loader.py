from typing import Tuple

import numpy as np
from sklearn.datasets import load_iris
from sklearn.utils import Bunch

def load_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the Iris dataset.
    
    Returns:
        Tuple containing:
        - Feature matrix (X)
        - Target labels (y)
        """
    
    iris: Bunch = load_iris()

    X: np.ndarray = iris.data
    y: np.ndarray = iris.target

    return X, y

def describe_dataset() -> None:
    """
    Print basic information about the Iris dataset.
    """
    iris: Bunch = load_iris()

    print("=" * 50)
    print("Iris dataset information")
    print("=" * 50)

    print(f"Samples       : {iris.data.shape[0]} ")
    print(f"Features      : {iris.data.shape[1]}")
    print(f"Target Classes: {iris.target_names.tolist()}")

    print("\nFeature Names:")
    for feature in iris.feature_names:
        print(f"  - {feature}")

    print("\nDataset Shape:")
    print(iris.data.shape)