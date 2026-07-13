# Iris KNN Classification Pipeline

A modular machine learning project implementing the **K-Nearest Neighbors (KNN)** classification algorithm on the classic Iris dataset. The project demonstrates the complete supervised machine learning workflow—from data preprocessing and feature scaling to model training, evaluation, hyperparameter tuning, and visualization.

## Overview

This project was developed as part of the **DecodeLabs AI Engineering** program to demonstrate the implementation of a complete classification pipeline following software engineering best practices.

The project emphasizes:

- Modular architecture
- Type-safe Python code
- Data preprocessing
- Hyperparameter optimization
- Model evaluation
- Visualization of results

---

## Features

- Load and inspect the Iris dataset
- Randomized train-test split (70/30)
- Feature scaling using `StandardScaler`
- K-Nearest Neighbors (KNN) classifier
- Hyperparameter tuning (K = 1–15)
- Confusion Matrix generation
- Classification Report
- Macro & Weighted F1 Score
- Accuracy vs. K visualization
- Error Rate vs. K visualization
- Strongly typed, modular codebase

---

## Project Structure

```text
iris-knn-classifier/
│
├── plots/
│   ├── accuracy_vs_k.png
│   ├── confusion_matrix.png
│   └── error_rate_vs_k.png
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── schemas.py
│   ├── evaluate.py
│   ├── visualize.py
│   └── pipeline.py
│
├── main.py
├── requirements.txt
├── README.md
└── mypy.ini
```

---

## Machine Learning Pipeline

```
Load Dataset
      │
      ▼
Dataset Inspection
      │
      ▼
Train/Test Split
      │
      ▼
Feature Scaling
      │
      ▼
Hyperparameter Tuning
      │
      ▼
Train Final KNN Model
      │
      ▼
Prediction
      │
      ▼
Evaluation
      │
      ▼
Visualization
```

---

## Technologies Used

| Category | Technologies |
|----------|--------------|
| Language | Python 3 |
| Data Processing | NumPy, Pandas |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib |
| Static Type Checking | MyPy |

---

## Evaluation Metrics

The trained classifier is evaluated using:

- Accuracy
- Confusion Matrix
- Classification Report
- Macro F1 Score
- Weighted F1 Score

These metrics provide a comprehensive assessment of model performance and generalization.

---

## Hyperparameter Optimization

The classifier evaluates multiple neighborhood sizes (`K = 1–15`) and automatically selects the value producing the highest test accuracy.

Performance is visualized using:

- Accuracy vs. K
- Error Rate vs. K

---

## Installation

Clone the repository

```bash
git clone https://github.com/<Jami436>/iris-knn-classifier.git
```

Navigate into the project

```bash
cd iris-knn-classifier
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python main.py
```

---

## Example Output

The pipeline generates:

- Dataset statistics
- Train/Test split information
- Feature scaling verification
- Optimal K value
- Model accuracy
- Classification Report
- Confusion Matrix
- Accuracy vs. K graph
- Error Rate vs. K graph

---

## Future Improvements

Potential enhancements include:

- Cross-validation for hyperparameter tuning
- Support for additional classification algorithms
- Interactive visualizations
- Command-line arguments for configuration
- Automated unit tests
- CI/CD workflow with GitHub Actions

---

## License

This project is intended for educational purposes as part of the DecodeLabs AI Engineering program.