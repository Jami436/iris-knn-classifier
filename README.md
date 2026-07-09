# Iris KNN Classification Pipeline

A modular machine learning project that implements the **K-Nearest Neighbors (KNN)** classification algorithm on the classic Iris dataset. This project focuses on building a complete supervised learning pipeline—from data preprocessing and feature scaling to model training, evaluation, and hyperparameter optimization—using clean, production-inspired Python code.

> **Status:** Work in Progress

## Project Overview

This project is being developed as part of the **DecodeLabs AI Engineering** training program. The objective is to transition from rule-based programming to data-driven machine learning by constructing a robust end-to-end classification pipeline.

The implementation follows industry best practices by emphasizing modular architecture, type safety, reproducibility, and code readability.

## Features

* Load and inspect the Iris dataset
* Randomized data shuffling
* Train-test split (70/30)
* Feature normalization using `StandardScaler`
* K-Nearest Neighbors (KNN) classifier
* Hyperparameter tuning across multiple K values
* Confusion Matrix visualization
* Macro & Weighted F1-Score evaluation
* Accuracy vs. K performance graph
* Modular Python architecture
* Strict type hints with MyPy

## Project Structure

```text
iris-knn-classifier/
│
├── data/
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── evaluate.py
│   ├── visualize.py
│   └── pipeline.py
│
├── plots/
├── tests/
│
├── main.py
├── requirements.txt
├── README.md
└── mypy.ini
```

## Tech Stack

* Python 3.10+
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* MyPy

## Project Workflow

```
Load Dataset
      │
      ▼
Shuffle Data
      │
      ▼
Train/Test Split
      │
      ▼
Feature Scaling
      │
      ▼
Train KNN Model
      │
      ▼
Hyperparameter Tuning
      │
      ▼
Prediction
      │
      ▼
Performance Evaluation
      │
      ▼
Visualization
```

## Getting Started

Clone the repository:

```bash
git clone <repository-url>
cd iris-knn-classifier
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux/macOS**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

## Development Status

This repository is currently under active development.

Upcoming milestones include:

* [ ] Data loading module
* [ ] Data preprocessing
* [ ] Feature scaling
* [ ] KNN model implementation
* [ ] Hyperparameter tuning
* [ ] Evaluation metrics
* [ ] Visualization
* [ ] Documentation improvements

## License

This project is intended for educational and learning purposes.
