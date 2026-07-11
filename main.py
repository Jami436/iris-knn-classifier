from src.data_loader import describe_dataset, load_dataset
from src.model import find_best_k, predict, train_knn
from src.preprocessing import scale_features, split_dataset
from src.evaluate import evaluate_model

def main() -> None:
    """
    Main entry point for the Iris KNN Classification Pipeline.
    """

    # Step 1: Load and inspect the dataset
    describe_dataset()

    X, y = load_dataset()

    print("\nFeature Matrix Shape:", X.shape)
    print("Target Vector Shape :", y.shape)

    # Step 2: Shuffle and split the dataset
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    print("\nDataset Split")
    print("-" * 40)
    print(f"Training Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")

    # Step 3: Feature Scaling
    X_train_scaled, X_test_scaled, _ = scale_features(
        X_train,
        X_test,
    )

    print("\nScaled Data Statistics")
    print("-" * 40)

    print("Mean:")
    print(X_train_scaled.mean(axis=0))

    print("\nStandard Deviation:")
    print(X_train_scaled.std(axis=0))

    # Step 4: Hyperparameter Tuning
    print("\nSearching for the best K...")
    print("-" * 40)

    best_k, accuracies, error_rates = find_best_k(
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
    )

    print(f"Best K        : {best_k}")
    print(f"Best Accuracy : {max(accuracies):.4f}")

    # Step 5: Train Final Model
    model = train_knn(
        X_train_scaled,
        y_train,
        best_k,
    )

    # Step 6: Predictions
    predictions = predict(
        model,
        X_test_scaled,
    )

    print("\nFirst 10 Predictions")
    print("-" * 40)
    print(predictions[:10])

    print("\nFirst 10 Actual Labels")
    print("-" * 40)
    print(y_test[:10])

    evaluation = evaluate_model(
        y_test,
        predictions,
    )

    print("\nEvaluation Results")
    print("=" * 50)

    print("\nConfusion Matrix")
    print(evaluation["confusion_matrix"])

    print("\nClassification Report")
    print(evaluation["classification_report"])

    print(f"Macro F1 Score      : {evaluation['macro_f1']:.4f}")
    print(f"Weighted F1 Score   : {evaluation['weighted_f1']:.4f}")


if __name__ == "__main__":
    main()