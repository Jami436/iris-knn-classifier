from src.data_loader import describe_dataset, load_dataset

def main() -> None:
    describe_dataset()

    X, y = load_dataset()

    print("\nFeature Matrix Shape:", X.shape)
    print("Target Vector Shape :", y.shape)

if __name__ == "__main__":
    main()