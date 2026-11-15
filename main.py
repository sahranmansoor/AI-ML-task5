import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    df = pd.read_csv("heart.csv")

    X, y = df.drop("target", axis=1), df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)


if __name__ == "__main__":
    main()
