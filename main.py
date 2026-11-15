import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("heart.csv")

    X, y = df.drop("target", axis=1), df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    dt_full = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
    print(f"Full tree  -> Train: {dt_full.score(X_train, y_train):.3f} | "
          f"Test: {dt_full.score(X_test, y_test):.3f} | "
          f"Depth: {dt_full.get_depth()}, Leaves: {dt_full.get_n_leaves()}")

    depths = range(1, 16)
    train_acc, test_acc = [], []
    for d in depths:
        m = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_train, y_train)
        train_acc.append(m.score(X_train, y_train))
        test_acc.append(m.score(X_test, y_test))

    best_depth = list(depths)[int(np.argmax(test_acc))]
    print("Best depth:", best_depth)

    plt.figure(figsize=(8, 5))
    plt.plot(depths, train_acc, "o-", label="Train accuracy")
    plt.plot(depths, test_acc, "o-", label="Test accuracy")
    plt.axvline(best_depth, ls="--", color="gray", label=f"best depth={best_depth}")
    plt.xlabel("max_depth"); plt.ylabel("Accuracy"); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig("depth_vs_accuracy.png", dpi=200, bbox_inches="tight"); plt.close()

    dt_best = DecisionTreeClassifier(max_depth=best_depth, random_state=42).fit(X_train, y_train)
    plt.figure(figsize=(20, 10))
    plot_tree(dt_best, feature_names=X.columns.tolist(),
              class_names=["No Disease", "Disease"], filled=True, rounded=True, fontsize=9)
    plt.savefig("decision_tree.png", dpi=200, bbox_inches="tight"); plt.close()

    export_graphviz(dt_best, out_file="tree.dot", feature_names=X.columns.tolist(),
                    class_names=["No Disease", "Disease"], filled=True, rounded=True)

if __name__ == "__main__":
    main()
