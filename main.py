import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

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

    rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)
    print(f"Random Forest -> Train: {rf.score(X_train, y_train):.3f} | "
          f"Test: {rf.score(X_test, y_test):.3f}")
    print(classification_report(y_test, rf.predict(X_test),
                                target_names=["No Disease", "Disease"]))
    ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test,
                                          display_labels=["No Disease", "Disease"])
    plt.savefig("confusion_matrix.png", dpi=200, bbox_inches="tight"); plt.close()

    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
    print(imp.sort_values(ascending=False).head(10))
    imp.plot.barh(figsize=(8, 5), title="Random Forest - Feature Importance")
    plt.tight_layout(); plt.savefig("feature_importance.png", dpi=200); plt.close()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in [("Full tree", DecisionTreeClassifier(random_state=42)),
                        ("Tuned tree", DecisionTreeClassifier(max_depth=best_depth, random_state=42)),
                        ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=42))]:
        s = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        print(f"{name:12s} CV accuracy: {s.mean():.3f} +/- {s.std():.3f}")

if __name__ == "__main__":
    main()
