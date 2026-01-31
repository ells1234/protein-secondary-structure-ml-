import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# Load data
# -------------------------
df = pd.read_csv("data/protein_aac_features.csv")
X = df.drop(columns=["label"])
y = df["label"]

le = LabelEncoder()
y_enc = le.fit_transform(y)
classes = le.classes_
n_classes = len(classes)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

os.makedirs("figures", exist_ok=True)

# -------------------------
# Plot 1: Class distribution
# -------------------------
plt.figure()
counts = pd.Series(y).value_counts().reindex(classes)
plt.bar(counts.index, counts.values)
plt.title("Class Distribution (Full Dataset)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("figures/01_class_distribution.png", dpi=200)
plt.close()

# -------------------------
# Train models (same as before)
# -------------------------
lr = LogisticRegression(max_iter=1000)
svm = SVC(kernel="rbf", probability=True)  # probability=True needed for ROC/PR curves
rf = RandomForestClassifier(n_estimators=200, random_state=42)

models = {
    "Logistic Regression": lr,
    "SVM (RBF)": svm,
    "Random Forest": rf,
}

# -------------------------
# Helper: confusion matrix plot
# -------------------------
def plot_confusion_matrix(model, name):
    y_pred = model.predict(X_test)
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=classes,
        xticks_rotation=45,
        values_format="d",
    )
    plt.title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.savefig(f"figures/02_confusion_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png", dpi=200)
    plt.close()
    return y_pred

# -------------------------
# Helper: ROC + PR curves (multiclass one-vs-rest)
# -------------------------
def plot_roc_pr(model, name):
    # Binarize y for one-vs-rest curves
    y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))

    # Get scores/probabilities
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    else:
        # fallback for models without predict_proba
        y_score = model.decision_function(X_test)

    # ROC curves
    plt.figure()
    for i in range(n_classes):
        RocCurveDisplay.from_predictions(
            y_test_bin[:, i],
            y_score[:, i],
            name=str(classes[i]),
        )
    plt.title(f"One-vs-Rest ROC — {name}")
    plt.tight_layout()
    plt.savefig(f"figures/03_roc_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png", dpi=200)
    plt.close()

    # PR curves
    plt.figure()
    for i in range(n_classes):
        PrecisionRecallDisplay.from_predictions(
            y_test_bin[:, i],
            y_score[:, i],
            name=str(classes[i]),
        )
    plt.title(f"One-vs-Rest Precision–Recall — {name}")
    plt.tight_layout()
    plt.savefig(f"figures/04_pr_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png", dpi=200)
    plt.close()

# -------------------------
# Train + evaluate + plot
# -------------------------
for name, model in models.items():
    model.fit(X_train, y_train)

    print("\n" + "=" * 60)
    print(name)
    y_pred = plot_confusion_matrix(model, name)
    print(classification_report(y_test, y_pred, target_names=classes))

    # ROC/PR plots for models that can produce probabilities/scores
    plot_roc_pr(model, name)

# -------------------------
# Optional: Random Forest feature importance
# -------------------------
# This is a very nice graph to include in Results/Discussion.
rf.fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(8, 6))
top = importances.head(15)[::-1]  # top 15, reverse for horizontal bar
plt.barh(top.index, top.values)
plt.title("Random Forest — Top 15 AAC Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("figures/05_rf_feature_importance.png", dpi=200)
plt.close()

print("\nSaved figures to ./figures/")
