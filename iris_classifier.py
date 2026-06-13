import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score, 
    accuracy_score,
    ConfusionMatrixDisplay,
)

# ──────────────────────────────────────────────
# SECTION 1 — LOAD & EXPLORE THE DATASET
# ──────────────────────────────────────────────

print("  DECODE LABS PROJECT 2 — KNN IRIS CLASSIFIER")
print("-" * 60)

iris = load_iris()
X = iris.data        
y = iris.target       
feature_names = iris.feature_names
class_names   = iris.target_names

print("\nDATASET OVERVIEW")
print(f"  Samples      : {X.shape[0]}")
print(f"  Features     : {X.shape[1]}  → {feature_names}")
print(f"  Classes      : {list(class_names)}")
print(f"  Class counts : {dict(zip(class_names, np.bincount(y)))}")

# ──────────────────────────────────────────────
# SECTION 2 — TRAIN / TEST SPLIT
# ──────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y        
)

print("\nTRAIN / TEST SPLIT  (80 / 20)")
print(f"  Training samples : {X_train.shape[0]}")
print(f"  Testing  samples : {X_test.shape[0]}")

# ──────────────────────────────────────────────
# SECTION 3 — PREPROCESSING
# ──────────────────────────────────────────────

print("\nPREPROCESSING")

# — Normalization (MinMaxScaler)
norm_scaler = MinMaxScaler()
X_train_norm = norm_scaler.fit_transform(X_train)
X_test_norm  = norm_scaler.transform(X_test)

print("  MinMaxScaler  → feature range [0, 1]")

# — Standardization (StandardScaler)
std_scaler = StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)
X_test_std  = std_scaler.transform(X_test)

print("  StandardScaler → mean≈0, std≈1")

# ──────────────────────────────────────────────
# SECTION 4 — FIND OPTIMAL K USING CROSS-VALIDATION
# ──────────────────────────────────────────────

print("\nFINDING OPTIMAL K (cross-validation on training set)")

k_values = range(1, 21)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    scores = cross_val_score(knn, X_train_std, y_train, cv=5, scoring="accuracy")
    cv_scores.append(scores.mean())

best_k = k_values[np.argmax(cv_scores)]
print(f"  Best K = {best_k}  (CV accuracy = {max(cv_scores):.4f})")

# ──────────────────────────────────────────────
# SECTION 5 — TRAIN THE KNN MODEL
# ──────────────────────────────────────────────

print("\nTRAINING KNN CLASSIFIER")
knn_model = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean")
knn_model.fit(X_train_std, y_train)
print(f"  Model: {knn_model}")

# ──────────────────────────────────────────────
# SECTION 6 — EVALUATE THE MODEL
# ──────────────────────────────────────────────

y_pred = knn_model.predict(X_test_std)

accuracy    = accuracy_score(y_test, y_pred)
f1_macro    = f1_score(y_test, y_pred, average="macro")
f1_weighted = f1_score(y_test, y_pred, average="weighted")
f1_per_cls  = f1_score(y_test, y_pred, average=None)
cm          = confusion_matrix(y_test, y_pred)

print("\nEVALUATION METRICS")
print(f"  Accuracy          : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
print(f"  F1-Score (macro)  : {f1_macro:.4f}")
print(f"  F1-Score (weighted): {f1_weighted:.4f}")
print(f"  F1-Score per class:")
for cls, score in zip(class_names, f1_per_cls):
    print(f"    {cls:15s}: {score:.4f}")

print("\nFULL CLASSIFICATION REPORT")
print(classification_report(y_test, y_pred, target_names=class_names))

# ──────────────────────────────────────────────
# SECTION 7 — VISUALIZATIONS
# ──────────────────────────────────────────────

plt.style.use('default')
fig = plt.figure(figsize=(12, 6))
fig.suptitle(
    "Iris KNN Classifier Dashboard"
)
gs = gridspec.GridSpec(1, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── Plot 1 : K vs CV Accuracy ────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(k_values, cv_scores, color='cornflowerblue', linewidth=2.5, marker="o",
         markersize=5)
ax1.axvline(best_k, color='firebrick', linestyle="--", linewidth=1.5,
            label=f"Best K = {best_k}")
ax1.set_title("K vs Cross-Validation Accuracy", fontsize=11, pad=8)
ax1.set_xlabel("K (neighbors)")
ax1.set_ylabel("CV Accuracy")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.2)

# ── Plot 2 : Confusion Matrix ────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=class_names, yticklabels=class_names,
    ax=ax2, linewidths=0.5, linecolor="white",
    annot_kws={"size": 13, "weight": "bold"}
)
ax2.set_title("Confusion Matrix", fontsize=11, pad=8)
ax2.set_xlabel("Predicted Label")
ax2.set_ylabel("True Label")


# ──────────────────────────────────────────────
# SAVE & SHOW
# ──────────────────────────────────────────────

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
output_path = "iris_knn_results.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n┖ Dashboard saved → {output_path}")
plt.show()

print("\n✅ DONE — KNN Classification Pipeline Complete!")
print("=" * 60)