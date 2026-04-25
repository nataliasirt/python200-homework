from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits, load_iris
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


OUTPUTS_DIR = Path("assignments_03/outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def show_section(title):
    print("=" * 60)
    print(title)
    print("=" * 60)


iris = load_iris(as_frame=True)
X = iris.data
y = iris.target


# --- Preprocessing ---

show_section("PREPROCESSING QUESTION 1: Train/Test Split")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
print()

show_section("PREPROCESSING QUESTION 2: StandardScaler")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Mean of each column in X_train_scaled:")
print(X_train_scaled.mean(axis=0))
print()
# We fit the scaler only on the training set to avoid leaking information
# from the test set into preprocessing.


# --- KNN ---

show_section("KNN QUESTION 1: k=5 on unscaled data")
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
y_pred_unscaled = knn_unscaled.predict(X_test)
acc_unscaled = accuracy_score(y_test, y_pred_unscaled)
print(f"Accuracy: {acc_unscaled:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_unscaled, target_names=iris.target_names))
print()

show_section("KNN QUESTION 2: k=5 on scaled data")
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = knn_scaled.predict(X_test_scaled)
acc_scaled = accuracy_score(y_test, y_pred_scaled)
print(f"Accuracy: {acc_scaled:.4f}")
print()
# On this split, scaling slightly hurts KNN performance.
# KNN depends on distances, but Iris features are already on fairly similar scales, so
# rescaling does not help here and can slightly change which neighbors are closest.

show_section("KNN QUESTION 3: 5-fold cross-validation with k=5")
knn_cv = KNeighborsClassifier(n_neighbors=5)
cv_scores = cross_val_score(knn_cv, X_train, y_train, cv=5)
print("Cross-validation scores for each fold:")
for i, score in enumerate(cv_scores, start=1):
    print(f"Fold {i}: {score:.4f}")
print(f"\nMean CV score: {cv_scores.mean():.4f}")
print(f"Standard deviation: {cv_scores.std():.4f}")
print()
# Cross-validation is more reliable than a single split because it averages
# performance across multiple train/validation partitions.

show_section("KNN QUESTION 4: Hyperparameter tuning (k values)")
k_values = [1, 3, 5, 7, 9, 11, 13, 15]
best_k = None
best_score = -1.0
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    mean_score = scores.mean()
    print(f"k = {k:2d}: mean CV score = {mean_score:.4f}")
    if mean_score > best_score:
        best_k = k
        best_score = mean_score
print()
print(f"Best k value: {best_k}")
print(f"Best mean CV score: {best_score:.4f}")
print()
# k=5 is a reasonable choice because it performed well without becoming too sensitive
# to noise like k=1 or too smooth like larger k values.

show_section("CLASSIFIER EVALUATION QUESTION 1: Confusion Matrix")
cm = confusion_matrix(y_test, y_pred_unscaled)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("KNN Confusion Matrix")
fig.tight_layout()
fig.savefig(OUTPUTS_DIR / "knn_confusion_matrix.png")
plt.close(fig)
print(cm)
print(f"Saved: {OUTPUTS_DIR / 'knn_confusion_matrix.png'}")
print()
# For this split, the model does not confuse any species pair; the confusion matrix
# is perfectly diagonal.


# --- Decision Trees ---

show_section("DECISION TREES QUESTION 1")
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
tree_accuracy = accuracy_score(y_test, y_pred_tree)
print(f"Accuracy: {tree_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_tree, target_names=iris.target_names))
print()
# On this split, the Decision Tree is slightly less accurate than the unscaled KNN
# model (0.9667 vs. 1.0000).
# Scaling should not affect a Decision Tree result because tree splits depend on
# feature thresholds, not on distance calculations.


# --- Logistic Regression and Regularization ---

show_section("LOGISTIC REGRESSION QUESTION 1")
for c_value in [0.01, 1.0, 100.0]:
    # In this scikit-learn version, liblinear cannot fit the 3-class Iris problem
    # directly, so OneVsRestClassifier preserves the requested solver behavior.
    model = OneVsRestClassifier(
        LogisticRegression(C=c_value, max_iter=1000, solver="liblinear")
    )
    model.fit(X_train_scaled, y_train)
    coef_total = sum(np.abs(estimator.coef_).sum() for estimator in model.estimators_)
    print(f"C = {c_value:<6} total |coef| sum = {coef_total:.4f}")
print()
# As C increases, the total coefficient magnitude grows.
# That means weaker regularization allows the model to use larger weights, while
# stronger regularization keeps coefficients smaller and the decision boundary simpler.


# --- PCA ---

digits = load_digits()
X_digits = digits.data
y_digits = digits.target
images = digits.images

show_section("PCA QUESTION 1")
print(f"X_digits shape: {X_digits.shape}")
print(f"images shape: {images.shape}")

fig, axes = plt.subplots(1, 10, figsize=(15, 3))
for digit in range(10):
    sample_idx = np.where(y_digits == digit)[0][0]
    axes[digit].imshow(images[sample_idx], cmap="gray_r")
    axes[digit].set_title(str(digit))
    axes[digit].axis("off")
fig.tight_layout()
fig.savefig(OUTPUTS_DIR / "sample_digits.png")
plt.close(fig)
print(f"Saved: {OUTPUTS_DIR / 'sample_digits.png'}")
print()

show_section("PCA QUESTION 2")
pca = PCA()
pca.fit(X_digits)
scores = pca.transform(X_digits)

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap="tab10", s=10)
ax.set_xlabel("PC1 Score")
ax.set_ylabel("PC2 Score")
ax.set_title("Digits Projected onto First Two Principal Components")
fig.colorbar(scatter, ax=ax, label="Digit")
fig.tight_layout()
fig.savefig(OUTPUTS_DIR / "pca_2d_projection.png")
plt.close(fig)
print(f"Saved: {OUTPUTS_DIR / 'pca_2d_projection.png'}")
print()
# Same-digit images generally form visible clusters in the 2D projection, although
# several classes still overlap because two principal components do not capture all detail.

show_section("PCA QUESTION 3")
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o", ms=3)
ax.set_xlabel("Number of Components")
ax.set_ylabel("Cumulative Explained Variance")
ax.set_title("PCA Cumulative Explained Variance")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUTS_DIR / "pca_variance_explained.png")
plt.close(fig)

components_for_80 = np.argmax(cumulative_variance >= 0.80) + 1
print(f"Components needed for about 80% variance: {components_for_80}")
print(f"Saved: {OUTPUTS_DIR / 'pca_variance_explained.png'}")
print()
# It takes about 13 components to explain 80% of the variance.


def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)


show_section("PCA QUESTION 4")
sample_indices = list(range(5))
component_counts = [2, 5, 15, 40]

fig, axes = plt.subplots(len(component_counts) + 1, len(sample_indices), figsize=(10, 10))

for col, idx in enumerate(sample_indices):
    axes[0, col].imshow(images[idx], cmap="gray_r")
    axes[0, col].set_title(f"Digit {y_digits[idx]}")
    axes[0, col].axis("off")

for row, n_components in enumerate(component_counts, start=1):
    for col, idx in enumerate(sample_indices):
        axes[row, col].imshow(
            reconstruct_digit(idx, scores, pca, n_components), cmap="gray_r"
        )
        axes[row, col].axis("off")
        if col == 0:
            axes[row, col].set_ylabel(f"n={n_components}", rotation=90, labelpad=14)

axes[0, 0].set_ylabel("Original", rotation=90, labelpad=14)
fig.suptitle("Digit Reconstructions from Principal Components", y=0.92)
fig.tight_layout()
fig.savefig(OUTPUTS_DIR / "pca_reconstructions.png")
plt.close(fig)
print(f"Saved: {OUTPUTS_DIR / 'pca_reconstructions.png'}")
print()
# The digits become clearly recognizable around n=15, which lines up reasonably well
# with the variance curve since about 13 components already capture around 80% variance.
