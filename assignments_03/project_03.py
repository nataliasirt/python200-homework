import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


OUTPUTS_DIR = Path("assignments_03/outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = Path("assignments_03/spambase.csv")
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"

COLUMN_NAMES = [
    "word_freq_make",
    "word_freq_address",
    "word_freq_all",
    "word_freq_3d",
    "word_freq_our",
    "word_freq_over",
    "word_freq_remove",
    "word_freq_internet",
    "word_freq_order",
    "word_freq_mail",
    "word_freq_receive",
    "word_freq_will",
    "word_freq_people",
    "word_freq_report",
    "word_freq_addresses",
    "word_freq_free",
    "word_freq_business",
    "word_freq_email",
    "word_freq_you",
    "word_freq_credit",
    "word_freq_your",
    "word_freq_font",
    "word_freq_000",
    "word_freq_money",
    "word_freq_hp",
    "word_freq_hpl",
    "word_freq_george",
    "word_freq_650",
    "word_freq_lab",
    "word_freq_labs",
    "word_freq_telnet",
    "word_freq_857",
    "word_freq_data",
    "word_freq_415",
    "word_freq_85",
    "word_freq_technology",
    "word_freq_1999",
    "word_freq_parts",
    "word_freq_pm",
    "word_freq_direct",
    "word_freq_cs",
    "word_freq_meeting",
    "word_freq_original",
    "word_freq_project",
    "word_freq_re",
    "word_freq_edu",
    "word_freq_table",
    "word_freq_conference",
    "char_freq_;",
    "char_freq_(",
    "char_freq_[",
    "char_freq_!",
    "char_freq_$",
    "char_freq_#",
    "capital_run_length_average",
    "capital_run_length_longest",
    "capital_run_length_total",
    "spam_label",
]


def show_section(title):
    print("=" * 72)
    print(title)
    print("=" * 72)


def load_spambase():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)

    df = pd.read_csv(DATA_URL, header=None, names=COLUMN_NAMES)
    df.to_csv(DATA_PATH, index=False)
    return df


def save_boxplot(df, feature_name):
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [
        df.loc[df["spam_label"] == 0, feature_name],
        df.loc[df["spam_label"] == 1, feature_name],
    ]
    ax.boxplot(data, tick_labels=["Ham (0)", "Spam (1)"])
    ax.set_title(f"{feature_name}: Ham vs Spam")
    ax.set_ylabel(feature_name)
    fig.tight_layout()
    safe_name = feature_name.replace("!", "exclamation").replace("$", "dollar")
    fig.savefig(OUTPUTS_DIR / f"{safe_name}_boxplot.png")
    plt.close(fig)


def evaluate_classifier(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
    return {
        "name": name,
        "model": model,
        "accuracy": accuracy,
        "predictions": y_pred,
    }


def print_top_features(name, feature_names, importances, top_n=10):
    top_indices = np.argsort(importances)[::-1][:top_n]
    print(f"{name} top {top_n} features:")
    for idx in top_indices:
        print(f"  {feature_names[idx]:30s} {importances[idx]:.4f}")
    print()


def plot_random_forest_importances(feature_names, importances):
    top_indices = np.argsort(importances)[::-1][:10]
    top_features = [feature_names[idx] for idx in top_indices][::-1]
    top_values = [importances[idx] for idx in top_indices][::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_features, top_values, color="forestgreen")
    ax.set_title("Random Forest Top 10 Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "feature_importances.png")
    plt.close(fig)


df = load_spambase()
feature_names = [col for col in df.columns if col != "spam_label"]
X = df[feature_names]
y = df["spam_label"]


# --- Task 1: Load and Explore ---

show_section("TASK 1: LOAD AND EXPLORE")
print(f"Dataset shape: {df.shape}")
print(f"Number of emails: {len(df)}")
class_counts = y.value_counts().sort_index()
class_props = y.value_counts(normalize=True).sort_index()
print(f"Ham emails (0):  {class_counts[0]} ({class_props[0]:.2%})")
print(f"Spam emails (1): {class_counts[1]} ({class_props[1]:.2%})")
baseline_accuracy = class_props.max()
print(f"Majority-class baseline accuracy: {baseline_accuracy:.4f}")
print()
# The classes are somewhat imbalanced, with ham more common than spam.
# That means raw accuracy must be interpreted carefully because a model can look decent
# just by leaning toward the majority class.

for feature in ["word_freq_free", "char_freq_!", "capital_run_length_total"]:
    save_boxplot(df, feature)

print("Selected feature medians by class:")
for feature in ["word_freq_free", "char_freq_!", "capital_run_length_total"]:
    ham_median = df.loc[df["spam_label"] == 0, feature].median()
    spam_median = df.loc[df["spam_label"] == 1, feature].median()
    print(f"{feature:25s} ham median = {ham_median:8.3f}   spam median = {spam_median:8.3f}")
print()

zero_fraction = (X == 0).mean().sort_values(ascending=False)
print("Most zero-heavy features:")
for feature, value in zero_fraction.head(10).items():
    print(f"  {feature:30s} {value:.2%} zeros")
print()

feature_scales = X.agg(["min", "max", "mean"]).T
print("Selected scale comparison:")
print(
    feature_scales.loc[
        ["word_freq_free", "char_freq_!", "capital_run_length_total"]
    ].round(3)
)
print()
# Many word and character features are zero for most emails, so the dataset is sparse:
# most messages simply do not contain most tracked words or punctuation patterns.
# The scale varies because some features are percentages in roughly 0-100, while the
# capital-run totals can be much larger counts. That matters for distance-based models
# and PCA because large-scale features can dominate the geometry if data is not scaled.
# The spam-vs-ham differences for these three features are fairly dramatic rather than
# subtle, especially for exclamation marks and total capital-letter runs.


# --- Task 2: Prepare Your Data ---

show_section("TASK 2: PREPARE YOUR DATA")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape:  {y_test.shape}")
print()
# The split is stratified so the train and test sets keep a similar spam/ham balance.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# The scaler is fit only on training data to avoid leaking test-set information.

pca = PCA()
pca.fit(X_train_scaled)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
print(f"Components needed to reach 90% explained variance: {n_components_90}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o", ms=3)
ax.axhline(0.90, color="red", linestyle="--", linewidth=1)
ax.axvline(n_components_90, color="red", linestyle="--", linewidth=1)
ax.set_xlabel("Number of Components")
ax.set_ylabel("Cumulative Explained Variance")
ax.set_title("Spambase PCA Cumulative Explained Variance")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUTS_DIR / "spambase_pca_variance_explained.png")
plt.close(fig)

X_train_pca = pca.transform(X_train_scaled)[:, :n_components_90]
X_test_pca = pca.transform(X_test_scaled)[:, :n_components_90]
# PCA is applied after scaling and fitted only on the training data for the same
# reason as the scaler: test data must stay unseen during preprocessing.


# --- Task 3: A Classifier Comparison ---

show_section("TASK 3: CLASSIFIER COMPARISON")
results = []

results.append(
    evaluate_classifier(
        "KNN (unscaled)",
        KNeighborsClassifier(n_neighbors=5),
        X_train,
        X_test,
        y_train,
        y_test,
    )
)

results.append(
    evaluate_classifier(
        "KNN (scaled)",
        KNeighborsClassifier(n_neighbors=5),
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
    )
)

results.append(
    evaluate_classifier(
        "KNN (PCA-reduced)",
        KNeighborsClassifier(n_neighbors=5),
        X_train_pca,
        X_test_pca,
        y_train,
        y_test,
    )
)

print("Decision Tree depth comparison:")
tree_depth_scores = []
for depth in [3, 5, 10, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, tree.predict(X_train))
    test_acc = accuracy_score(y_test, tree.predict(X_test))
    tree_depth_scores.append((depth, train_acc, test_acc))
    print(f"  max_depth={str(depth):>4} train accuracy={train_acc:.4f} test accuracy={test_acc:.4f}")
print()
# As depth increases, training accuracy climbs sharply and reaches essentially 1.0 for
# the unlimited tree. Test accuracy improves only a little at the deepest settings, so
# the fully grown tree is starting to memorize instead of buying much generalization.
# For production, max_depth=10 is a better compromise than an unlimited tree because it
# keeps nearly the same test accuracy with less overfitting risk.

decision_tree = DecisionTreeClassifier(max_depth=10, random_state=42)
results.append(
    evaluate_classifier(
        "Decision Tree (max_depth=10)",
        decision_tree,
        X_train,
        X_test,
        y_train,
        y_test,
    )
)

random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
results.append(
    evaluate_classifier(
        "Random Forest",
        random_forest,
        X_train,
        X_test,
        y_train,
        y_test,
    )
)

results.append(
    evaluate_classifier(
        "Logistic Regression (scaled)",
        LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"),
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
    )
)

results.append(
    evaluate_classifier(
        "Logistic Regression (PCA-reduced)",
        LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"),
        X_train_pca,
        X_test_pca,
        y_train,
        y_test,
    )
)

print_top_features(
    "Decision Tree",
    feature_names,
    decision_tree.feature_importances_,
)
print_top_features(
    "Random Forest",
    feature_names,
    random_forest.feature_importances_,
)
plot_random_forest_importances(feature_names, random_forest.feature_importances_)
# The tree and forest broadly agree on the main signals: punctuation like ! and $,
# words like "remove" and "free", and capitalization features all rank highly, which
# matches common intuition about what spam emails tend to look like.

best_result = max(results, key=lambda item: item["accuracy"])
print(f"Best single-split model: {best_result['name']} ({best_result['accuracy']:.4f})")
print()
# On this split, the Random Forest performs best.
# For KNN and logistic regression, PCA is slightly worse than keeping the full scaled
# feature set, so dimensionality reduction did not help these models here.
# For spam filtering, accuracy is useful but not sufficient. False positives are often
# more costly because losing a legitimate email can be worse than letting one spam
# message through, so precision for the spam class deserves extra attention.

cm = confusion_matrix(y_test, best_result["predictions"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ham", "spam"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Best Model Confusion Matrix")
fig.tight_layout()
fig.savefig(OUTPUTS_DIR / "best_model_confusion_matrix.png")
plt.close(fig)
print("Best model confusion matrix:")
print(cm)
print()
# For the best model on this split, false negatives are slightly more common than false
# positives, meaning a few spam messages still get through.


# --- Task 4: Cross-Validation ---

show_section("TASK 4: CROSS-VALIDATION")
cv_models = {
    "KNN (unscaled)": KNeighborsClassifier(n_neighbors=5),
    "KNN (scaled)": Pipeline(
        [("scaler", StandardScaler()), ("classifier", KNeighborsClassifier(n_neighbors=5))]
    ),
    "KNN (PCA-reduced)": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components_90)),
            ("classifier", KNeighborsClassifier(n_neighbors=5)),
        ]
    ),
    "Decision Tree (max_depth=10)": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression (scaled)": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")),
        ]
    ),
    "Logistic Regression (PCA-reduced)": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components_90)),
            ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")),
        ]
    ),
}

cv_summary = []
for name, model in cv_models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5)
    cv_summary.append((name, scores.mean(), scores.std()))
    print(f"{name:32s} mean = {scores.mean():.4f}   std = {scores.std():.4f}")
print()
most_accurate_cv = max(cv_summary, key=lambda item: item[1])
most_stable_cv = min(cv_summary, key=lambda item: item[2])
print(f"Most accurate by CV: {most_accurate_cv[0]} ({most_accurate_cv[1]:.4f})")
print(f"Most stable by CV:   {most_stable_cv[0]} (std={most_stable_cv[2]:.4f})")
print()
# Cross-validation favors the same top-tier models as the test split, but it also shows
# stability. Here, logistic regression has the lowest fold-to-fold variance, while the
# Random Forest remains the most accurate overall.


# --- Task 5: Building a Prediction Pipeline ---

show_section("TASK 5: PREDICTION PIPELINES")
best_tree_pipeline = Pipeline(
    [("classifier", RandomForestClassifier(n_estimators=100, random_state=42))]
)
best_non_tree_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")),
    ]
)

best_tree_pipeline.fit(X_train, y_train)
tree_pipeline_pred = best_tree_pipeline.predict(X_test)
print("Best tree-based pipeline classification report:")
print(classification_report(y_test, tree_pipeline_pred, target_names=["ham", "spam"]))

best_non_tree_pipeline.fit(X_train, y_train)
non_tree_pipeline_pred = best_non_tree_pipeline.predict(X_test)
print("Best non-tree-based pipeline classification report:")
print(classification_report(y_test, non_tree_pipeline_pred, target_names=["ham", "spam"]))

tree_pipeline_acc = accuracy_score(y_test, tree_pipeline_pred)
non_tree_pipeline_acc = accuracy_score(y_test, non_tree_pipeline_pred)
print(f"Tree pipeline accuracy:      {tree_pipeline_acc:.4f}")
print(f"Non-tree pipeline accuracy:  {non_tree_pipeline_acc:.4f}")
print()
# The two pipelines do not have the same structure because tree-based models do not
# need scaling or PCA, while logistic regression does depend on feature magnitudes.
# Packaging preprocessing and prediction together is valuable because it guarantees the
# exact same steps are applied at inference time, which reduces mistakes during reuse
# and makes deployment or handoff much safer.
# The pipeline results match the earlier manual approach, which is the main sanity check
# that the preprocessing and model packaging are consistent.
