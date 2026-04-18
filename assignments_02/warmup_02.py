import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

# --- scikit-learn API ---

# Q1
years = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

model = LinearRegression()
model.fit(years, salary)

pred4 = model.predict([[4]])
pred8 = model.predict([[8]])

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Prediction for 4 years:", pred4[0])
print("Prediction for 8 years:", pred8[0])

# Q2
x = np.array([10, 20, 30, 40, 50])
print("Original shape:", x.shape)
x_2d = x.reshape(-1, 1)
print("New shape:", x_2d.shape)
# scikit-learn needs X to be 2D because it expects a matrix where rows are samples and columns are features, even if there's only one feature.

# Q3
X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_clusters)
labels = kmeans.predict(X_clusters)

print("Cluster centers:", kmeans.cluster_centers_)
print("Points per cluster:", np.bincount(labels))

plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=100, c='black')
plt.title('K-Means Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.savefig('assignments_02/outputs/kmeans_clusters.png')

# --- Linear Regression ---

np.random.seed(42)
num_patients = 100
age = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

# Q1
plt.figure()
plt.scatter(age, cost, c=smoker, cmap="coolwarm")
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Cost")
plt.savefig("assignments_02/outputs/cost_vs_age.png")
# There are two distinct groups visible, one for smokers and one for non-smokers, suggesting that the smoker variable has a significant impact on cost.

# Q2
X = age.reshape(-1, 1)
y = cost
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Q3
model = LinearRegression()
model.fit(X_train, y_train)
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
y_pred = model.predict(X_test)
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = model.score(X_test, y_test)
print("RMSE:", rmse)
print("R²:", r2)
# The slope means that for each additional year of age, the medical cost increases by about that amount.

# Q4
X_full = np.column_stack([age, smoker])
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_full, y, test_size=0.2, random_state=42)
model_full = LinearRegression()
model_full.fit(X_train_full, y_train_full)
r2_full = model_full.score(X_test_full, y_test_full)
print("R² with smoker:", r2_full)
print("Previous R²:", r2)
print("age coefficient:", model_full.coef_[0])
print("smoker coefficient:", model_full.coef_[1])
# The smoker coefficient represents the additional cost for smokers compared to non-smokers.

# Q5
y_pred_full = model_full.predict(X_test_full)
plt.figure()
plt.scatter(y_pred_full, y_test_full)
plt.plot([y_test_full.min(), y_test_full.max()], [y_test_full.min(), y_test_full.max()], 'k--', lw=2)
plt.title("Predicted vs Actual")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("assignments_02/outputs/predicted_vs_actual.png")
# Points above the diagonal mean the model underestimated the cost, below means overestimated.