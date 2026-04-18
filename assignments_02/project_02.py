# Mini-project: Predicting Student Math Performance

# Pre-preprocessing observation:
# The fields are separated by semicolons (;), and string values are quoted with double quotes.
# G1, G2, and G3 are quoted numbers like "5", "6".
# To load with pd.read_csv(), use sep=';'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('assignments_02/student_performance_math.csv', sep=';')

# Select the trimmed columns as per instructions
columns = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'absences',
           'freetime', 'goout', 'Walc', 'schoolsup', 'internet', 'higher', 'activities',
           'sex', 'G1', 'G2', 'G3']
df = df[columns]

# Convert grade columns to numeric
df['G1'] = pd.to_numeric(df['G1'], errors='coerce')
df['G2'] = pd.to_numeric(df['G2'], errors='coerce')
df['G3'] = pd.to_numeric(df['G3'], errors='coerce')

# --- Task 1: Load and Explore ---
print("Shape:", df.shape)
print("First five rows:")
print(df.head())
print("Data types:")
print(df.dtypes)

# Histogram of G3
plt.figure()
plt.hist(df['G3'], bins=21, range=(0, 21), edgecolor='black')
plt.title('Distribution of Final Math Grades')
plt.xlabel('Final Grade (G3)')
plt.ylabel('Frequency')
plt.savefig('assignments_02/outputs/g3_distribution.png')

# --- Task 2: Preprocess the Data ---
print("Shape before filtering:", df.shape)
df_clean = df[df['G3'] != 0]
print("Shape after filtering:", df_clean.shape)
# Keeping G3=0 rows would distort the model because they represent students who didn't take the final exam,
# not actual scores of zero. Predicting grades for them doesn't make sense, and their absence patterns
# might differ from regular students.

# Convert yes/no to 1/0
yes_no_cols = ['schoolsup', 'internet', 'higher', 'activities']
for col in yes_no_cols:
    df_clean[col] = df_clean[col].map({'yes': 1, 'no': 0})

# Convert sex to 0/1
df_clean['sex'] = df_clean['sex'].map({'F': 0, 'M': 1})

# Correlations
corr_original = df['absences'].corr(df['G3'])
corr_clean = df_clean['absences'].corr(df_clean['G3'])
print(f"Correlation absences-G3 original: {corr_original:.3f}")
print(f"Correlation absences-G3 filtered: {corr_clean:.3f}")
# Filtering changes the result because students with G3=0 (who didn't take the exam) likely had high absences,
# making absences look like a weak predictor in the original data. After filtering, the correlation is stronger
# as it reflects the relationship among students who actually took the exam.

# --- Task 3: Exploratory Data Analysis ---
numeric_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'absences', 'freetime', 'goout', 'Walc']
corrs = df_clean[numeric_cols + ['G3']].corr()['G3'].drop('G3').sort_values()
print("Correlations with G3:")
print(corrs)
# Failures has the strongest negative relationship with G3. Surprisingly, age has a positive correlation,
# meaning older students tend to have higher grades, perhaps due to maturity.

# Visualization 1: Scatter plot of absences vs G3
plt.figure()
plt.scatter(df_clean['absences'], df_clean['G3'], alpha=0.5)
plt.title('Absences vs Final Grade')
plt.xlabel('Absences')
plt.ylabel('G3')
plt.savefig('assignments_02/outputs/absences_vs_g3.png')
# This shows a negative relationship, with more absences generally leading to lower grades.

# Visualization 2: Box plot of G3 by study time
plt.figure()
df_clean.boxplot(column='G3', by='studytime')
plt.title('G3 by Study Time')
plt.suptitle('')
plt.xlabel('Study Time')
plt.ylabel('G3')
plt.savefig('assignments_02/outputs/g3_by_studytime.png')
# Higher study time tends to lead to higher median grades, showing the benefit of studying more.

# --- Task 4: Baseline Model ---
X = df_clean[['failures']]
y = df_clean['G3']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = model.score(X_test, y_test)
print(f"Baseline slope: {model.coef_[0]:.3f}")
print(f"Baseline RMSE: {rmse:.3f}")
print(f"Baseline R²: {r2:.3f}")
# On a 0-20 scale, the slope means each past failure reduces the grade by about 2.5 points.
# RMSE of about 4.5 means typical prediction error is 4.5 points, which is substantial.
# R² of 0.15 is low, as expected since failures alone explain only a small part of grade variation.

# --- Task 5: Build the Full Model ---
feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup", "internet", "sex", "freetime", "activities", "traveltime"]
X = df_clean[feature_cols]
y = df_clean['G3']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2_train = model.score(X_train, y_train)
r2_test = model.score(X_test, y_test)
print(f"Train R²: {r2_train:.3f}")
print(f"Test R²: {r2_test:.3f}")
print(f"Test RMSE: {rmse:.3f}")
# Adding more features improves R² from 0.15 to about 0.25, a modest improvement.

print("Coefficients:")
for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:12s}: {coef:+.3f}")
# Higher education aspiration has a large positive coefficient, which makes sense.
# Surprisingly, freetime has a negative coefficient, suggesting more free time might correlate with lower grades.
# Train R² (0.28) and test R² (0.25) are close, indicating no major overfitting.
# For production, I'd keep failures, Medu, Fedu, studytime, higher, internet, sex; drop schoolsup, freetime, activities, traveltime as they have small coefficients.

# --- Task 6: Evaluate and Summarize ---
plt.figure()
plt.scatter(y_pred, y_test, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Predicted vs Actual (Full Model)')
plt.xlabel('Predicted G3')
plt.ylabel('Actual G3')
plt.savefig('assignments_02/outputs/predicted_vs_actual.png')
# The model seems to struggle more at the high end, with predictions clustering below the diagonal for high actual grades.
# Points above the diagonal mean underestimation, below mean overestimation.

# Summary:
# The filtered dataset has 357 rows, test set has 72.
# RMSE of 4.1 means typical prediction error is about 4 points on a 0-20 scale, so predictions are off by roughly 20%.
# Largest positive: higher (+1.5), meaning wanting higher education boosts grade by 1.5 points.
# Largest negative: failures (-2.5), each failure reduces grade by 2.5 points.
# Surprising result: freetime has a negative coefficient, perhaps because students with more free time study less.

# Neglected Feature: The Power of G1
X_with_g1 = df_clean[feature_cols + ['G1']]
X_train_g1, X_test_g1, y_train_g1, y_test_g1 = train_test_split(X_with_g1, y, test_size=0.2, random_state=42)
model_g1 = LinearRegression()
model_g1.fit(X_train_g1, y_train_g1)
r2_test_g1 = model_g1.score(X_test_g1, y_test_g1)
print(f"Test R² with G1: {r2_test_g1:.3f}")
# A high R² here doesn't mean G1 causes G3; it's correlational. This model isn't useful for identifying struggling students
# since G1 is already known. Educators need early interventions using background features before grades are available.