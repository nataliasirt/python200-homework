"""
Warmup exercises for Assignment 01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --- Pandas ---
# Pandas Q1: Create DataFrame and display properties

print("=" * 50)
print("Pandas Question 1")
print("=" * 50)

data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

# Display first three rows
print("\nFirst three rows:")
print(df.head(3))

# Display shape
print(f"\nShape: {df.shape}")

# Display data types
print(f"\nData types:")
print(df.dtypes)

print("\n" + "=" * 50)

# Pandas Q2: Filter passed students with grade above 80
print("Pandas Question 2")
print("=" * 50)
filtered_df = df[(df["passed"] == True) & (df["grade"] > 80)]
print(filtered_df)

print("\n" + "=" * 50)
# Pandas Q3: Add grade_curved column
print("Pandas Question 3")
print("=" * 50)
df["grade_curved"] = df["grade"] + 5
print(df)

print("\n" + "=" * 50)
# Pandas Q4: Add name_upper column and print names
print("Pandas Question 4")
print("=" * 50)
df["name_upper"] = df["name"].str.upper()
print(df[["name", "name_upper"]])

print("\n" + "=" * 50)
# Pandas Q5: Group by city and compute mean grade
print("Pandas Question 5")
print("=" * 50)
mean_by_city = df.groupby("city")["grade"].mean()
print(mean_by_city)

print("\n" + "=" * 50)
# Pandas Q6: Replace Austin with Houston and print name/city
print("Pandas Question 6")
print("=" * 50)
df["city"] = df["city"].replace("Austin", "Houston")
print(df[["name", "city"]])

print("\n" + "=" * 50)
# Pandas Q7: Sort by grade descending and print top 3 rows
print("Pandas Question 7")
print("=" * 50)
top3 = df.sort_values(by="grade", ascending=False).head(3)
print(top3)

print("\n" + "=" * 50)

# --- NumPy ---
# NumPy Q1: Create 1D array from list and print properties
print("NumPy Question 1")
print("=" * 50)
arr1d = np.array([10, 20, 30, 40, 50])
print(f"Shape: {arr1d.shape}")
print(f"Dtype: {arr1d.dtype}")
print(f"Ndim: {arr1d.ndim}")

print("\n" + "=" * 50)
# NumPy Q2: Create 2D array and print shape and size
print("NumPy Question 2")
print("=" * 50)
arr2d = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
print(f"Shape: {arr2d.shape}")
print(f"Size: {arr2d.size}")

print("\n" + "=" * 50)
# NumPy Q3: Slice top-left 2x2 block from 2D array
print("NumPy Question 3")
print("=" * 50)
top_left_2x2 = arr2d[0:2, 0:2]
print(top_left_2x2)

print("\n" + "=" * 50)
# NumPy Q4: Create zeros and ones arrays
print("NumPy Question 4")
print("=" * 50)
zeros_3x4 = np.zeros((3, 4))
print("3x4 array of zeros:")
print(zeros_3x4)
ones_2x5 = np.ones((2, 5))
print("\n2x5 array of ones:")
print(ones_2x5)

print("\n" + "=" * 50)
# NumPy Q5: Create array using arange and compute statistics
print("NumPy Question 5")
print("=" * 50)
arr_arange = np.arange(0, 50, 5)
print(f"Array: {arr_arange}")
print(f"Shape: {arr_arange.shape}")
print(f"Mean: {arr_arange.mean()}")
print(f"Sum: {arr_arange.sum()}")
print(f"Standard Deviation: {arr_arange.std()}")

print("\n" + "=" * 50)
# NumPy Q6: Generate random normal distribution and compute statistics
print("NumPy Question 6")
print("=" * 50)
random_normal = np.random.normal(loc=0, scale=1, size=200)
print(f"Mean: {random_normal.mean()}")
print(f"Standard Deviation: {random_normal.std()}")

print("\n" + "=" * 50)

# --- Matplotlib ---
# Matplotlib Q1: Line plot of squares
print("Matplotlib Question 1")
print("=" * 50)
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]
plt.figure(figsize=(8, 5))
plt.plot(x, y)
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.savefig("outputs/matplotlib_q1_squares.png")
plt.close()
print("Saved plot to outputs/matplotlib_q1_squares.png")

print("\n" + "=" * 50)
# Matplotlib Q2: Bar plot of subject scores
print("Matplotlib Question 2")
print("=" * 50)
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]
plt.figure(figsize=(8, 5))
plt.bar(subjects, scores)
plt.title("Subject Scores")
plt.xlabel("Subject")
plt.ylabel("Score")
plt.savefig("outputs/matplotlib_q2_subject_scores.png")
plt.close()
print("Saved plot to outputs/matplotlib_q2_subject_scores.png")

print("\n" + "=" * 50)
# Matplotlib Q3: Scatter plot with two datasets
print("Matplotlib Question 3")
print("=" * 50)
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]
plt.figure(figsize=(8, 5))
plt.scatter(x1, y1, color="blue", label="Dataset 1")
plt.scatter(x2, y2, color="red", label="Dataset 2")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Scatter Plot - Two Datasets")
plt.grid(True)
plt.savefig("outputs/matplotlib_q3_scatter.png")
plt.close()
print("Saved plot to outputs/matplotlib_q3_scatter.png")

print("\n" + "=" * 50)
# Matplotlib Q4: Subplots with line and bar plots
print("Matplotlib Question 4")
print("=" * 50)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left subplot: line plot
ax1.plot(x, y)
ax1.set_title("Squares")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.grid(True)

# Right subplot: bar plot
ax2.bar(subjects, scores)
ax2.set_title("Subject Scores")
ax2.set_xlabel("Subject")
ax2.set_ylabel("Score")

plt.tight_layout()
plt.savefig("outputs/matplotlib_q4_subplots.png")
plt.close()
print("Saved plot to outputs/matplotlib_q4_subplots.png")

print("\n" + "=" * 50)

# --- Descriptive Statistics ---
# Descriptive Stats Q1: Compute mean, median, variance, and standard deviation
print("Descriptive Stats Question 1")
print("=" * 50)
data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
print(f"Mean: {np.mean(data)}")
print(f"Median: {np.median(data)}")
print(f"Variance: {np.var(data)}")
print(f"Standard Deviation: {np.std(data)}")

print("\n" + "=" * 50)
# Descriptive Stats Q2: Histogram of random normal distribution
print("Descriptive Stats Question 2")
print("=" * 50)
scores = np.random.normal(65, 10, 500)
plt.figure(figsize=(8, 5))
plt.hist(scores, bins=20)
plt.title("Distribution of Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.savefig("outputs/descriptive_stats_q2_histogram.png")
plt.close()
print("Saved histogram to outputs/descriptive_stats_q2_histogram.png")

print("\n" + "=" * 50)
# Descriptive Stats Q3: Boxplot comparing two groups
print("Descriptive Stats Question 3")
print("=" * 50)
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]
plt.figure(figsize=(8, 5))
plt.boxplot([group_a, group_b], labels=["Group A", "Group B"])
plt.title("Score Comparison")
plt.ylabel("Score")
plt.savefig("outputs/descriptive_stats_q3_boxplot.png")
plt.close()
print("Saved boxplot to outputs/descriptive_stats_q3_boxplot.png")

print("\n" + "=" * 50)
# Descriptive Stats Q4: Side-by-side boxplots comparing normal and exponential distributions
print("Descriptive Stats Question 4")
print("=" * 50)
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)

plt.figure(figsize=(8, 5))
plt.boxplot([normal_data, skewed_data], labels=["Normal", "Exponential"])
plt.title("Distribution Comparison")
plt.ylabel("Value")
plt.savefig("outputs/descriptive_stats_q4_distributions.png")
plt.close()
print("Saved distribution comparison to outputs/descriptive_stats_q4_distributions.png")

# Comment on skewness and appropriate statistics
print("\nComment on distributions:")
print("The exponential distribution is more skewed (right-skewed).")
print("For the normal distribution, mean and median are similar, so either works well.")
print("For the exponential (skewed) distribution, median is more appropriate because it's robust to outliers.")

print("\n" + "=" * 50)
# Descriptive Stats Q5: Mean, median, and mode comparison
print("Descriptive Stats Question 5")
print("=" * 50)
data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

print("Data1: [10, 12, 12, 16, 18]")
print(f"Mean: {np.mean(data1)}")
print(f"Median: {np.median(data1)}")
print(f"Mode: {pd.Series(data1).mode()[0]}")

print("\nData2: [10, 12, 12, 16, 150]")
print(f"Mean: {np.mean(data2)}")
print(f"Median: {np.median(data2)}")
print(f"Mode: {pd.Series(data2).mode()[0]}")

# Comment on why mean and median differ for data2
print("\nWhy are median and mean so different for data2?")
print("Data2 contains an outlier (150) that heavily influences the mean.")
print("The mean is pulled upward by this extreme value, while the median remains")
print("stable because it only depends on the middle value(s) in the sorted data.")
print("This demonstrates that median is more robust to outliers than mean.")

print("\n" + "=" * 50)

# --- Hypothesis Testing ---
# Hypothesis Question 1: Independent samples t-test
print("Hypothesis Question 1")
print("=" * 50)
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

t_stat, p_val = stats.ttest_ind(group_a, group_b)
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_val}")

print("\n" + "=" * 50)
# Hypothesis Question 2: Interpret p-value at alpha = 0.05
print("Hypothesis Question 2")
print("=" * 50)
alpha = 0.05
if p_val < alpha:
    print(f"Result is statistically significant (p-value {p_val:.4f} < {alpha})")
    print("We reject the null hypothesis.")
else:
    print(f"Result is NOT statistically significant (p-value {p_val:.4f} >= {alpha})")
    print("We fail to reject the null hypothesis.")

print("\n" + "=" * 50)
# Hypothesis Question 3: Paired t-test
print("Hypothesis Question 3")
print("=" * 50)
before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

t_stat_paired, p_val_paired = stats.ttest_rel(before, after)
print(f"T-statistic: {t_stat_paired}")
print(f"P-value: {p_val_paired}")

print("\n" + "=" * 50)
# Hypothesis Question 4: One-sample t-test
print("Hypothesis Question 4")
print("=" * 50)
scores = [72, 68, 75, 70, 69, 74, 71, 73]
benchmark = 70

t_stat_one_sample, p_val_one_sample = stats.ttest_1samp(scores, benchmark)
print(f"T-statistic: {t_stat_one_sample}")
print(f"P-value: {p_val_one_sample}")
print(f"Testing if mean of scores is significantly different from benchmark {benchmark}")

print("\n" + "=" * 50)
# Hypothesis Question 5: One-tailed t-test (group_a < group_b)
print("Hypothesis Question 5")
print("=" * 50)
t_stat_one_tailed, p_val_one_tailed = stats.ttest_ind(group_a, group_b, alternative='less')
print(f"One-tailed test (group_a < group_b)")
print(f"T-statistic: {t_stat_one_tailed}")
print(f"P-value: {p_val_one_tailed}")

print("\n" + "=" * 50)
# Hypothesis Question 6: Plain-language conclusion for Q1
print("Hypothesis Question 6")
print("=" * 50)
conclusion = (
    f"The independent samples t-test comparing group_a (mean={np.mean(group_a):.1f}) "
    f"and group_b (mean={np.mean(group_b):.1f}) shows that group_b has significantly "
    f"higher scores than group_a (t={t_stat:.3f}, p={p_val:.4f}). "
    f"This difference is extremely unlikely to occur by chance alone, suggesting a "
    f"real and meaningful difference between the two groups."
)
print(conclusion)

print("\n" + "=" * 50)

# --- Correlation ---
# Correlation Q1: Pearson correlation using np.corrcoef()
print("Correlation Question 1")
print("=" * 50)
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Compute correlation matrix
corr_matrix = np.corrcoef(x, y)
print("Correlation matrix:")
print(corr_matrix)
print(f"\nPearson correlation coefficient: {corr_matrix[0, 1]}")

# Expected correlation comment
print("\nExpected correlation: 1.0 (perfect positive correlation)")
print("Why? Because y = 2*x (a perfect linear relationship).")
print("When one variable increases, the other increases proportionally.")

print("\n" + "=" * 50)
# Correlation Q2: Pearson correlation with p-value
print("Correlation Question 2")
print("=" * 50)
x2 = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y2 = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]

# Compute correlation using numpy
corr_coef_q2 = np.corrcoef(x2, y2)[0, 1]
print(f"Pearson correlation coefficient: {corr_coef_q2}")

# Compute p-value using pandas Series corr method
df_temp = pd.DataFrame({"x": x2, "y": y2})
print(f"(Note: scipy.stats.pearsonr() would also return a p-value)")
print(f"This correlation suggests a negative relationship between x and y.")

print("\n" + "=" * 50)
# Correlation Q3: Correlation matrix using df.corr()
print("Correlation Question 3")
print("=" * 50)
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df_corr = pd.DataFrame(people)
correlation_matrix = df_corr.corr()
print(correlation_matrix)

print("\n" + "=" * 50)
# Correlation Q4: Scatter plot with negative correlation
print("Correlation Question 4")
print("=" * 50)
x4 = [10, 20, 30, 40, 50]
y4 = [90, 75, 60, 45, 30]

plt.figure(figsize=(8, 5))
plt.scatter(x4, y4, color="red", s=100)
plt.title("Negative Correlation")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.savefig("outputs/correlation_q4_negative.png")
plt.close()
print("Saved scatter plot to outputs/correlation_q4_negative.png")

print("\n" + "=" * 50)
# Correlation Q5: Heatmap of correlation matrix
print("Correlation Question 5")
print("=" * 50)
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0, square=True)
plt.title("Correlation Heatmap")
plt.savefig("outputs/correlation_q5_heatmap.png")
plt.close()
print("Saved heatmap to outputs/correlation_q5_heatmap.png")

print("\n" + "=" * 50)

# --- Pipelines ---
# Pipeline Question 1: Data pipeline with plain functions
print("Pipeline Question 1")
print("=" * 50)

# Define the array with missing values
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

# Step 1: Create a Series from the array
def create_series(arr):
    """Takes a NumPy array and returns a pandas Series with the name 'values'."""
    return pd.Series(arr, name="values")

# Step 2: Clean the data by removing NaN values
def clean_data(series):
    """Takes a Series, removes NaN values using .dropna(), and returns the cleaned Series."""
    return series.dropna()

# Step 3: Summarize the data
def summarize_data(series):
    """
    Takes a Series and returns a dictionary with mean, median, std, and mode.
    """
    summary = {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }
    return summary

# Step 4: Create the pipeline
def data_pipeline(arr):
    """Chains the three functions together: create_series -> clean_data -> summarize_data."""
    series = create_series(arr)
    cleaned_series = clean_data(series)
    summary = summarize_data(cleaned_series)
    return summary

# Run the pipeline and print results
result = data_pipeline(arr)
print(f"Mean: {result['mean']}")
print(f"Median: {result['median']}")
print(f"Std: {result['std']}")
print(f"Mode: {result['mode']}")

print("\n" + "=" * 50)

