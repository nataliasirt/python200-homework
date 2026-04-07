"""
Warmup exercises for Assignment 01
"""

import pandas as pd
import numpy as np

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

# --- Pandas ---
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

