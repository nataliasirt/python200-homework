"""
Warmup exercises for Assignment 01
"""

import pandas as pd

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
