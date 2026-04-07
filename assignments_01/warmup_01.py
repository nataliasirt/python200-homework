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
