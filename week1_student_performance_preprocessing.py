# Week 1 - Data Acquisition and Pre-processing
# Dataset: UCI Student Performance (Mathematics)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 1. Data Acquisition
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
# Download student.zip manually from UCI if the URL is unavailable.
# Extract student-mat.csv and place it in the same folder as this script.

df = pd.read_csv("student-mat.csv", sep=";")

# 2. Initial Inspection
print("Shape:", df.shape)
print(df.head())
print(df.info())
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nStatistics:\n", df.describe())

# 3. Remove duplicates
df = df.drop_duplicates().copy()

# 4. Missing-value treatment
# UCI reports no missing values in this dataset.
# This check is retained as part of a reusable preprocessing pipeline.
numeric_cols = df.select_dtypes(include=np.number).columns
categorical_cols = df.select_dtypes(exclude=np.number).columns

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 5. Outlier detection using IQR
outlier_summary = {}
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_summary[col] = int(((df[col] < lower) | (df[col] > upper)).sum())

print("\nOutlier counts:\n", outlier_summary)

# Visualize important numeric variables
plt.figure(figsize=(7, 4))
sns.boxplot(data=df[["age", "absences", "G1", "G2", "G3"]])
plt.title("Outlier Detection using Box Plot")
plt.tight_layout()
plt.savefig("outlier_boxplot.png", dpi=200)
plt.show()

# 6. Categorical encoding
df_encoded = pd.get_dummies(
    df,
    columns=categorical_cols,
    drop_first=True,
    dtype=int
)

# 7. Standardization of selected numeric features
scale_cols = ["age", "absences", "G1", "G2", "G3"]
scaler = StandardScaler()
df_encoded[scale_cols] = scaler.fit_transform(df_encoded[scale_cols])

# 8. Correlation visualization
plt.figure(figsize=(7, 5))
sns.heatmap(df[["age", "studytime", "failures", "absences", "G1", "G2", "G3"]].corr(),
            annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation of Selected Numerical Features")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=200)
plt.show()

# 9. Save cleaned data
df_encoded.to_csv("student_performance_cleaned.csv", index=False)
print("\nPre-processing completed successfully.")
print("Cleaned shape:", df_encoded.shape)
