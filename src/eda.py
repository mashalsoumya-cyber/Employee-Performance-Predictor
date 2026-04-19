import os
import matplotlib.pyplot as plt
import pandas as pd


def perform_eda(df):
    os.makedirs("outputs", exist_ok=True)

    # 1. Class distribution
    plt.figure(figsize=(8, 5))
    df["PerformanceRating"].value_counts().plot(kind="bar")
    plt.title("Performance Rating Distribution")
    plt.xlabel("Performance Rating")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("outputs/class_distribution.png")
    plt.close()

    # 2. Correlation heatmap-like plot using pandas + matplotlib
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    corr = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, interpolation="nearest", aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Feature Correlation")
    plt.tight_layout()
    plt.savefig("outputs/feature_correlation.png")
    plt.close()

    # 3. Salary vs performance
    salary_by_perf = df.groupby("PerformanceRating")["MonthlySalary"].mean()
    plt.figure(figsize=(8, 5))
    salary_by_perf.plot(kind="bar")
    plt.title("Average Salary by Performance")
    plt.xlabel("Performance Rating")
    plt.ylabel("Average Monthly Salary")
    plt.tight_layout()
    plt.savefig("outputs/performance_by_salary.png")
    plt.close()

    print("EDA charts saved in outputs/")