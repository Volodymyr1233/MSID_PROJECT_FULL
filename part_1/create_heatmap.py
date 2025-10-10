import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
from read_file import read_file

def create_heatmap_age_financil():
    df = read_file("../depr_dataset.csv")
    df["Financial Stress"] = pd.to_numeric(df["Financial Stress"], errors="coerce")
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    df['Sleep Duration'] = df['Sleep Duration'].map({"'5-6 hours'": 5.5, "'Less than 5 hours'": 4, "'7-8 hours'": 7.5, "'More than 8 hours'": 10, "Others": 0})
    df['Dietary Habits'] = df['Dietary Habits'].map({"Healthy": 3, "Moderate": 2, "Unhealth": 1, "Others": 0})
    df["Have you ever had suicidal thoughts ?"] = df["Have you ever had suicidal thoughts ?"].map({'Yes': 1, 'No': 0})
    df["Family History of Mental Illness"] = df["Family History of Mental Illness"].map({'Yes': 1, 'No': 0})
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    correlation_matrix = numeric_df.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.xticks(rotation=30, ha='right')
    plt.subplots_adjust(bottom=0.2, left=0.15)
    plt.savefig("photos/heatmap/heamap.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    create_heatmap_age_financil()