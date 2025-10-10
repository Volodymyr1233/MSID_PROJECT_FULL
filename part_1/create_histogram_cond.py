import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
from read_file import read_file

df = read_file("../depr_dataset.csv")

def create_depression_finStress():
    df["Financial Stress"] = pd.to_numeric(df["Financial Stress"], errors="coerce")
    data = df[["Depression", "Financial Stress"]]

    sns.displot(data, x="Financial Stress", hue="Depression",  multiple="stack")
    plt.savefig("photos/histogramCond/financial_stress.png", dpi=300)
    plt.close()

def create_work_hours():
    data = df[["Work/Study Hours", "Have you ever had suicidal thoughts ?"]]
    sns.displot(data, x="Work/Study Hours", hue="Have you ever had suicidal thoughts ?", multiple="stack")

    plt.savefig("photos/histogramCond/work_study.png", dpi=300)
    plt.close()


def execute_create_diagram_cond():
    create_depression_finStress()
    create_work_hours()

if __name__ == "__main__":
    execute_create_diagram_cond()

