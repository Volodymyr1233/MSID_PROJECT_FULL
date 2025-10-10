import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
from read_file import read_file

df = read_file("../depr_dataset.csv")

def create_Age_Depression():
    data = df[["Depression", "Age", "Gender"]]
    sns.catplot(x="Depression", y="Age", data=data, hue="Gender", kind="violin")

    plt.savefig("photos/violinplots/age_depression.png", dpi=300)
    plt.close()

def create_workHours_finStress():
    df["Financial Stress"] = pd.to_numeric(df["Financial Stress"], errors="coerce")
    
    data = df[["Work/Study Hours", "Financial Stress", "Gender"]]

    

    sns.catplot(x="Work/Study Hours", y="Financial Stress", data=data, hue="Gender", kind="violin", height=6, aspect=1.8)

    plt.savefig("photos/violinplots/workHours_finStress.png", dpi=300)
    plt.close()



def execute_create_violinplots():
    create_Age_Depression()
    create_workHours_finStress()

if __name__ == "__main__":
    execute_create_violinplots()