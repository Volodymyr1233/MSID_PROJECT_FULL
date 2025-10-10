import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
from read_file import read_file

df = read_file("../depr_dataset.csv")
df["Financial Stress"] = pd.to_numeric(df["Financial Stress"], errors="coerce")

def create_depression_press_work():
    data = df[["Academic Pressure", "Financial Stress"]]
    
    sns.regplot(x="Financial Stress", y="Academic Pressure", data=data)
    plt.savefig("photos/regression/depression_press_regression.png", dpi=300)
    plt.close()

def create_depression_finance():
    data = df[["Work Pressure", "Job Satisfaction"]]


    sns.regplot(x="Work Pressure", y="Job Satisfaction", data=data)
    plt.savefig("photos/regression/depression_finance_regression.png", dpi=300)
    plt.close()

def create_depression_academic():
    data = df[["Work/Study Hours", "Study Satisfaction"]]


    sns.regplot(x="Work/Study Hours", y="Study Satisfaction", data=data)
    plt.savefig("photos/regression/depression_academic_regression.png", dpi=300)
    plt.close()



def execute_regression():
    create_depression_finance()
    create_depression_press_work()
    create_depression_academic()

if __name__ == "__main__":
    execute_regression()