import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
from read_file import read_file

df = read_file("../depr_dataset.csv")

def plot_errors_work_study():
    data = df[["Depression"]]

    f, axs = plt.subplots(2, figsize=(7, 2), sharex=True, layout="tight")


    sns.pointplot(data=data, x="Depression", capsize=.3, errorbar="sd", ax=axs[0])
    sns.stripplot(data=data, x="Depression", jitter=.3, ax=axs[1])
    
    plt.savefig("photos/errorBars/depression.png", dpi=300)
    plt.close()

def plot_errors_academic_pressure():
    data = df[["Academic Pressure"]]

    f, axs = plt.subplots(2, figsize=(7, 2), sharex=True, layout="tight")


    sns.pointplot(data=data, x="Academic Pressure", capsize=.3, errorbar="sd", ax=axs[0])
    sns.stripplot(data=data, x="Academic Pressure", jitter=.3, ax=axs[1])
    
    plt.savefig("photos/errorBars/academic_pressure.png", dpi=300)
    plt.close()

def plot_errors_financial_stress():
    data = df[["Financial Stress"]]
    df["Financial Stress"] = pd.to_numeric(df["Financial Stress"], errors="coerce")

    f, axs = plt.subplots(2, figsize=(7, 2), sharex=True, layout="tight")


    sns.pointplot(data=data, x="Financial Stress", capsize=.3, errorbar="sd", ax=axs[0])
    sns.stripplot(data=data, x="Financial Stress", jitter=.3, ax=axs[1])
    
    plt.savefig("photos/errorBars/financial_stress.png", dpi=300)
    plt.close()

def execute_errorBar():
    plot_errors_work_study()
    plot_errors_academic_pressure()
    plot_errors_financial_stress()

if __name__ == "__main__":
    execute_errorBar()