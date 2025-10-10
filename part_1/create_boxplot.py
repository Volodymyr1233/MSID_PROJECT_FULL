import seaborn as sns

import matplotlib.pyplot as plt
from read_file import read_file

df = read_file("../depr_dataset.csv")

def create_Sleep_Age():
    data = df[["Sleep Duration", "Age", "Gender"]]

    plt.figure(figsize=(12, 6))
    sns.boxplot(x="Sleep Duration", y="Age", data=data, hue="Gender")

    plt.savefig("photos/boxplots/sleep_age.png", dpi=300)
    plt.close()

def create_DHabits_Age():
    data = df[["Dietary Habits", "Age", "Gender"]]

    plt.figure(figsize=(12, 6))
    sns.boxplot(x="Dietary Habits", y="Age", data=data, hue="Gender")
   
    plt.savefig("photos/boxplots/dhabits_age.png", dpi=300)
    plt.close()
    

def create_AcadPress_Depression():
    data = df[["Academic Pressure", "Depression", "Gender"]]
    plt.figure(figsize=(12, 6))

    
    sns.boxplot(x="Depression", y="Academic Pressure", data=data, hue="Gender")
    plt.savefig("photos/boxplots/acad_press.png", dpi=300)
    plt.close()

def execute_create_boxplot():
    create_Sleep_Age()
    create_DHabits_Age()
    create_AcadPress_Depression()

if __name__ == "__main__":
    execute_create_boxplot()