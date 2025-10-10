import seaborn as sns
import matplotlib.pyplot as plt
from read_file import read_file

df = read_file("../depr_dataset.csv")

def create_depression():
    data = df[["Depression"]]

    sns.displot(data, x="Depression")
    plt.savefig("photos/histogram/depression.png", dpi=300)
    plt.close()

def create_sleep_duration():
    data = df[["Sleep Duration"]]

    sns.displot(data, x="Sleep Duration", height=6, aspect=1.8)
    plt.savefig("photos/histogram/work_study.png", dpi=300)
    plt.close()

def create_dietary_habbits():
    data = df[["Dietary Habits"]]


    sns.displot(data, x="Dietary Habits")
    plt.savefig("photos/histogram/dietary_habbits.png", dpi=300)
    plt.close()


def execute_histogram():
    create_depression()
    create_sleep_duration()
    create_dietary_habbits()

if __name__ == "__main__":
    execute_histogram()