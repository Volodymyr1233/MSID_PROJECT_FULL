from read_file import read_file
import pandas as pd

df = read_file("../depr_dataset.csv")

df["Financial Stress"] = pd.to_numeric(df["Financial Stress"], errors="coerce")

def get_info():
    numeric_cols = df.select_dtypes(include=['number'])

    stats_dict = {
        'średnia': numeric_cols.mean(),
        'mediana': numeric_cols.median(),
        'wartość minimalna': numeric_cols.min(),
        'wartość maksymalna': numeric_cols.max(),
        'odchylenie standardowe': numeric_cols.std(),
        '5%': numeric_cols.quantile(0.05),
        '95%': numeric_cols.quantile(0.95),
        'puste_wartości': numeric_cols.isnull().sum()
    }

    stats = pd.DataFrame(stats_dict)

    stats.to_csv("result.csv", sep=",", encoding="utf-8-sig")

if __name__ == "__main__":
    get_info()