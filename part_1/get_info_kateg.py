from read_file import read_file
import pandas as pd

df = read_file("../depr_dataset.csv")
df["Financial Stress"] = pd.to_numeric(df["Financial Stress"], errors="coerce")



def get_info_kateg():
    categorical_cols = df.select_dtypes(include=['object', 'category'])


    stats_dict = {
        "unikalne_klasy": categorical_cols.nunique(),
        "brakujące_wartosci": categorical_cols.isnull().sum()
    }

    stats = pd.DataFrame(stats_dict)

    stats.to_csv("res_kateg.csv", sep=",", index=True, encoding="utf-8-sig", )

    for col in categorical_cols:
        proportions = categorical_cols[col].value_counts(normalize=True).reset_index()
        proportions.columns = [col, "Proporcja"]

        proportions.to_csv("res_kateg.csv", mode='a', sep=",", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    get_info_kateg()
