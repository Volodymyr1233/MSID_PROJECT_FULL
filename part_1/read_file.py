import pandas as pd

def read_file(path):
    try:
        df = pd.read_csv(path)
        if df.empty:
            print("This file is empty")
            return
        return df
    except FileNotFoundError:
        print("File doesn't exist")
        return
    except Exception:
        print(f"Error:{Exception}")
        return