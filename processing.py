import pandas as pd

def load_raw(path: str) -> pd.DataFrame:
    return  pd.read_csv(path)

def clean_DataFrame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing_value = df.isnull().sum()
    if df.isnull().sum().sum() > 0:
        df.dropna()
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        df = df.drop_duplicates()
    return df

def Save_CSV(df: pd.DataFrame, path: str):
    return df.to_csv(path, index=False)

if __name__=='__main__':
    df = pd.read_csv("Data/RawDataSet.csv")
    df = clean_DataFrame(df)
    print(df)