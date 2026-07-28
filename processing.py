import pandas as pd


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_DataFrame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.isnull().sum().sum() > 0:
        df = df.dropna()
    if df.duplicated().sum() > 0:
        df = df.drop_duplicates()
    return df


def Save_CSV(df: pd.DataFrame, path: str):
    return df.to_csv(path, index=False)


if __name__ == '__main__':
    df = pd.read_csv("Data/RawDataSet.csv")
    df = clean_DataFrame(df)
    df = Save_CSV(df, "Data/CleanData.csv")
    print(df)