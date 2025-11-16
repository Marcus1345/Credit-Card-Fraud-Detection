import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import os
import joblib

def make_features(df: pd.DataFrame, scaler: StandardScaler = None) -> Tuple[pd.DataFrame, pd.Series, StandardScaler]:
    df = df.copy()
    y = df["Class"]
    x = df.drop(columns=["Class"])

    if scaler is None:
        scaler = StandardScaler()
        x[['Time','Amount']] = scaler.fit_transform(x[['Time','Amount']])
    else:
        x[['Time','Amount']] = scaler.transform(x[['Time','Amount']])
    return x, y, scaler

if __name__=='__main__':
    df = pd.read_csv('Data/RawDataSet.csv')
    x, y, scaler = make_features(df)
    print(x.head())
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler,"models/scaler.joblib")
    