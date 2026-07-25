import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple

NUM_COLS = [
    "amt",
    "city_pop",
    "distance",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "unix_time"
]

# Columns to drop if still present (identifiers, not features)
DROP_COLS = ["cc_num"]

TARGET_COL = "is_fraud"


def make_features(
    df: pd.DataFrame,
    scaler: StandardScaler = None
) -> Tuple[pd.DataFrame, pd.Series, StandardScaler]:

    df = df.copy()

    y = df[TARGET_COL]

    x = df.drop(columns=[TARGET_COL] + DROP_COLS, errors="ignore")

    if scaler is None:
        scaler = StandardScaler()
        x[NUM_COLS] = scaler.fit_transform(x[NUM_COLS])
    else:
        x[NUM_COLS] = scaler.transform(x[NUM_COLS])

    return x, y, scaler
