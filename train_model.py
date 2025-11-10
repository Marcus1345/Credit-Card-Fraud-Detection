import pandas as pd
from src.features import make_features
from src.models import build_model, load_model
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import os

def run_model(processed_csv: str, model_out: str):
    df = pd.read_csv(processed_csv)
    x,y,scaler = make_features(df)
    x_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,stratify=y,random_state=42)
    smote = SMOTE(random_state=42)
    x_train_res,y_train_res = smote.fit_resample(x_train,y_train)
