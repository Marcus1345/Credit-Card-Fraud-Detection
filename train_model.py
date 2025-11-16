import pandas as pd
from features import make_features
from models import build_model, save_model
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import os
import time

def run_model(processed_csv: str, model_out: str):
    print("Reading dataset...")
    df = pd.read_csv(processed_csv)
    
    print("Generating features...")
    x, y, scaler = make_features(df)
    
    print("Splitting train/test...")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, stratify=y, random_state=42
    )
    print("Original training size:", x_train.shape, y_train.shape)
    
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    x_train_res, y_train_res = smote.fit_resample(x_train, y_train)
    print("Training size after SMOTE:", x_train_res.shape, y_train_res.shape)
    
    print("Building model...")
    model = build_model()
    
    print("Start training...")
    start_time = time.time()
    model.fit(x_train_res, y_train_res)
    end_time = time.time()
    print(f"Training finished! Time elapsed: {end_time - start_time:.2f} seconds")
    
    save_model(model, model_out)
    print("Saved model to:", model_out)


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    run_model("Data/RawDataSet.csv", "models/fraud_model.joblib")

