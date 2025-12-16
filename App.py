import tkinter as tk
from tkinter import messagebox
from predict import predict_single
import numpy as np
import time

entries = {}

NUM_FEATURES = [
    "amt",
    "city_pop",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "unix_time",
    "distance",
    "merchant",
    "category",
    "hour",
    "day",
    "month",
    "gender"
]


def safe_float(x):
    try:
        return float(x)
    except:
        return 0.0


# ========================= PREDICT =============================
def predict_gui():
   ං
    row = {}
    for col, entry in entries.items():
        row[col] = safe_float(entry.get())

    try:
        proba, label = predict_single(
            model_path="models/fraud_model.joblib",
            scaler_path="models/scaler.joblib",
            thr_path="models/best_threshold.txt",
            row_dict=row
        )

        if label == 1:
            messagebox.showwarning(
                "Kết quả",
                f"⚠ GIAO DỊCH CÓ KHẢ NĂNG GIAN LẬN!\nXác suất: {proba:.4f}"
            )
        else:
            messagebox.showinfo(
                "Kết quả",
                f"✓ Giao dịch bình thường.\nXác suất: {proba:.4f}"
            )

    except Exception as e:
        messagebox.showerror("Lỗi", str(e))


# ========================= FRAUD SAMPLE =============================
def generate_fraud():
    fraud = {
        "amt": np.random.uniform(800, 3000),
        "city_pop": np.random.randint(50, 500),
        "lat": 40 + np.random.uniform(-5, 5),
        "long": -100 + np.random.uniform(-5, 5),
        "merch_lat": 35 + np.random.uniform(-10, 10),
        "merch_long": -110 + np.random.uniform(-10, 10),
        "unix_time": int(time.time()),
        "distance": np.random.uniform(500, 3000),
        "merchant": np.random.randint(200, 600),
        "category": np.random.randint(0, 10),
        "hour": np.random.randint(0, 5),
        "day": np.random.randint(1, 28),
        "month": np.random.randint(1, 12),
        "gender": np.random.randint(0, 2)
    }

    for k, v in fraud.items():
        entries[k].delete(0, tk.END)
        entries[k].insert(0, str(round(v, 4)))

    messagebox.showinfo("Fraud Sample", "Đã tạo giao dịch FRAUD!")


# ========================= NORMAL SAMPLE =============================
def generate_normal():
    normal = {
        "amt": np.random.uniform(5, 120),
        "city_pop": np.random.randint(1000, 50000),
        "lat": 40 + np.random.uniform(-1, 1),
        "long": -100 + np.random.uniform(-1, 1),
        "merch_lat": 40 + np.random.uniform(-1, 1),
        "merch_long": -100 + np.random.uniform(-1, 1),
        "unix_time": int(time.time()),
        "distance": np.random.uniform(1, 50),
        "merchant": np.random.randint(1, 50),
        "category": np.random.randint(0, 5),
        "hour": np.random.randint(9, 18),
        "day": np.random.randint(1, 28),
        "month": np.random.randint(1, 12),
        "gender": np.random.randint(0, 2)
    }

    for k, v in normal.items():
        entries[k].delete(0, tk.END)
        entries[k].insert(0, str(round(v, 4)))

    messagebox.showinfo("Normal Sample", "Đã tạo giao dịch NORMAL!")


# ========================= GUI =============================
root = tk.Tk()
root.title("Fraud Detection – LightGBM")

for i, col in enumerate(NUM_FEATURES):
    tk.Label(root, text=col).grid(row=i, column=0, padx=5, pady=3)
    entry = tk.Entry(root, width=25)
    entry.grid(row=i, column=1)
    entries[col] = entry

tk.Button(root, text="Predict", command=predict_gui, bg="lightblue", width=25)\
    .grid(row=len(NUM_FEATURES), column=0, columnspan=2, pady=10)

tk.Button(root, text="Tạo FRAUD", command=generate_fraud, bg="orange", width=25)\
    .grid(row=len(NUM_FEATURES)+1, column=0, columnspan=2)

tk.Button(root, text="Tạo NORMAL", command=generate_normal, bg="lightgreen", width=25)\
    .grid(row=len(NUM_FEATURES)+2, column=0, columnspan=2)

root.mainloop()
