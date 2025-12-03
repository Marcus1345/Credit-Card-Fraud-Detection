# gui.py
import tkinter as tk
from tkinter import messagebox
from predict import predict_single
import numpy as np

entries = {}

def safe_float(x):
    try:
        return float(x)
    except:
        return 0.0


# ========================= PREDICT =============================
def predict_gui():
    try:
        row = {}

        for col, entry in entries.items():
            value = entry.get().strip()
            row[col] = safe_float(value)

        proba, label = predict_single(
            model_path="models/fraud_model.joblib",
            scaler_path="models/scaler.joblib",
            thr_path="models/best_threshold.txt",
            row_dict=row
        )

        if label == 1:
            messagebox.showwarning(
                "Kết quả",
                f"⚠ GIAO DỊCH CÓ KHẢ NĂNG GIAN LẬN!\nXác suất: {proba:.6f}"
            )
        else:
            messagebox.showinfo(
                "Kết quả",
                f"✓ Giao dịch bình thường.\nXác suất: {proba:.6f}"
            )

    except Exception as e:
        messagebox.showerror("Lỗi", str(e))



# ========================= FAKE FRAUD GENERATOR =============================
def generate_fraud():
    fraud = {}

    fraud["Time"] = np.random.randint(10000, 85000)
    fraud["Amount"] = np.random.uniform(1500, 4500)

    # Feature fraud lệch mạnh
    for i in range(1, 29):
        fraud[f"V{i}"] = np.random.uniform(-5, 5)

    for col, val in fraud.items():
        entries[col].delete(0, tk.END)
        entries[col].insert(0, str(val))

    messagebox.showinfo("Fraud Sample", "Đã tạo giao dịch FRAUD!")


# ========================= NORMAL (NON-FRAUD) GENERATOR =============================
def generate_normal():
    normal = {}

    normal["Time"] = np.random.randint(0, 80000)
    normal["Amount"] = np.random.uniform(1, 100)

    # Giao dịch bình thường: V1–V28 gần 0 theo phân phối chuẩn nhẹ
    for i in range(1, 29):
        normal[f"V{i}"] = float(np.random.normal(0, 1))

    for col, val in normal.items():
        entries[col].delete(0, tk.END)
        entries[col].insert(0, str(val))

    messagebox.showinfo("Normal Sample", "Đã tạo giao dịch bình thường!")



# ========================= GUI =============================
root = tk.Tk()
root.title("Fraud Detection (LightGBM + Threshold)")

# Time + Amount
for i, col in enumerate(["Time", "Amount"]):
    tk.Label(root, text=col).grid(row=i, column=0, padx=5, pady=3)
    entry = tk.Entry(root, width=20)
    entry.grid(row=i, column=1)
    entries[col] = entry

# V1–V28
row_offset = 2
for i in range(1, 29):
    tk.Label(root, text=f"V{i}").grid(row=i + row_offset, column=0, padx=5, pady=3)
    entry = tk.Entry(root, width=20)
    entry.grid(row=i + row_offset, column=1)
    entries[f"V{i}"] = entry

# BUTTON PREDICT
tk.Button(
    root,
    text="Predict",
    command=predict_gui,
    width=25,
    bg="lightblue"
).grid(row=31, column=0, columnspan=2, pady=10)

# BUTTON FRAUD
tk.Button(
    root,
    text="Tạo giao dịch FRAUD",
    command=generate_fraud,
    width=25,
    bg="orange"
).grid(row=32, column=0, columnspan=2, pady=5)

# BUTTON NON-FRAUD
tk.Button(
    root,
    text="Tạo giao dịch NORMAL",
    command=generate_normal,
    width=25,
    bg="lightgreen"
).grid(row=33, column=0, columnspan=2, pady=5)

root.mainloop()
