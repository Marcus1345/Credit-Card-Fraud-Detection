import tkinter as tk
from tkinter import messagebox
from predict import predict_single

# Dictionary để lưu các entry widget
entries = {}

# Hàm Predict
def predict():
    try:
        # Lấy giá trị từ các entry
        row = {}
        for col, entry in entries.items():
            value = entry.get()
            if value.strip() == "":
                value = 0.0
            row[col] = float(value)
        
        # Gọi hàm predict_single
        proba, label = predict_single(
            "models/fraud_model.joblib",
            "models/scaler.joblib",
            row,
            threshold=0.3
        )

        # Hiển thị kết quả
        if label == 1:
            messagebox.showwarning("Kết quả", f" Giao dịch có khả năng gian lận!\nXác suất: {proba:.4f}")
        else:
            messagebox.showinfo("Kết quả", f" Giao dịch bình thường.\nXác suất: {proba:.4f}")

    except Exception as e:
        messagebox.showerror("Lỗi", str(e))


# Tạo cửa sổ chính
root = tk.Tk()
root.title("Fraud Detection GUI")

# Thêm entry cho Time & Amount
for i, col in enumerate(['Time','Amount']):
    tk.Label(root, text=col).grid(row=i, column=0)
    entry = tk.Entry(root)
    entry.grid(row=i, column=1)
    entries[col] = entry

# Thêm entry cho V1–V28
for i in range(1,29):
    tk.Label(root, text=f'V{i}').grid(row=i+1, column=0)
    entry = tk.Entry(root)
    entry.grid(row=i+1, column=1)
    entries[f'V{i}'] = entry

# Nút Predict
tk.Button(root, text="Predict", command=predict).grid(row=30, column=0, columnspan=2)

root.mainloop()
