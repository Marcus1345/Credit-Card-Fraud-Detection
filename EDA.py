import numpy as np
import pandas as pd   

df = pd.read_csv("Data/CleanData.csv")

print("\n Thống kê mô tả :")
print(df.describe)

#hien thi gia tri null
print("\n hiển thị giá trị null: ")
print(df.isnull().sum().sum())

#So giao dich khong hop le
print("Tỷ lệ fraud: ")
print(df["Class"].value_counts())
print(df["Class"].value_counts(normalize=True))




