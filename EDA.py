import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   

df = pd.read_csv("Data/RawDataSet.csv")

print("\n Thống kê mô tả :")
print(df.describe)

#hien thi gia tri null
print("\n hiển thị giá trị null: ")
print(df.isnull().sum().sum())

#So giao dich khong hop le
print("Tỷ lệ fraud: ")
print(df["Class"].value_counts())
print(df["Class"].value_counts(normalize=True))

#Biểu đồ 
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.histplot(df["Amount"], bins=50, kde=True)
plt.title(" Phan phoi Amount")

plt.figure(figsize=(10,5))
plt.subplot(1,2,2)
sns.histplot(df["Time"], bins=50, kde=True)
plt.title(" Phan phoi Time")
plt.show()


