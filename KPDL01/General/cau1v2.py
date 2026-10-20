import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot

df = pd.read_csv("D:/Telco_Customer_Churn.csv")
#====================================================================================
#CAU 1
#====================================================================================

# Kiem tra dong cot ten cot
df.shape
df.info()

# Chuyen gia tri str thanh float
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors="coerce")
df.info()

#Kiem tra trung lap du lieu
dong_trung = df[df.duplicated(subset='customerID')]
print(dong_trung)
df = df.drop_duplicates(subset='customerID')
print(df.info())

#Kiem tra gia tri rong
df = df.dropna()
df.info()

# Kiem tra mien gia tri
print(df.describe())
print(df.describe(include=object))

#====================================================================================
#CAU 2
#====================================================================================

#sl kh
print("So luong khach hang: ", len(df))

#sl kh theo gioi tinh
print()
print(df['gender'].value_counts())

#sl kh theo hop dong
print()
print(df['Contract'].value_counts())

#sl kh theo dich vu internet
print()
print(df['InternetService'].value_counts())

#cot MonthlyCharges
print("\nCot MonthlyCharges")
print("Trung binh: ", df['MonthlyCharges'].mean().round(2))
print("Trung vi: ", df['MonthlyCharges'].median())
print("Mode: ", df['MonthlyCharges'].mode().tolist()[0])
print("Do lech chuan: ", df['MonthlyCharges'].std().round(2))


#ve bieu do
plt.figure(figsize=(20, 10))
plt.hist(df['MonthlyCharges'], bins=20, edgecolor='black')
plt.title("biểu đồ phân bố của thuộc tính MonthlyCharges")
plt.show()