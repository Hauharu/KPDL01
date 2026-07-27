import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("Telco_Customer_Churn.csv")

# --- Kiem tra so dong, so cot, ten cot ---
print("So dong, so cot:", df.shape)
print("Ten cac cot:\n", list(df.columns))

# --- Kiem tra kieu du lieu ---
print("\nKieu du lieu ban dau:")
print(df.dtypes)
# Nhan xet: cot TotalCharges dang la 'object' (chuoi) nhung dang le phai la
# so -> chua phu hop. Nguyen nhan: co lan gia tri ' ' (rong) va 'Unknown'.

# Sua kieu: ep TotalCharges ve so, gia tri loi -> NaN
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# --- Kiem tra & loai ban ghi trung lap ---
print("\nSo ban ghi trung lap:", df.duplicated().sum())
df = df.drop_duplicates()
print("Sau khi loai trung, so dong:", len(df))

# --- Kiem tra gia tri khuyet ---
print("\nGia tri khuyet truoc khi xu ly:")
print(df.isnull().sum())

# Xu ly khuyet: BO LUON cac dong co gia tri khuyet (dropna).
# Ly do: so dong khuyet rat nho so voi tong so dong, nen viec loai bo khong anh huong dang ke den ket qua, dong thoi tranh phai bia/uoc luong gia tri.
print("So dong truoc khi bo NaN:", len(df))
df = df.dropna()
df = df.reset_index(drop=True)
print("So dong sau khi bo NaN:", len(df))

print("\nGia tri khuyet sau khi xu ly:")
print(df.isnull().sum())

# --- Kiem tra mien gia tri ---
print("\nMien gia tri cot SO:")
print("tenure        :", df["tenure"].min(), "-", df["tenure"].max())
print("MonthlyCharges:", df["MonthlyCharges"].min(), "-", df["MonthlyCharges"].max())
print("TotalCharges  :", round(df["TotalCharges"].min(), 2), "-", round(df["TotalCharges"].max(), 2))

print("\nMien gia tri cot PHAN LOAI (kiem tra co thong nhat khong):")
for c in ["gender", "Partner", "PhoneService", "InternetService",
          "Contract", "PaymentMethod", "Churn"]:
    print(c, "->", df[c].unique())
# Nhan xet: cac cot phan loai deu thong nhat (vd Yes/No, Male/Female), khong
# co loi chinh ta hay viet hoa/thuong lan lon -> khong can chuan hoa them.

df.to_csv("Telco_clean.csv", index=False)

# CAU 2: THONG KE
print("\n========== CAU 2 ==========")
print("So luong khach hang:", len(df))

print("\nSo khach hang theo gioi tinh:")
print(df["gender"].value_counts())

print("\nSo khach hang theo loai hop dong (Contract):")
print(df["Contract"].value_counts())

print("\nSo khach hang theo dich vu Internet (InternetService):")
print(df["InternetService"].value_counts())

print("\nThong ke MonthlyCharges:")
print("Trung binh   :", round(df["MonthlyCharges"].mean(), 2))
print("Trung vi     :", df["MonthlyCharges"].median())
print("Mode         :", df["MonthlyCharges"].mode()[0])
print("Do lech chuan:", round(df["MonthlyCharges"].std(), 2))

# Bieu do phan bo MonthlyCharges (giong ve pho diem)
bins = np.arange(0, 120 + 5, 5)
plt.figure(figsize=(13, 5))
plt.hist(df["MonthlyCharges"].dropna(),
         bins=bins,
         color="royalblue", edgecolor="white")
plt.title("Cau 2 - Phan bo MonthlyCharges")
plt.xlabel("MonthlyCharges"); plt.ylabel("So luong khach hang")
plt.grid(axis="y", alpha=0.3)
plt.xticks(bins, rotation=90)
plt.tight_layout()
plt.show()

# Nhan xet (xem hinh): bieu do co 2 vung tap trung - mot cum o khoang
# 20 (khach hang dung it dich vu, cuoc thap) va mot cum lon o 70-90 (cuoc
# cao). Phan bo khong can doi (khong hinh chuong), nghieng theo 2 cum.

# Câu 3:
print("\n========== CAU 3 ==========")

def phan_loai(t):
    if t >= 48:
        return "Khach hang trung thanh"
    if t >= 24:
        return "Khach hang on dinh"
    return "Khach hang moi"

df["NhomKH"] = df["tenure"].apply(phan_loai)

print("So luong khach hang tung nhom:")
print(df["NhomKH"].value_counts())

# Bieu do cot so luong tung nhom
dem = df["NhomKH"].value_counts()
plt.figure(figsize=(8, 5))
plt.bar(dem.index, dem.values, color=["green", "royalblue", "orange"])
plt.title("Cau 3 - So luong khach hang theo nhom")
plt.ylabel("So luong"); plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# Quan he giua tenure va kha nang roi bo (Churn)
print("\nTy le roi bo dich vu (Churn=Yes) theo tung nhom (%):")
ty_le = df[df["Churn"] == "Yes"].groupby("NhomKH").size() / df.groupby("NhomKH").size() * 100
print(ty_le.round(1))
# Nhan xet: nhom "Khach hang moi" co ty le roi bo cao nhat (~42%), nhom
# "on dinh" ~21%, nhom "trung thanh" thap nhat ~10%. => tenure cang lon
# (su dung cang lau) thi kha nang roi bo cang THAP. Quan he NGHICH bien.

# CÂU 4:
print("\n========== CAU 4 ==========")
data = df.drop(columns=["customerID", "NhomKH"])   # bo cot dinh danh / cot phu

# 1) Chuyen thuoc tinh NHI PHAN (Binary) sang so 0/1
binary_map = {"Female": 0, "Male": 1, "No": 0, "Yes": 1}
for c in ["gender", "Partner", "PhoneService", "Churn"]:
    data[c] = data[c].map(binary_map)

# 2) Ma hoa thuoc tinh PHAN LOAI (Nominal) bang One-Hot Encoding
data = pd.get_dummies(data,
                      columns=["InternetService", "Contract", "PaymentMethod"],
                      dtype=int)

# 3) Chuan hoa thuoc tinh Số bang Min-Max Scaling -> ve [0, 1]
minmax = MinMaxScaler()
numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
data[numeric_columns] = minmax.fit_transform(data[numeric_columns])

# 4) Kiem tra khong con khuyet
print("Tong so o khuyet sau tien xu ly:", data.isnull().sum().sum())
print("Kich thuoc sau tien xu ly:", data.shape)

# In day du cac cot
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

print("\nMot vai dong ket qua:")
print(data.head())

# 5) Xuat file
data.to_csv("Telco_Customer_Churn_Preprocessed.csv", index=False)
print("\nDa xuat file Telco_Customer_Churn_Preprocessed.csv")