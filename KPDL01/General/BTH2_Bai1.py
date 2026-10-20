# ================================================================
# Thu vien pandas: chuyen xu ly du lieu dang bang (DataFrame)
# Viet tat la 'pd' de goi ngan hon trong code
# ================================================================
import pandas as pd

# pd.read_csv(): doc file CSV va chuyen thanh DataFrame (bang du lieu 2 chieu)
# Tham so "Full_Mark_2020.csv": duong dan den file du lieu diem thi
# Ket qua luu vao bien 'df' (viet tat cua DataFrame)
df = pd.read_csv("Full_Mark_2020.csv")

# print(df): in ra mot so dong dau va cuoi cua bang de kiem tra nhanh
print(df)

# df.info(): in thong tin tong quan cua bang: so dong, ten cot, kieu du lieu,
#            so gia tri khong bi trong (non-null) cua tung cot
print(df.info())

# df.columns = [...]: gan lai ten cho tat ca cac cot theo dung thu tu
# Muc dich: doi ten cot goc (tieng Anh kho doc) sang ten ngan, de hieu hon
df.columns = ['STT','Ma','Diali', 'GDCD', 'Hoahoc', 'KHTN','KHXH',
              'LichSu','Ngoaingu','Nguvan','Sinhhoc','Toan','Vatli','Tinhthanh']

# pd.to_numeric(): ep kieu du lieu cua cot 'Vatli' ve dang so (float)
# errors='coerce': neu gap gia tri khong phai so (vi du: chu, ky tu la),
#                  se tu dong chuyen thanh NaN (Not a Number - gia tri trong)
#                  thay vi bao loi va dung chuong trinh
df['Vatli'] = pd.to_numeric(df['Vatli'], errors='coerce')

# Tuong tu cho cot 'Tinhthanh': ep ve kieu so
df['Tinhthanh'] = pd.to_numeric(df['Tinhthanh'], errors='coerce')

# Kiem tra lai kieu du lieu sau khi ep: cot Vatli va Tinhthanh phai la float64
print(df.info())

# df.duplicated(subset='Ma'): kiem tra xem co dong nao bi trung lap
#                              dua tren cot 'Ma' (ma thi sinh) hay khong
# Tra ve Series True/False, True = dong do bi trung
# df[...]: loc va chi giu lai nhung dong co gia tri True (tuc la cac dong trung)
cac_dong_trung = df[df.duplicated(subset='Ma')]

# In ra tat ca cac dong bi trung de kiem tra
print(cac_dong_trung)

# df.drop_duplicates(subset='Ma'): xoa cac dong bi trung theo cot 'Ma'
# Chi giu lai lan xuat hien DAU TIEN cua moi ma thi sinh
# Gan lai vao 'df' de ghi de bang cu bang bang da lam sach
df = df.drop_duplicates(subset='Ma')

# Kiem tra lai sau khi xoa trung: so dong phai giam xuong
print(df.info())

# len(df): dem tong so dong con lai trong bang (= so thi sinh khong trung)
print("So luong hoc sinh du thi:", len(df))

# df['Toan'].count(): dem so gia tri KHONG TRONG trong cot 'Toan'
# Nho hon len(df) neu co thi sinh bo thi mon Toan (diem la NaN)
print("So luong bai thi mon toan: ", df['Toan'].count())

# ================================================================
# TINH TOAN THONG KE CO BAN CHO MON TOAN
# ================================================================

# df['Toan'].mean(): tinh trung binh cong cua tat ca gia tri trong cot 'Toan'
# Tu dong bo qua cac gia tri NaN khi tinh
mean_toan = df['Toan'].mean()

# df['Toan'].median(): tim trung vi - sap xep tat ca diem tu thap den cao,
#                      lay gia tri o vi tri chinh giua
median_toan = df['Toan'].median()

# df['Toan'].mode(): tra ve danh sach cac gia tri xuat hien NHIEU NHAT
# [0]: lay phan tu dau tien trong danh sach do (Mode thu nhat)
mode_toan = df['Toan'].mode()[0]

# df['Toan'].std(): tinh do lech chuan (Standard Deviation)
# Do lech chuan cang lon: diem so cang phan tan, chenh lech nhieu
# Do lech chuan cang nho: diem so cang dong deu, tap trung
std_toan = df['Toan'].std()

# f"...{bien:.2f}": in ket qua ket hop voi chu, :.2f = lam tron 2 chu so thap phan
print(f"Diem trung binh mon Toan: {mean_toan:.2f}")
print(f"Diem trung vi (median) mon Toan: {median_toan}")
print(f"Diem mode mon Toan: {mode_toan}")
print(f"Do lech chuan mon Toan: {std_toan:.2f}")

# ================================================================
# VE BIEU DO PHO DIEM MON TOAN
# ================================================================

# Thu vien matplotlib.pyplot: ve bieu do, do thi trong Python
import matplotlib.pyplot as plt

# df['Toan'].value_counts(): dem so luong hoc sinh dat tung muc diem
#   => tra ve Series: index = gia tri diem, values = so luong hoc sinh
# .sort_index(): sap xep lai theo thu tu diem tang dan (0 -> 10)
diem_toan_counts = df['Toan'].value_counts().sort_index()

# plt.figure(figsize=(15, 6)): tao mot khung ve moi voi kich thuoc 15x6 inch
plt.figure(figsize=(15, 6))

# .plot(kind='bar'): ve bieu do cot (bar chart) tu Series diem_toan_counts
# color='skyblue': mau xanh nhat cho cot
# edgecolor='black': vien den bao quanh moi cot
# width=0.8: do rong moi cot (0 -> 1, cang gan 1 cang rong)
diem_toan_counts.plot(kind='bar', color='skyblue', edgecolor='black', width=0.8)

# plt.title(): dat tieu de cho bieu do
# fontsize=16: co chu 16, fontweight='bold': in dam
plt.title('Pho diem mon Toan nam 2020', fontsize=16, fontweight='bold')

# plt.xlabel(): nhan truc hoanh (truc X)
plt.xlabel('Diem so (0, 0.2, 0.4, ..., 10)', fontsize=12)

# plt.ylabel(): nhan truc tung (truc Y)
plt.ylabel('So luong hoc sinh', fontsize=12)

# plt.xticks(rotation=90): xoay nhan tren truc X di 90 do
# Tranh tinh trang cac nhan diem so bi de chong len nhau
plt.xticks(rotation=90)

# plt.grid(axis='y'): ve duong luoi ngang doc theo truc Y
# linestyle='--': kieu duong ke dut
# alpha=0.7: do trong suot 70% (de nhin qua luoi thay cot phia sau)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# plt.tight_layout(): tu dong can chinh lai cac phan cua bieu do
# Tranh bi cat mat tieu de, nhan truc khi luu/hien thi
plt.tight_layout()

# plt.savefig(): luu bieu do ra file anh PNG vao thu muc hien tai
plt.savefig("pho_diem_toan.png")

# plt.show(): hien thi bieu do len cua so pop-up de xem truc tiep
plt.show()

