import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

df = pd.read_csv("Telco_Customer_Churn.csv", low_memory=False)

df['Gender']=df['Gender'].map({'Male':0,'Female':1})
df['Partner']=df['Partner'].map({'No':0,'Yes':1})
df['PhoneService']=df['PhoneService'].map({'No':0,'Yes':1})

df['customerID'] = pd.to_numeric(df['customerID'], errors='coerce')
df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
df['InternetService'] = pd.to_numeric(df['InternetService'], errors='coerce')
df['Contract'] = pd.to_numeric(df['Contract'], errors='coerce')








df = df.drop_duplicates(subset='')










# # ================================================================
# # TIM HIEU MA TINH THANH & TRICH XUAT DU LIEU 2 TP
# # ================================================================
#
# # df['Tinhthanh'].value_counts(): dem so dong theo tung ma tinh thanh
# # .sort_index(): sap xep tang dan theo gia tri ma
# print("=== MA TINH THANH CO TRONG DU LIEU ===")
# print(df['Tinhthanh'].value_counts().sort_index())
#
# # Loc hang theo dieu kien: chi lay nhung dong ma cot 'Tinhthanh' == 2.0
# # == 2.0 vì cot Tinhthanh la float64 sau khi qua to_numeric
# # .copy(): tao ban sao doc lap, tranh loi SettingWithCopyWarning khi sua sau
# # Ma tinh trong dataset nay:
# #   1.0 = Ha Noi (ma vung dien thoai)
# #   2.0 = TP. Ho Chi Minh (ma vung dien thoai)
# df_hcm = df[df['Tinhthanh'] == 2.0].copy()
# df_hn  = df[df['Tinhthanh'] == 1.0].copy()
#
# # f-string: in chuoi ket hop bien (len() = dem so dong = so thi sinh)
# print(f"\nMa 2 => TP. Ho Chi Minh: {len(df_hcm)} thi sinh")
# print(f"Ma 1 => Ha Noi:          {len(df_hn)} thi sinh")
#
#
# # ================================================================
# # HAM TIEN ICH: phan tich thong ke cho moi thanh pho
# # Ham nay duoc goi 2 lan: 1 cho HCM, 1 cho Ha Noi
# # ================================================================
# def phan_tich(ten_tp, df_tp):
#     """
#     ten_tp (str): Ten thanh pho, dung de dat tieu de va ten file
#     df_tp (DataFrame): Bang du lieu cua thanh pho do
#     """
#
#     # In duong ke phan cach va ten thanh pho
#     # '='*60: tao chuoi gom 60 dau '='
#     print(f"\n{'='*60}")
#     print(f"  PHAN TICH: {ten_tp}")
#     print(f"{'='*60}")
#
#     # ---- [1] So luong thi sinh ----
#     # len(df_tp): tong so dong = tong so thi sinh cua thanh pho
#     print(f"\n[1] So luong thi sinh du thi: {len(df_tp)}")
#
#     # df_tp['Toan'].count(): chi dem cac o co GIA TRI (bo qua NaN)
#     # => so thi sinh thuc su co diem mon Toan (khong tinh nguoi bo thi)
#     print(f"    So bai thi mon Toan:       {df_tp['Toan'].count()}")
#
#     # ---- [2] Thong ke diem Toan ----
#     # .mean(): trung binh cong tat ca diem Toan (bo qua NaN tu dong)
#     mean_t = df_tp['Toan'].mean()
#
#     # .median(): trung vi - gia tri chinh giua khi sap xep tang dan
#     med_t  = df_tp['Toan'].median()
#
#     # .mode(): tra ve Series chua tat ca gia tri xuat hien nhieu nhat
#     mode_series = df_tp['Toan'].mode()
#
#     # .iloc[0]: lay phan tu dau tien theo vi tri (index = 0)
#     # Kiem tra .empty truoc de tranh loi khi DataFrame khong co du lieu
#     mode_t = mode_series.iloc[0] if not mode_series.empty else float('nan')
#
#     # .std(): do lech chuan (Standard Deviation) - do muc do phan tan diem so
#     std_t  = df_tp['Toan'].std()
#
#     print(f"\n[2] Thong ke diem Toan - {ten_tp}:")
#     print(f"    Trung binh : {mean_t:.2f}")   # :.2f = lam tron 2 chu so thap phan
#     print(f"    Trung vi   : {med_t:.2f}")
#     print(f"    Mode       : {mode_t:.2f}")
#     print(f"    Do lech chuan: {std_t:.2f}")
#
#     # .value_counts(): dem so hoc sinh dat tung muc diem
#     # .sort_index(): sap xep lai cac muc diem tang dan (0 -> 10)
#     counts_toan = df_tp['Toan'].value_counts().sort_index()
#
#     # plt.subplots(): tao mot Figure (khung ve) va mot Axes (vung ve)
#     # figsize=(15, 5): kich thuoc khung ve la 15x5 inch
#     # Tra ve cap (fig, ax); ax la doi tuong de ve bieu do
#     fig, ax = plt.subplots(figsize=(15, 5))
#
#     # ax.bar(): ve bieu do cot len truc ax
#     # counts_toan.index.astype(str): chuyen gia tri diem (float) sang chuoi
#     #   de hien thi dep hon tren truc X
#     # counts_toan.values: mang so luong hoc sinh tuong ung
#     # color='steelblue': mau xanh duong dam
#     # edgecolor='black': vien den giua cac cot
#     # width=0.8: do rong cot (0 -> 1)
#     ax.bar(counts_toan.index.astype(str), counts_toan.values,
#            color='steelblue', edgecolor='black', width=0.8)
#
#     # ax.set_title(): dat tieu de cho vung ve (ax), khac plt.title() o cho
#     # fontweight='bold': chu dam
#     ax.set_title(f'Pho diem mon Toan - {ten_tp}', fontsize=14, fontweight='bold')
#
#     # ax.set_xlabel() / ax.set_ylabel(): dat nhan truc X va Y
#     ax.set_xlabel('Diem so', fontsize=11)
#     ax.set_ylabel('So luong hoc sinh', fontsize=11)
#
#     # ax.tick_params(): dieu chinh cac vach (tick) va nhan (label) tren truc
#     # axis='x': chi ap dung cho truc X
#     # rotation=90: xoay nhan 90 do de khong bi chong len nhau
#     ax.tick_params(axis='x', rotation=90)
#
#     # ax.grid(): ve duong luoi ngang (axis='y') kieu net dut ('--')
#     # alpha=0.6: 60% duc (trong suot 40%)
#     ax.grid(axis='y', linestyle='--', alpha=0.6)
#
#     # plt.tight_layout(): tu can chinh layout de khong bi cut mat chu
#     plt.tight_layout()
#
#     # Tao ten file tu ten thanh pho: thay ' ' -> '_', xoa dau '.'
#     ten_file_toan = f"pho_diem_toan_{ten_tp.replace(' ','_').replace('.','')}.png"
#
#     # plt.savefig(): luu bieu do ra file anh
#     plt.savefig(ten_file_toan)
#
#     # plt.show(): hien thi bieu do len man hinh
#     plt.show()
#     print(f"    => Da luu bieu do: {ten_file_toan}")
#
#     # ---- [3] Kiem tra thi ca KHTN va KHXH ----
#     # .notna(): tra ve True neu o do KHONG phai NaN (tuc la co gia tri)
#     # &: phep AND bitwise - ca hai dieu kien phai dung
#     # => Loc ra nhung dong ma ca KHTN lan KHXH deu co gia tri
#     ca_hai = df_tp[(df_tp['KHTN'].notna() & (df_tp['KHTN'] > 0)) & (df_tp['KHXH'].notna() & (df_tp['KHXH']>0))]
#     print(f"\n[3] Thi sinh thi ca KHTN va KHXH: {len(ca_hai)} nguoi")
#
#     if len(ca_hai) > 0:
#         # .head(5): lay toi da 5 dong dau tien
#         # .to_string(index=False): in ra dang chuoi, bo cot so thu tu (index)
#         print(ca_hai[['Ma','KHTN','KHXH']].head(5).to_string(index=False))
#
#     # ---- [4 & 5] Diem tong cong ----
#     # Tao ban sao de khong anh huong den DataFrame goc
#     df_tp = df_tp.copy()
#
#     # .fillna(df_tp['KHXH']): neu KHTN la NaN thi lay gia tri KHXH thay the
#     # Quy tac: uu tien KHTN, neu khong co thi dung KHXH
#     df_tp['KH_chon'] = df_tp['KHTN'].fillna(df_tp['KHXH'])
#
#     # .sum(axis=1): tinh tong theo chieu ngang (cong tung hang)
#     # min_count=4: chi tinh tong khi ca 4 cot deu co gia tri
#     #              neu thieu bat ky cot nao => ket qua la NaN
#     df_tp['Tong'] = df_tp[['Toan','Nguvan','Ngoaingu','KH_chon']].sum(axis=1, min_count=4)
#
#     mean_tg = df_tp['Tong'].mean()
#     med_tg  = df_tp['Tong'].median()
#     mode_tg_series = df_tp['Tong'].mode()
#     mode_tg = mode_tg_series.iloc[0] if not mode_tg_series.empty else float('nan')
#     std_tg  = df_tp['Tong'].std()
#
#     print(f"\n[4+5] Thong ke diem TONG (Toan+Van+Ngoaingu+KH) - {ten_tp}:")
#
#     # .count(): dem so gia tri KHONG TRONG trong cot 'Tong'
#     print(f"    So thi sinh co du 4 mon: {df_tp['Tong'].count()}")
#     print(f"    Trung binh : {mean_tg:.2f}")
#     print(f"    Trung vi   : {med_tg:.2f}")
#     print(f"    Mode       : {mode_tg:.2f}")
#     print(f"    Do lech chuan: {std_tg:.2f}")
#
#     # .dropna(): xoa cac gia tri NaN truoc khi ve (tranh loi ve bieu do)
#     # .round(2): lam tron 2 chu so thap phan de gom nhom diem sat nhau
#     counts_tong = df_tp['Tong'].dropna().round(2).value_counts().sort_index()
#
#     fig2, ax2 = plt.subplots(figsize=(16, 5))
#
#     # Ve bieu do cot cho diem tong, mau cam dam (darkorange)
#     ax2.bar(counts_tong.index.astype(str), counts_tong.values,
#             color='darkorange', edgecolor='black', width=0.8)
#     ax2.set_title(f'Pho diem Tong cong (Toan+Van+Ngoaingu+KH) - {ten_tp}',
#                   fontsize=13, fontweight='bold')
#     ax2.set_xlabel('Tong diem', fontsize=11)
#     ax2.set_ylabel('So luong hoc sinh', fontsize=11)
#
#     # labelsize=7: thu nho co chu nhan truc X vi co rat nhieu muc diem tong
#     ax2.tick_params(axis='x', rotation=90, labelsize=7)
#     ax2.grid(axis='y', linestyle='--', alpha=0.6)
#     plt.tight_layout()
#
#     ten_file_tong = f"pho_diem_tong_{ten_tp.replace(' ','_').replace('.','')}.png"
#     plt.savefig(ten_file_tong)
#     plt.show()
#     print(f"    => Da luu bieu do: {ten_file_tong}")
#
#
# # ================================================================
# # GOI HAM phan_tich() CHO TUNG THANH PHO
# # ================================================================
#
# # Goi ham voi tham so: ten_tp="TP Ho Chi Minh", df_tp=df_hcm
# phan_tich("TP Ho Chi Minh", df_hcm)
#
# # Goi lan 2 cho Ha Noi voi cung ham nhung tham so khac
# phan_tich("Ha Noi", df_hn)
