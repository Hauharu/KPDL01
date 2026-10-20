# ================================================================
# Thu vien pandas: xu ly du lieu dang bang (DataFrame)
import pandas as pd

# Thu vien matplotlib.pyplot: ve bieu do, do thi
import matplotlib.pyplot as plt

# Thu vien matplotlib: cau hinh toan cuc cho bieu do
import matplotlib

# matplotlib.rcParams: thiet lap font mac dinh de tranh loi hien thi Unicode
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ================================================================
# LOAD & CLEAN DU LIEU
# ================================================================

# pd.read_csv(): doc file CSV thanh DataFrame
# low_memory=False: doc toan bo file 1 lan, tranh suy dien sai kieu du lieu
df = pd.read_csv("Full_Mark_2020.csv", low_memory=False)

# Dat lai ten cot theo dung thu tu cot trong file goc
df.columns = ['STT', 'Ma', 'Diali', 'GDCD', 'Hoahoc', 'KHTN', 'KHXH',
              'LichSu', 'Ngoaingu', 'Nguvan', 'Sinhhoc', 'Toan', 'Vatli', 'Tinhthanh']

# pd.to_numeric(..., errors='coerce'): ep cot ve kieu so (float)
# errors='coerce': gia tri khong phai so => chuyen thanh NaN, khong bao loi
df['Vatli']     = pd.to_numeric(df['Vatli'],     errors='coerce')
df['Tinhthanh'] = pd.to_numeric(df['Tinhthanh'], errors='coerce')

# drop_duplicates(subset='Ma'): xoa dong trung theo ma thi sinh
# .copy(): tao ban sao de tranh loi SettingWithCopyWarning khi them cot moi
df = df.drop_duplicates(subset='Ma').copy()

# ================================================================
# TRICH XUAT DU LIEU THI SINH TP. HO CHI MINH
# (Ma vung dien thoai: 2.0)
# ================================================================

# Loc cac dong co Tinhthanh == 2.0 (TP.HCM)
# .copy(): tao ban sao rieng de thao tac an toan, khong anh huong df goc
df_hcm = df[df['Tinhthanh'] == 2.0].copy()

print(f"Tong so thi sinh TP. HCM: {len(df_hcm)}")

# ================================================================
# CHUAN BI DU LIEU CHO XEP LOAI
# ================================================================

# Xac dinh diem bai thi to hop ma thi sinh dang ky:
#   - Neu KHTN khong phai NaN => hoc sinh dang ky thi KHTN
#   - Neu KHTN la NaN => lay diem KHXH thay the (fillna)
# .fillna(df_hcm['KHXH']): dien NaN cua KHTN bang gia tri tuong ung tu KHXH
df_hcm['KH_chon'] = df_hcm['KHTN'].fillna(df_hcm['KHXH'])

# Cac mon thanh phan cua bai thi to hop KHTN
# Dung de kiem tra dieu kien: tat ca mon thanh phan phai > 1 diem
khtn_thanh_phan = ['Vatli', 'Hoahoc', 'Sinhhoc']

# Cac mon thanh phan cua bai thi to hop KHXH
khxh_thanh_phan = ['Diali', 'LichSu', 'GDCD']

# 3 bai thi bat buoc (khong thuoc to hop)
bat_buoc = ['Toan', 'Nguvan', 'Ngoaingu']

# ================================================================
# HAM XEP LOAI TOT NGHIEP
# Ham nay se duoc ap dung cho tung dong (thi sinh) bang .apply()
# ================================================================
def xep_loai_tot_nghiep(row):
    """
    Nhan vao 1 dong du lieu (1 thi sinh), tra ve xep loai tot nghiep.

    Quy tac xep loai theo quy che:
      - Diem xep loai = Tong 4 bai thi / 4
      - Dieu kien tot nghiep:
          + Tat ca bai thi va mon thanh phan deu > 1 diem
          + Diem xep loai >= 5
      - Loai gioi  : diem xl >= 8.0 va khong bai thi nao < 7.0
      - Loai kha   : diem xl >= 6.5 va khong bai thi nao < 6.0
      - Loai trung binh: cac truong hop con lai khi da tot nghiep
      - Khong tot nghiep: khong thoa man dieu kien tren
    """

    # --- Buoc 1: Lay diem 4 bai thi chinh ---
    # row['Toan'], row['Nguvan'], row['Ngoaingu']: diem 3 bai thi bat buoc
    # row['KH_chon']: diem bai thi to hop (KHTN hoac KHXH)
    bai_thi = [row['Toan'], row['Nguvan'], row['Ngoaingu'], row['KH_chon']]

    # pd.isna(): kiem tra xem gia tri co phai NaN hay khong (True = NaN)
    # any(): tra ve True neu it nhat 1 phan tu la True
    # Neu thieu bat ky bai thi nao => khong du dieu kien xet tot nghiep
    if any(pd.isna(s) for s in bai_thi):
        return 'Khong tot nghiep'

    # --- Buoc 2: Xac dinh mon thanh phan dang ky ---
    # pd.notna(row['KHTN']): True neu cot KHTN co gia tri (khong phai NaN)
    # => xac dinh hoc sinh thi KHTN hay KHXH
    if pd.notna(row['KHTN']):
        # Lay diem cac mon thanh phan KHTN (Vatli, Hoahoc, Sinhhoc)
        # dropna() loc bo NaN de chi kiem tra nhung mon co diem thuc su
        thanh_phan = [row[m] for m in khtn_thanh_phan if pd.notna(row[m])]
    else:
        # Lay diem cac mon thanh phan KHXH (Diali, LichSu, GDCD)
        thanh_phan = [row[m] for m in khxh_thanh_phan if pd.notna(row[m])]

    # --- Buoc 3: Kiem tra dieu kien > 1 diem ---
    # Gop tat ca diem lai: 4 bai thi chinh + cac mon thanh phan to hop
    tat_ca_diem = bai_thi + thanh_phan

    # any(s <= 1 for s in ...): True neu co bat ky diem nao <= 1
    # Neu co => khong du dieu kien tot nghiep
    if any(s <= 1 for s in tat_ca_diem):
        return 'Khong tot nghiep'

    # --- Buoc 4: Tinh diem xep loai ---
    # Theo quy che: diem xep loai = tong 4 bai thi / 4
    # sum(): tinh tong, len(): dem so luong
    diem_xl = sum(bai_thi) / len(bai_thi)

    # Dieu kien co ban de tot nghiep: diem xep loai >= 5
    if diem_xl < 5:
        return 'Khong tot nghiep'

    # --- Buoc 5: Phan loai tot nghiep ---

    # Loai GIOI: diem xl >= 8.0 VA khong co bai thi nao < 7.0
    # all(s >= 7.0 for s in bai_thi): True neu TAT CA bai thi >= 7.0
    if diem_xl >= 8.0 and all(s >= 7.0 for s in bai_thi):
        return 'Gioi'

    # Loai KHA: diem xl >= 6.5 VA khong co bai thi nao < 6.0
    if diem_xl >= 6.5 and all(s >= 6.0 for s in bai_thi):
        return 'Kha'

    # Loai TRUNG BINH: cac truong hop con lai (da tot nghiep nhung khong dat kha/gioi)
    return 'Trung binh'


# ================================================================
# AP DUNG HAM XEP LOAI CHO TUNG THI SINH
# ================================================================

# .apply(func, axis=1): ap dung ham 'xep_loai_tot_nghiep' len tung HANG (thi sinh)
# axis=1: xu ly theo chieu ngang (moi lan truyen vao 1 dong - 1 Series)
# Ket qua tra ve 1 Series chua xep loai cua tung thi sinh
# Gan vao cot moi 'Xep_loai' trong DataFrame df_hcm
print("Dang xu ly xep loai... (co the mat vai giay)")
df_hcm['Xep_loai'] = df_hcm.apply(xep_loai_tot_nghiep, axis=1)

# ================================================================
# IN KET QUA SO LUONG TUNG XEP LOAI
# ================================================================

# .value_counts(): dem so luong thi sinh trong moi loai xep loai
# Tra ve Series: index = ten xep loai, values = so luong
ket_qua = df_hcm['Xep_loai'].value_counts()

print("\n=== KET QUA XEP LOAI TOT NGHIEP TP. HO CHI MINH ===")

# Dinh nghia thu tu hien thi de in ra theo thu tu co y nghia
# reindex([...]): sap xep lai Series theo thu tu cho truoc
# fill_value=0: neu loai nao khong co thi sinh, dat gia tri 0 thay vi NaN
thu_tu = ['Gioi', 'Kha', 'Trung binh', 'Khong tot nghiep']
ket_qua_sapxep = ket_qua.reindex(thu_tu, fill_value=0)

# Lap qua tung cap (ten_loai, so_luong) de in ra
for ten_loai, so_luong in ket_qua_sapxep.items():
    # f-string voi :>12,: can phai, dinh dang co dau phan cach hang nghin
    print(f"  {ten_loai:<20}: {so_luong:>10,} thi sinh")

# Tinh tong so thi sinh tot nghiep (3 loai dau)
tot_nghiep = ket_qua_sapxep[['Gioi','Kha','Trung binh']].sum()
khong_tot_nghiep = ket_qua_sapxep['Khong tot nghiep']
print(f"\n  => Tong tot nghiep      : {tot_nghiep:>10,} thi sinh")
print(f"  => Tong khong tot nghiep: {khong_tot_nghiep:>10,} thi sinh")

# ================================================================
# VE BIEU DO
# ================================================================

# Dinh nghia nhan, mau sac va gia tri cho bieu do
labels = ['Gioi', 'Kha', 'Trung binh', 'Khong\ntot nghiep']

# Lay gia tri tuong ung tu Series da sap xep
values = ket_qua_sapxep.values

# Danh sach mau sac cho tung cot
# Xanh la: Gioi, Xanh duong: Kha, Vang: Trung binh, Do: Khong tot nghiep
colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']

# --- Bieu do 1: Bar chart (Bieu do cot) ---
# plt.subplots(): tao Figure va Axes (vung ve)
fig1, ax1 = plt.subplots(figsize=(10, 6))

# ax1.bar(): ve bieu do cot
# range(len(labels)): vi tri cac cot tren truc X (0, 1, 2, 3)
bars = ax1.bar(range(len(labels)), values, color=colors,
               edgecolor='black', width=0.6)

# ax1.set_xticks(): dat vi tri cac vach tren truc X
# ax1.set_xticklabels(): dat nhan chu cho cac vach do
ax1.set_xticks(range(len(labels)))
ax1.set_xticklabels(labels, fontsize=12)

ax1.set_title('Xep loai tot nghiep THPT - TP. Ho Chi Minh (2020)',
              fontsize=14, fontweight='bold')
ax1.set_ylabel('So luong thi sinh', fontsize=12)
ax1.set_xlabel('Xep loai', fontsize=12)

# ax1.grid(): ve duong luoi ngang de de so sanh gia tri
ax1.grid(axis='y', linestyle='--', alpha=0.6)

# Gan nhan so luong tren dinh moi cot
for bar, val in zip(bars, values):
    # bar.get_x(): lay toa do X goc trai cua cot
    # bar.get_width(): lay do rong cot
    # => Cong lai chia 2 de tim chinh giua cot
    x_center = bar.get_x() + bar.get_width() / 2

    # bar.get_height(): chieu cao cot = gia tri du lieu
    # ax1.text(): dat chu len toa do (x, y) tren bieu do
    # ha='center': can giua theo chieu ngang; va='bottom': dat phia tren dinh cot
    ax1.text(x_center, bar.get_height() + 200, f'{val:,}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# plt.tight_layout(): tu can chinh de khong bi cat mat chu
plt.tight_layout()
plt.savefig('xep_loai_HCM_bar.png')
plt.show()
print("Da luu: xep_loai_HCM_bar.png")

# --- Bieu do 2: Pie chart (Bieu do tron) ---
fig2, ax2 = plt.subplots(figsize=(8, 8))

# ax2.pie(): ve bieu do tron
# values: gia tri cac phan; labels: nhan; colors: mau sac
# autopct='%1.1f%%': hien thi % voi 1 chu so thap phan tren moi manh
# startangle=90: bat dau tu 12 gio (goc 90 do)
# pctdistance=0.75: khoang cach cua nhan % tinh tu tam
wedges, texts, autotexts = ax2.pie(
    values,
    labels=labels,
    colors=colors,
    autopct='%1.1f%%',
    startangle=90,
    pctdistance=0.75,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}  # vien trang giua cac manh
)

# Dinh dang chu trong bieu do tron
for text in texts:
    text.set_fontsize(12)
for autotext in autotexts:
    autotext.set_fontsize(11)
    autotext.set_fontweight('bold')

ax2.set_title('Ti le xep loai tot nghiep THPT - TP. Ho Chi Minh (2020)',
              fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('xep_loai_HCM_pie.png')
plt.show()
print("Da luu: xep_loai_HCM_pie.png")

# ================================================================
# NHAN XET
# ================================================================
print("\n=== NHAN XET ===")
tong = len(df_hcm)
pct_gioi    = ket_qua_sapxep['Gioi']        / tong * 100
pct_kha     = ket_qua_sapxep['Kha']         / tong * 100
pct_tb      = ket_qua_sapxep['Trung binh']  / tong * 100
pct_khong   = ket_qua_sapxep['Khong tot nghiep'] / tong * 100
pct_tot     = (tong - ket_qua_sapxep['Khong tot nghiep']) / tong * 100

# f-string voi :.1f: lam tron 1 chu so thap phan
print(f"- Ti le tot nghiep     : {pct_tot:.1f}%")
print(f"- Ti le khong tot nghiep: {pct_khong:.1f}%")
print(f"- Trong so tot nghiep:")
print(f"    + Loai Gioi     : {pct_gioi:.1f}%")
print(f"    + Loai Kha      : {pct_kha:.1f}%")
print(f"    + Loai Trung binh: {pct_tb:.1f}%")
