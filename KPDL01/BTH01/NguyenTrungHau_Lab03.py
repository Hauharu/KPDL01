import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

df = pd.read_csv('diem_thi_thpt_2024.csv', dtype={'sbd': str})

print("TỔNG SỐ HỌC SINH THAM GIA KỲ THI")
print(f"Tổng số học sinh: {len(df)}")

df['ma_tinh'] = df['sbd'].str[:2]
city_dict = {
    '01': 'Hà Nội',
    '03': 'Hải Phòng',
    '04': 'Đà Nẵng',
    '02': 'TpHCM',
    '55': 'Cần Thơ'
}
print("\nSỐ LƯỢNG THÍ SINH CỦA 5 THÀNH PHỐ")
for code, name in city_dict.items():
    count = (df['ma_tinh'] == code).sum()
    print(f"- {name:<12} (Mã {code}): {count} thí sinh")

print("\nĐIỂM MEAN, MOD, MEDIAN CẢ NƯỚC (TOÁN, VĂN, NGOẠI NGỮ)")
subjects = [('toan', 'Toán'), ('ngu_van', 'Ngữ văn'), ('ngoai_ngu', 'Ngoại ngữ')]
for col, name in subjects:
    s = df[col].dropna()
    mean_val = s.mean()
    mod_val = s.mode()[0]
    median_val = s.median()
    print(f"- Môn {name:<10}: Mean = {mean_val:.2f} | Mod = {mod_val:.2f} | Median = {median_val:.2f}")

df_hcm = df[df['ma_tinh'] == '02'].copy()
print(f"\nDANH SÁCH THÍ SINH TPHCM: {len(df_hcm):,} HỌC SINH")

plt.figure(figsize=(10, 5))
sns.histplot(df_hcm['toan'].dropna(), bins=50, color='royalblue', kde=True)
plt.title('Phổ điểm môn Toán - Học sinh TpHCM (THPT 2024)', fontsize=13, fontweight='bold')
plt.xlabel('Điểm Toán')
plt.ylabel('Số lượng thí sinh')
plt.tight_layout()
plt.savefig('pho_diem_toan_tphcm.png', dpi=300)
plt.show()

df_hcm_khtn = df_hcm.dropna(subset=['vat_li', 'hoa_hoc', 'sinh_hoc']).copy()
df_hcm_khtn['khtn'] = df_hcm_khtn['vat_li'] + df_hcm_khtn['hoa_hoc'] + df_hcm_khtn['sinh_hoc']
print(f"- Số lượng thí sinh TpHCM thi tổ hợp KHTN: {len(df_hcm_khtn):,}")

plt.figure(figsize=(10, 5))
sns.histplot(df_hcm_khtn['khtn'], bins=60, color='seagreen', kde=True)
plt.title('Phổ điểm tổ hợp KHTN (Lý + Hóa + Sinh) - Học sinh TpHCM (THPT 2024)', fontsize=13, fontweight='bold')
plt.xlabel('Tổng điểm KHTN (Thang 30)')
plt.ylabel('Số lượng thí sinh')
plt.tight_layout()
plt.savefig('pho_diem_khtn_tphcm.png', dpi=300)
plt.show()

s_toan_hcm = df_hcm['toan'].dropna()
s_khtn_hcm = df_hcm_khtn['khtn']
print("\nTHỐNG KÊ MEAN, MOD, MEDIAN TẠI TPHCM")
print(f"- Điểm Toán : Mean = {s_toan_hcm.mean():.2f} | Mod = {s_toan_hcm.mode()[0]:.2f} | Median = {s_toan_hcm.median():.2f}")
print(f"- Điểm KHTN : Mean = {s_khtn_hcm.mean():.2f} | Mod = {s_khtn_hcm.mode()[0]:.2f} | Median = {s_khtn_hcm.median():.2f}")

df_toan_ly = df_hcm.dropna(subset=['toan', 'vat_li'])
corr = df_toan_ly['toan'].corr(df_toan_ly['vat_li'])
print("\nKIỂM ĐỊNH: 'HỌC GIỎI TOÁN THÌ CŨNG GIỎI LÝ")
print(f"- Hệ số tương quan Pearson (Toán - Lý, TpHCM): r = {corr:.4f}")

gioi_toan = df_toan_ly[df_toan_ly['toan'] >= 8.0]
gioi_toan_ly = gioi_toan[gioi_toan['vat_li'] >= 8.0]
kha_ly = gioi_toan[gioi_toan['vat_li'] >= 6.5]

print(f"- Số học sinh giỏi Toán (>= 8.0): {len(gioi_toan):,}")
print(f"- Trong đó số học sinh giỏi Lý (>= 8.0): {len(gioi_toan_ly):,} ({len(gioi_toan_ly)/len(gioi_toan)*100:.2f}%)")
print(f"- Trong đó số học sinh đạt Lý từ khá trở lên (>= 6.5): {len(kha_ly):,} ({len(kha_ly)/len(gioi_toan)*100:.2f}%)")
print("=> KẾT LUẬN: Bác bỏ nhận định nếu hiểu theo nghĩa tuyệt đối (vì chỉ có 37.5% học sinh giỏi Toán đạt điểm giỏi Lý);")
print("   nhưng ủng hộ về mặt xu hướng (hai môn tương quan thuận r = 0.55, có 76.6% học sinh giỏi Toán đạt điểm Lý từ khá trở lên).")

plt.figure(figsize=(9, 6))
sample_tl = df_toan_ly.sample(n=min(5000, len(df_toan_ly)), random_state=42)
sns.regplot(x='toan', y='vat_li', data=sample_tl, color='purple', scatter_kws={'alpha': 0.25}, line_kws={'color': 'red'})
plt.title('Tương quan giữa điểm Toán và điểm Vật lý (Học sinh TpHCM)', fontsize=13, fontweight='bold')
plt.xlabel('Điểm Toán')
plt.ylabel('Điểm Vật lý')
plt.tight_layout()
plt.savefig('tuong_quan_toan_ly_tphcm.png', dpi=300)
plt.show()

s_toan_all = df['toan'].dropna()
print("\nSO SÁNH KHẢ NĂNG HỌC TOÁN TPHCM VÀ CẢ NƯỚC")
print(f"- Cả nước : Mean = {s_toan_all.mean():.2f} | Mod = {s_toan_all.mode()[0]:.2f} | Median = {s_toan_all.median():.2f}")
print(f"- TpHCM   : Mean = {s_toan_hcm.mean():.2f} | Mod = {s_toan_hcm.mode()[0]:.2f} | Median = {s_toan_hcm.median():.2f}")
print(f"=> Điểm trung bình Toán TpHCM cao hơn cả nước {s_toan_hcm.mean() - s_toan_all.mean():.2f} điểm.")
print("=> Nhận xét: Học sinh TpHCM có phổ điểm và học lực môn Toán phân bố cao và đồng đều hơn so với mặt bằng chung cả nước.")