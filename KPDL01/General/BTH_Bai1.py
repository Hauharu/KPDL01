import pandas as pd

df1 = pd.read_excel("DSSV1.xlsx")

df1.rename(columns={'Ho5': 'HoLot'}, inplace=True)
df1 = df1.fillna(0)
df1['Ngay'] = df1['Ngay'].astype(int)
df1['Thang'] = df1['Thang'].astype(int)
df1['Nam'] = df1['Nam'].astype(int)

##########################################################################################################

df2 = pd.read_excel("DSSV2.xlsx")
df2.columns = ['STT', 'MSSV', 'HoLot', 'Ten', 'yyyy-mm-dd']

df2[['Nam', 'Thang', 'Ngay']] = df2['yyyy-mm-dd'].str.split('-', expand=True)
df2 = df2.fillna(0)
df2['Ngay'] = df2['Ngay'].astype(int)
df2['Thang'] = df2['Thang'].astype(int)
df2['Nam'] = df2['Nam'].astype(int)

df2 = df2.drop('yyyy-mm-dd', axis=1)

##########################################################################################################

df3 = pd.read_excel("DSSV3.xlsx")
df3.rename(columns={'Unnamed: 0': 'STT', 'Unnamed: 1': 'MSSV', 'Unnamed: 2': 'HoLot', 'Unnamed: 3': 'Ten'},
           inplace=True)

df3[['Ngay', 'Thang', 'Nam']] = df3['dd/mm/yyyy'].str.split('/', expand=True)
df3 = df3.fillna(0)
df3['Ngay'] = df3['Ngay'].astype(int)
df3['Thang'] = df3['Thang'].astype(int)
df3['Nam'] = df3['Nam'].astype(int)
df3 = df3.drop('dd/mm/yyyy', axis=1)

##########################################################################################################

df4 = pd.read_excel("DSSV4.xlsx")
df4.columns = ['STT', 'MSSV', 'HoLot', 'Ten', 'dd-mm-yyyy']
df4[['Ngay', 'Thang', 'Nam']] = df4['dd-mm-yyyy'].str.split('-', expand=True)
df4 = df4.fillna(0)
df4['Ngay'] = df4['Ngay'].astype(int)
df4['Thang'] = df4['Thang'].astype(int)
df4['Nam'] = df4['Nam'].astype(int)
df4 = df4.drop('dd-mm-yyyy', axis=1)

columns_order = ['STT', 'MSSV', 'HoLot', 'Ten', 'Ngay', 'Thang', 'Nam']

df1 = df1[columns_order]
df2 = df2[columns_order]
df3 = df3[columns_order]
df4 = df4[columns_order]

df = pd.concat([df1, df2, df3, df4], ignore_index=True)
df = df.drop('STT', axis=1)

replacements = {
    'Ngũyn': 'Nguyễn',
    'Ngũyên': 'Nguyễn',
    'Nguyn': 'Nguyễn',
    'Đon': 'Đoàn',
    'Thnh': 'Thành',
    'Hong': 'Hoàng',
    'Phc': 'Phúc',
    'Đnh': 'Đình',
    'Thin': 'Thiên',
    'Ton': 'Toàn',
    'Bi': 'Bùi',
    'Qun': 'Quân',
    'Cng': 'Công'
}

for wrong, right in replacements.items():
    df['HoLot'] = df['HoLot'].astype(str).str.replace(wrong, right)
    df['Ten'] = df['Ten'].astype(str).str.replace(wrong, right)

def is_valid_date(ngay, thang, nam):
    import datetime
    try:
        datetime.datetime(int(nam), int(thang), int(ngay))
        return True
    except Exception:
        raise Exception("False")

def is_valid_name(mssv, holot, ten):
    if str(mssv) == '0' or str(holot) == '0' or str(ten) == '0':
        return False
    return True

valid_rows = []
for index, row in df.iterrows():
    is_valid = True
    
    # Kiểm tra tên, MSSV
    if not is_valid_name(row['MSSV'], row['HoLot'], row['Ten']):
        is_valid = False
    
    # Kiểm tra ngày
    if is_valid:
        try:
            is_valid_date(row['Ngay'], row['Thang'], row['Nam'])
        except Exception:
            is_valid = False

    valid_rows.append(is_valid)

df['is_valid'] = valid_rows

df_valid = df[df['is_valid'] == True].drop(columns=['is_valid'])
df_invalid = df[df['is_valid'] == False].drop(columns=['is_valid'])

df_valid.to_excel("DSSV_HopLe.xlsx", index=False)
df_invalid.to_excel("DSSV_KhongHopLe.xlsx", index=False)

print(f"Số SV hợp lệ: {len(df_valid)}")
print("Danh sách SV hợp lệ:")
print(df_valid.to_string(index=False))

print(f"\nSố SV không hợp lệ: {len(df_invalid)}")
print("Danh sách SV không hợp lệ:")
print(df_invalid.to_string(index=False))

# =====================================================================

import datetime
today = datetime.date.today()

def tinh_tuoi_chinh_xac(row):
    try:
        ngaysinh = datetime.date(int(row['Nam']), int(row['Thang']), int(row['Ngay']))
        tuoi = today.year - ngaysinh.year
        # Nếu chưa qua sinh nhật trong năm nay thì trừ thêm 1
        if (today.month, today.day) < (ngaysinh.month, ngaysinh.day):
            tuoi -= 1
        return tuoi
    except:
        return None

df_stats = df_valid.copy()
df_stats['Tuoi'] = df_stats.apply(tinh_tuoi_chinh_xac, axis=1)

# a. Thông tin người có tuổi cao nhất, thấp nhất
df_stats['NgaySinh'] = pd.to_datetime(
    df_stats[['Nam', 'Thang', 'Ngay']].rename(columns={'Nam': 'year', 'Thang': 'month', 'Ngay': 'day'})
)
df_sorted = df_stats.sort_values('NgaySinh')

# Lấy đúng 1 người sinh sớm nhất (lớn tuổi nhất) và 1 người sinh muộn nhất (trẻ nhất)
nguoi_cao_tuoi_nhat = df_sorted.head(1).drop(columns=['NgaySinh'])
nguoi_thap_tuoi_nhat = df_sorted.tail(1).drop(columns=['NgaySinh'])

print("\na. Thông tin người có tuổi cao nhất:")
print(nguoi_cao_tuoi_nhat.to_string(index=False))

print("\n   Thông tin người có tuổi thấp nhất:")
print(nguoi_thap_tuoi_nhat.to_string(index=False))

# b. Độ tuổi trung bình
mean_age = df_stats['Tuoi'].mean()
print(f"\nb. Độ tuổi trung bình của danh sách hợp lệ: {mean_age:.2f}")

# c. Số lượng theo từng độ tuổi
print("\nc. Số lượng sinh viên theo từng độ tuổi:")
age_counts = df_stats['Tuoi'].value_counts().sort_index()
for age, count in age_counts.items():
    print(f"   - Tuổi {age}: {count} sinh viên")

# d. Giá trị Mod và Median của danh sách
mode_age = df_stats['Tuoi'].mode().tolist()
median_age = df_stats['Tuoi'].median()
print(f"\nd. Giá trị Mode (Mod) của độ tuổi: {mode_age}")
print(f"   Giá trị Median của độ tuổi: {median_age}")

# =====================================================================
pivot = df_valid.pivot_table(index='Nam', columns='Thang', aggfunc='size', fill_value=0)

all_months = list(range(1, 13))
pivot = pivot.reindex(columns=all_months, fill_value=0)
pivot.columns = [f"T{m}" for m in all_months]

pivot['Tổng theo cột'] = pivot.sum(axis=1)

tong_hang = pivot.sum(axis=0)
tong_hang.name = 'Tổng theo hàng'
pivot = pd.concat([pivot, tong_hang.to_frame().T])

pivot.index.name = 'Năm/Tháng'
print(pivot.to_string())

# =====================================================================
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import matplotlib.ticker as ticker

pivot_plot = df_valid.pivot_table(index='Nam', columns='Thang', aggfunc='size', fill_value=0)
pivot_plot = pivot_plot.reindex(columns=list(range(1, 13)), fill_value=0)

fig, ax = plt.subplots(figsize=(14, 6))
colors = plt.cm.tab10.colors

for i, (nam, row) in enumerate(pivot_plot.iterrows()):
    ax.plot(range(1, 13), row.values,
            marker='o', linewidth=2, markersize=6,
            label=str(int(nam)), color=colors[i % len(colors)])

ax.set_title('Số lượng sinh viên sinh vào tháng T theo từng năm N',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Tháng', fontsize=12)
ax.set_ylabel('Số lượng sinh viên', fontsize=12)
ax.set_xticks(range(1, 13))
ax.set_xticklabels([f'T{m}' for m in range(1, 13)])
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.legend(title='Năm sinh', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('bieudo_sinh_theo_thang_nam.png', dpi=150, bbox_inches='tight')
plt.show()
print("Đã lưu biểu đồ: bieudo_sinh_theo_thang_nam.png")

# =====================================================================
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

pivot_3d = df_valid.pivot_table(index='Nam', columns='Thang', aggfunc='size', fill_value=0)
pivot_3d = pivot_3d.reindex(columns=list(range(1, 13)), fill_value=0)

years  = pivot_3d.index.tolist()        # các năm
months = list(range(1, 13))            # tháng 1-12

xpos_all, ypos_all, zpos_all = [], [], []
dz_all = []

for i, nam in enumerate(years):
    for j, thang in enumerate(months):
        xpos_all.append(i)
        ypos_all.append(j)
        zpos_all.append(0)
        dz_all.append(pivot_3d.loc[nam, thang])

xpos_all = np.array(xpos_all)
ypos_all = np.array(ypos_all)
zpos_all = np.array(zpos_all)
dz_all   = np.array(dz_all, dtype=float)
dx = dy  = 0.6

# Màu theo từng năm
cmap = plt.cm.tab10
colors_3d = [cmap(i / len(years)) for i in range(len(years)) for _ in months]

fig3d = plt.figure(figsize=(16, 9))
ax3d  = fig3d.add_subplot(111, projection='3d')

ax3d.bar3d(xpos_all, ypos_all, zpos_all, dx, dy, dz_all,
           color=colors_3d, alpha=0.85, zsort='average')

ax3d.set_xticks(np.arange(len(years)) + dx/2)
ax3d.set_xticklabels([str(y) for y in years], rotation=30, fontsize=9)
ax3d.set_yticks(np.arange(len(months)) + dy/2)
ax3d.set_yticklabels([f'T{m}' for m in months], fontsize=9)

ax3d.set_xlabel('Năm (N)', fontsize=11, labelpad=10)
ax3d.set_ylabel('Tháng (T)', fontsize=11, labelpad=10)
ax3d.set_zlabel('Số lượng SV', fontsize=11, labelpad=8)
ax3d.set_title('Số lượng sinh viên sinh vào tháng T của năm N (3D)',
               fontsize=13, fontweight='bold', pad=20)

# Chú thích màu theo năm
handles = [plt.Rectangle((0,0),1,1, color=cmap(i/len(years))) for i in range(len(years))]
ax3d.legend(handles, [str(y) for y in years], title='Năm',
            loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=9)

plt.tight_layout()
plt.savefig('bieudo_3d_thang_nam.png', dpi=150, bbox_inches='tight')
plt.show()
print("Đã lưu biểu đồ 3D: bieudo_3d_thang_nam.png")
