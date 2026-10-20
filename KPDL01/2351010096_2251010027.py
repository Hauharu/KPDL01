import pandas as pd
import numpy as np

MA_NHOM = 2251010027
np.random.seed(MA_NHOM)
n = 500

# 1. TẠO DANH SÁCH MSSV (500 MÃ)
mssv = [f'SV{MA_NHOM - i}' for i in range(n)]

# 2. HỌ TÊN: Dính khoảng trắng thừa 2 đầu
# Xử lý: df['Ho_Ten'] = df['Ho_Ten'].str.strip()
ho = ['Nguyen', 'Tran', 'Le', 'Pham', 'Hoang', 'Huynh', 'Phan', 'Vu', 'Vo', 'Dang']
td = ['Van', 'Thi', 'Minh', 'Hoang', 'Duc', 'Thu', 'Anh', 'Quoc', '']
ten = ['An', 'Binh', 'Cuong', 'Dung', 'Em', 'Phong', 'Giang', 'Hai', 'Khanh', 'Lan',
       'Mai', 'Nam', 'Oanh', 'Phuc', 'Quyen', 'Son', 'Tam', 'Uyen', 'Vinh', 'Yen']

ho_ten = []
for _ in range(n):
    name = f'{np.random.choice(ho)} {np.random.choice(td)} {np.random.choice(ten)}'
    name = name.replace('  ', ' ').strip()
    if np.random.rand() < 0.35:
        name = ' ' + name + '  '
    ho_ten.append(name)

# 3. TUỔI: Khuyết NaN (~8%), ngoại lai (-12, 0, 195, 999)
# Xử lý: Điền median df['Tuoi'].fillna(df['Tuoi'].median()), lọc 18 <= Tuoi <= 30
tuoi = np.random.randint(18, 31, size=n).astype(float)
idx = np.random.choice(n, size=int(n * 0.08), replace=False)
tuoi[idx] = np.nan

out = np.random.choice([i for i in range(n) if i not in idx], size=6, replace=False)
tuoi[out[:2]] = -12
tuoi[out[2:4]] = 195
tuoi[out[4]] = 0
tuoi[out[5]] = 999

# 4. ĐIỂM SỐ: Khuyết NaN (~7%), lỗi thang điểm (-2, -1, 11.5, 95, 100)
# Xử lý: Điền mean df['Diem_So'].fillna(df['Diem_So'].mean()), lọc 0 <= Diem_So <= 10
diem = np.round(np.random.uniform(0, 10, size=n), 1)
idx2 = np.random.choice(n, size=int(n * 0.07), replace=False)
diem[idx2] = np.nan

err = np.random.choice([i for i in range(n) if i not in idx2], size=5, replace=False)
diem[err] = [-2.0, 95.0, 11.5, -1.0, 100.0][:5]

# 5. TỈNH THÀNH: Viết tắt, viết thường, có dấu ('HN', 'ha noi', 'tp.hcm', 'Đà Nẵng')
# Xử lý: df['Tinh_Thanh'].replace({'HN': 'Ha Noi', 'ha noi': 'Ha Noi', 'HCM': 'TP.HCM', 'tp.hcm': 'TP.HCM', 'Đà Nẵng': 'Da Nang'})
tinh = ['Ha Noi', 'HN', 'TP.HCM', 'HCM', 'Da Nang', 'Can Tho', 'Hai Phong',
        'ha noi', 'tp.hcm', 'Đà Nẵng']
tinh_list = [np.random.choice(tinh) for _ in range(n)]

# 6. EMAIL: Mất '@', viết hoa, khoảng trắng thừa, chuỗi rỗng ''
# Xử lý: df['Email'] = df['MSSV'].str.lower() + '@st.edu.vn'
emails = []
for i in range(n):
    e = f'sv{MA_NHOM - i}@st.edu.vn'
    r = np.random.rand()
    if r < 0.10:
        e = e.replace('@', '')
    elif r < 0.18:
        e = e.upper()
    elif r < 0.25:
        e = ' ' + e + ' '
    elif r < 0.30:
        e = ''
    emails.append(e)

# 7. NGÀY SINH: Trộn 3 định dạng ('%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y'), chuỗi rỗng ''
# Xử lý: pd.to_datetime(df['Ngay_Sinh'].replace('', np.nan), format='mixed', dayfirst=True)
ngays = []
for i in range(n):
    y = 2006 - (i % 4)
    m = (i % 12) + 1
    d = (i % 28) + 1
    fmt = np.random.choice(['%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y'])
    s = pd.Timestamp(year=y, month=m, day=d).strftime(fmt)
    if np.random.rand() < 0.08:
        s = ''
    ngays.append(s)

# 8. ĐÓNG GÓI DATAFRAME
df = pd.DataFrame({
    'MSSV': mssv,
    'Ho_Ten': ho_ten,
    'Tuoi': tuoi,
    'Diem_So': diem,
    'Tinh_Thanh': tinh_list,
    'Email': emails,
    'Ngay_Sinh': ngays
})

# 9. NHÂN BẢN 12 DÒNG TRÙNG LẶP
# Xử lý: df.drop_duplicates(inplace=True)
dups = df.sample(12, random_state=MA_NHOM)
df = pd.concat([df, dups], ignore_index=True)

# 10. XUẤT TỆP DỮ LIỆU THÔ CHỨA LỖI
raw_filename = f'data_nhom_{MA_NHOM}_raw.csv'
df.to_csv(raw_filename, index=False, encoding='utf-8')
print(f'Dữ liệu thô đã lưu: {raw_filename}, số dòng: {len(df)}')

# 11. DỮ LIỆU SẠCH (điền giá trị thiếu, chuẩn hoá, loại bỏ ngoại lệ)
orig_rows = len(df)

# Clean Ho_Ten: strip whitespace
df['Ho_Ten'] = df['Ho_Ten'].str.strip()
print('Đã làm sạch cột Ho_Ten: loại bỏ khoảng trắng thừa')

# Clean Tuoi: fill NaN with median, replace outliers (<18 or >30) with median
median_age = df.loc[(df['Tuoi'] >= 18) & (df['Tuoi'] <= 30), 'Tuoi'].median()
df['Tuoi'] = df['Tuoi'].fillna(median_age)
df.loc[(df['Tuoi'] < 18) | (df['Tuoi'] > 30), 'Tuoi'] = median_age
print('Đã làm sạch cột Tuổi: điền giá trị trung vị cho NaN và thay thế giá trị bất thường (<18 hoặc >30) bằng trung vị')

# Clean Diem_So: fill NaN with mean, replace outliers (<0 or >10) with mean
mean_score = df.loc[(df['Diem_So'] >= 0) & (df['Diem_So'] <= 10), 'Diem_So'].mean()
df['Diem_So'] = df['Diem_So'].fillna(mean_score)
df.loc[(df['Diem_So'] < 0) | (df['Diem_So'] > 10), 'Diem_So'] = mean_score
print('Đã làm sạch cột Điểm: điền giá trị trung bình cho NaN và thay thế giá trị bất thường (<0 hoặc >10) bằng trung bình')

# Clean Tinh_Thanh: chuẩn hoá tên tỉnh/thành phố
df['Tinh_Thanh'] = df['Tinh_Thanh'].replace({
    'HN': 'Ha Noi', 'ha noi': 'Ha Noi',
    'HCM': 'TP.HCM', 'tp.hcm': 'TP.HCM',
    'Đà Nẵng': 'Da Nang'
})
print('Đã chuẩn hoá cột Tỉnh/Thành phố: thống nhất cách viết')

# Clean Email: loại bỏ khoảng trắng, thiếu '@', viết thường, thay thế rỗng
df['Email'] = df['Email'].str.strip()
df['Email'] = df['Email'].replace('', np.nan)
df['Email'] = df['Email'].fillna(df['MSSV'].str.lower() + '@st.edu.vn')
df['Email'] = df['Email'].str.lower()
print('Đã làm sạch cột Email: loại bỏ khoảng trắng, xử lý thiếu @, chuyển về chữ thường và điền email mặc định nếu rỗng')

# Clean Ngay_Sinh: chuyển rỗng thành NaT, chuyển sang datetime, thay thế NaT bằng ngày mặc định
df['Ngay_Sinh'] = pd.to_datetime(df['Ngay_Sinh'].replace('', np.nan), errors='coerce', dayfirst=True)
df['Ngay_Sinh'] = df['Ngay_Sinh'].fillna(pd.Timestamp('2000-01-01'))
print('Đã chuẩn hoá cột Ngày Sinh: chuyển chuỗi rỗng thành NaT, chuyển sang datetime và điền ngày mặc định 2000-01-01 cho NaT')

# Xóa các dòng trùng lặp (đã thêm 12 dòng trùng ở trên)
df.drop_duplicates(inplace=True)
print('Đã xóa các dòng trùng lặp, bao gồm 12 dòng được sao chép thêm ở trên')

clean_rows = len(df)
clean_filename = f'data_nhom_{MA_NHOM}_clean.csv'
df.to_csv(clean_filename, index=False, encoding='utf-8')
print(f'Dữ liệu đã làm sạch được lưu: {clean_filename}, số dòng trước: {orig_rows}, sau: {clean_rows}')
