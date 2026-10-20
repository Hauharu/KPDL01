import pandas as pd  # Thư viện chuyên thao tác, xử lý và phân tích bảng dữ liệu (DataFrame)
import numpy as np   # Thư viện tính toán số học, sinh số ngẫu nhiên và xử lý mảng (Array)

# ------------------------------------------------------------------------------
# CẤU HÌNH THÔNG TIN BAN ĐẦU
# ------------------------------------------------------------------------------
MSSV = 2251010027  # Mã số sinh viên cá nhân dùng làm seed tạo dữ liệu độc bản
np.random.seed(MSSV)  # Cố định hạt giống ngẫu nhiên: giúp số liệu luôn nhất quán mỗi lần chạy lại
n = 120  # Số lượng mẫu (bản ghi) sinh viên cần tạo ban đầu

print("=" * 70)
print(f"BẮT ĐẦU QUY TRÌNH LÀM SẠCH DỮ LIỆU CHO MSSV: {MSSV}")
print("=" * 70)


# ==============================================================================
# BƯỚC 1: KHỞI TẠO VÀ TẠO TỆP DỮ LIỆU THÔ CHỨA LỖI (DATA GENERATION)
# ==============================================================================
print("\n--- BƯỚC 1: KHỞI TẠO VÀ SINH TỆP DỮ LIỆU LỖI ---")

# --- 1. Tạo danh sách MSSV ---
# List comprehension: tạo 120 mã sinh viên dạng 'SV2251010027', 'SV2251010026',...
mssv_list = [f'SV{MSSV - i}' for i in range(n)]

# --- 2. Tạo họ tên (Cố tình tạo lỗi khoảng trắng thừa ở 2 đầu) ---
ho = ['Nguyen', 'Tran', 'Le', 'Pham', 'Hoang', 'Huynh', 'Phan', 'Vu', 'Vo', 'Dang']
ten_dem = ['Van', 'Thi', 'Minh', 'Hoang', 'Duc', 'Thu', 'Anh', 'Quoc', '']
ten = ['An', 'Binh', 'Cuong', 'Dung', 'Em', 'Phong', 'Giang', 'Hai', 'Khanh', 'Lan',
       'Mai', 'Nam', 'Oanh', 'Phuc', 'Quyen', 'Son', 'Tam', 'Uyen', 'Vinh', 'Yen']

ho_ten_list = []
for i in range(n):
    h = np.random.choice(ho)       # np.random.choice(): Chọn ngẫu nhiên 1 phần tử trong danh sách họ
    td = np.random.choice(ten_dem) # Chọn ngẫu nhiên 1 tên đệm
    t = np.random.choice(ten)      # Chọn ngẫu nhiên 1 tên chính
    name = f'{h} {td} {t}'.replace('  ', ' ').strip() # .replace('  ', ' '): Gộp khoảng trắng kép thành đơn
    
    # np.random.rand() < 0.40: Xác suất 40% cố tình chèn khoảng trắng thừa ở 2 đầu chuỗi để mô phỏng lỗi nhập liệu
    if np.random.rand() < 0.40:
        name = '  ' + name + '  '
    ho_ten_list.append(name)

# --- 3. Tạo tuổi (Cố tình tạo lỗi: giá trị khuyết NaN, tuổi âm, tuổi quá lớn) ---
# np.random.randint(18, 31): Sinh số nguyên trong khoảng [18, 30]. .astype(float): ép sang số thực để chứa được np.nan
tuoi_list = np.random.randint(18, 31, size=n).astype(float)

# np.random.choice(..., replace=False): Lấy ngẫu nhiên khoảng 8% vị trí không trùng lặp để gán thành NaN (khuyết)
nan_idx = np.random.choice(n, size=max(1, int(n * 0.08)), replace=False)
tuoi_list[nan_idx] = np.nan

# Cố tình gán 2 giá trị ngoại lai cực đoan: -12 (tuổi âm vô lý) và 195 (tuổi người vượt quá tuổi thọ)
outlier_idx = np.random.choice([i for i in range(n) if i not in nan_idx], size=2, replace=False)
tuoi_list[outlier_idx[0]] = -12
tuoi_list[outlier_idx[1]] = 195

# --- 4. Tạo điểm số (Cố tình tạo lỗi: khuyết NaN, điểm âm, điểm vượt thang 10) ---
# np.random.uniform(0, 10): Sinh số thực phân phối đều từ 0 đến 10. np.round(..., 1): làm tròn 1 chữ số thập phân
diem_list = np.round(np.random.uniform(0, 10, size=n), 1)

# Lấy ngẫu nhiên khoảng 7% số dòng để gán thành NaN
nan_idx2 = np.random.choice(n, size=max(1, int(n * 0.07)), replace=False)
diem_list[nan_idx2] = np.nan

# Cố tình gán 2 giá trị vi phạm thang điểm: -2.0 (điểm âm) và 95.0 (vượt thang điểm 10)
err_idx = np.random.choice([i for i in range(n) if i not in nan_idx2], size=2, replace=False)
diem_list[err_idx[0]] = -2.0
diem_list[err_idx[1]] = 95.0

# --- 5. Tạo tỉnh/thành (Cố tình tạo lỗi không nhất quán dữ liệu dạng chữ) ---
# Các biến thể không đồng nhất: 'Ha Noi' vs 'HN', 'TP.HCM' vs 'HCM'
tinh_variants = ['Ha Noi', 'HN', 'TP.HCM', 'HCM', 'Da Nang', 'Can Tho', 'Hai Phong']
tinh_list = [np.random.choice(tinh_variants) for _ in range(n)]

# --- 6. Tạo DataFrame tổng hợp ---
# pd.DataFrame(): Đóng gói các cột danh sách dữ liệu thành bảng 2 chiều hoàn chỉnh
df = pd.DataFrame({
    'MSSV': mssv_list,
    'Ho_Ten': ho_ten_list,
    'Tuoi': tuoi_list,
    'Diem_So': diem_list,
    'Tinh_Thanh': tinh_list
})

# --- 7. Nhân bản 6 bản ghi ở cuối (Cố tình tạo lỗi dữ liệu trùng lặp) ---
# df.sample(n=6): Lấy ngẫu nhiên 6 dòng từ bảng theo seed MSSV
duplicates = df.sample(n=6, random_state=MSSV)
# pd.concat(): Ghép nối 6 dòng nhân bản vào cuối bảng. ignore_index=True: đánh lại chỉ mục index từ 0 -> 125
df = pd.concat([df, duplicates], ignore_index=True)

# --- 8. Xuất file dữ liệu thô chứa lỗi ra đĩa ---
raw_filename = f'data_loi_{MSSV}.csv'
# df.to_csv(): Ghi bảng ra file CSV. index=False: không lưu cột số thứ tự mặc định của pandas
df.to_csv(raw_filename, index=False, encoding='utf-8')

print(f'Đã tạo file: {raw_filename}')
print(f'Số dòng: {len(df)} (gồm 120 dòng gốc + 6 dòng nhân bản trùng lặp)')
# df['Tuoi'].isna().sum(): .isna() trả về True/False xem có khuyết không, .sum() tính tổng số ô True
print(f'Số giá trị khuyết Tuoi ban đầu: {df["Tuoi"].isna().sum()}')
print(f'Số giá trị khuyết Diem_So ban đầu: {df["Diem_So"].isna().sum()}')
print("\n10 dòng đầu tiên của dữ liệu thô (df.head(10)):")
print(df.head(10))


# ==============================================================================
# BƯỚC 2: ĐỌC DỮ LIỆU VÀ ĐÁNH GIÁ CẤU TRÚC TỔNG THỂ (DATA PROFILING)
# ==============================================================================
print("\n" + "=" * 70)
print("--- BƯỚC 2: DATA PROFILING (ĐÁNH GIÁ CẤU TRÚC DỮ LIỆU) ---")
# pd.read_csv(): Đọc dữ liệu từ file CSV vừa tạo vào bộ nhớ
df = pd.read_csv(raw_filename, encoding='utf-8')

print("\n=== Thông tin cấu trúc dữ liệu (df.info()) ===")
# df.info(): Hiển thị số dòng, tên cột, số giá trị không rỗng (Non-Null Count) và kiểu dữ liệu (Dtype)
df.info()

print("\n=== Thống kê mô tả (df.describe()) ===")
# df.describe(): Thống kê định lượng gồm: số lượng (count), trung bình (mean), độ lệch chuẩn (std),
# giá trị nhỏ nhất (min), các mốc phân vị (25%, 50%, 75%) và lớn nhất (max).
# Giúp phát hiện nhanh các giá trị ngoại lai bất thường (min Tuoi = -12, max = 195; min Diem = -2, max = 95).
print(df.describe())

print("\n=== 5 dòng đầu tiên (df.head()) ===")
# df.head(): Trích xuất nhanh 5 dòng đầu tiên để xem cấu trúc thực tế của dữ liệu
print(df.head())


# ==============================================================================
# BƯỚC 3: XỬ LÝ BẢN GHI TRÙNG LẶP (DUPLICATE RECORDS)
# ==============================================================================
print("\n" + "=" * 70)
print("--- BƯỚC 3: XỬ LÝ BẢN GHI TRÙNG LẶP ---")
# df.duplicated(): Trả về Series True/False, True nếu dòng đó trùng hoàn toàn với 1 dòng đã xuất hiện trước đó
# .sum(): Tính tổng số dòng bị trùng lặp
num_duplicates = df.duplicated().sum()
print(f"Số dòng trùng lặp phát hiện được: {num_duplicates}")

# df.drop_duplicates(inplace=True): Xóa bỏ các dòng trùng, chỉ giữ lại bản ghi gốc đầu tiên
# inplace=True: Cập nhật trực tiếp lên biến df mà không cần gán lại (df = df...)
df.drop_duplicates(inplace=True)
print(f"Số dòng còn lại sau khi loại bỏ trùng: {len(df)}")


# ==============================================================================
# BƯỚC 4: XỬ LÝ CÁC GIÁ TRỊ KHUYẾT THIẾU (MISSING VALUES)
# ==============================================================================
print("\n" + "=" * 70)
print("--- BƯỚC 4: XỬ LÝ CÁC GIÁ TRỊ KHUYẾT THIẾU ---")

# Đếm số lượng ô khuyết TRƯỚC khi điền để lưu kết quả trả lời Câu hỏi thu hoạch 1
missing_tuoi = int(df['Tuoi'].isna().sum())
missing_diem = int(df['Diem_So'].isna().sum())
print(f"Số giá trị khuyết cột Tuoi: {missing_tuoi}")
print(f"Số giá trị khuyết cột Diem_So: {missing_diem}")

# --- Tính toán giá trị đại diện để điền khuyết ---
# .mean(): Tính giá trị Trung bình cộng của cột điểm số
diem_mean = df['Diem_So'].mean()

# .median(): Tính giá trị Trung vị (vị trí 50% sau khi sắp xếp) của cột tuổi.
# LÝ DO DÙNG MEDIAN: Cột tuổi có ngoại lai cực đoan (-12 và 195). Trung vị có tính bền vững (Robust),
# không bị kéo lệch như Trung bình, giúp phản ánh trung thực độ tuổi sinh viên.
tuoi_median = df['Tuoi'].median()

print(f"Giá trị Trung bình (Mean) của Diem_So dùng để điền: {diem_mean:.4f}")
print(f"Giá trị Trung vị (Median) của Tuoi dùng để điền: {tuoi_median:.1f}")

# .fillna(): Điền các ô có giá trị NaN bằng giá trị chỉ định
df['Diem_So'] = df['Diem_So'].fillna(diem_mean)
df['Tuoi'] = df['Tuoi'].fillna(tuoi_median)

print("Đã điền xong giá trị khuyết.")
# Kiểm tra lại: đảm bảo số ô khuyết sau khi xử lý phải bằng 0
print(f"Kiểm tra - Tuoi còn khuyết: {df['Tuoi'].isna().sum()}")
print(f"Kiểm tra - Diem_So còn khuyết: {df['Diem_So'].isna().sum()}")


# ==============================================================================
# BƯỚC 5: CHUẨN HÓA DỮ LIỆU DẠNG VĂN BẢN (TEXT STANDARDIZATION)
# ==============================================================================
print("\n" + "=" * 70)
print("--- BƯỚC 5: CHUẨN HÓA DỮ LIỆU DẠNG VĂN BẢN ---")

# 1. Xóa khoảng trắng thừa ở hai đầu chuỗi họ tên
# .str: Cho phép gọi các hàm xử lý chuỗi ký tự trên toàn bộ cột (vectorized string methods)
# .strip(): Cắt bỏ các ký tự dấu cách trống ở đầu và cuối chuỗi
df['Ho_Ten'] = df['Ho_Ten'].str.strip()

# 2. Chuẩn hóa tên tỉnh/thành phố không nhất quán
# .replace(dict): Ánh xạ thay thế các từ viết tắt về tên quy chuẩn thống nhất
df['Tinh_Thanh'] = df['Tinh_Thanh'].replace({
    'HN': 'Ha Noi',
    'HCM': 'TP.HCM'
})

# .unique(): Trả về mảng chứa các giá trị độc nhất (phân biệt), giúp kiểm tra xem còn từ viết tắt lạ nào không
print("Các giá trị độc nhất của cột Tinh_Thanh sau chuẩn hóa:")
print(df['Tinh_Thanh'].unique())


# ==============================================================================
# BƯỚC 6: XỬ LÝ DỮ LIỆU SAI LỖI LOGIC VÀ XUẤT TỆP TIN SẠCH (DATA FILTERING)
# ==============================================================================
print("\n" + "=" * 70)
print("--- BƯỚC 6: XỬ LÝ LỖI LOGIC VÀ XUẤT TỆP TIN SẠCH ---")

# Lọc bỏ các bản ghi vi phạm tính logic của dữ liệu:
# - Tuổi sinh viên hợp lệ: từ 18 đến 30 tuổi (loại bỏ -12 và 195)
# - Điểm số hợp lệ: thang điểm từ 0.0 đến 10.0 (loại bỏ -2.0 và 95.0)
# Toán tử '&': Phép VÀ (AND) theo từng phần tử giữa các điều kiện Boolean
# .copy(): Tạo bản sao độc lập của DataFrame trên bộ nhớ để tránh cảnh báo SettingWithCopyWarning
df_clean = df[
    (df['Tuoi'] >= 18) & (df['Tuoi'] <= 30) &
    (df['Diem_So'] >= 0) & (df['Diem_So'] <= 10)
].copy()

# Xuất dữ liệu sạch ra tệp CSV để sẵn sàng cho các thuật toán Data Mining / Machine Learning
clean_filename = f'{MSSV}_clean.csv'
df_clean.to_csv(clean_filename, index=False, encoding='utf-8')
print(f"Số dòng còn lại sau khi lọc logic: {len(df_clean)}")
print(f"Đã xuất tệp dữ liệu sạch: {clean_filename}")

print("\n=== Kiểm tra chất lượng dữ liệu sạch ===")
print(df_clean.info())
# .min(), .max(): Kiểm tra cận giá trị nhỏ nhất và lớn nhất sau khi làm sạch để chắc chắn không còn giá trị phi lý
print(f"Min Tuoi = {df_clean['Tuoi'].min()}, Max Tuoi = {df_clean['Tuoi'].max()}")
print(f"Min Diem_So = {df_clean['Diem_So'].min()}, Max Diem_So = {df_clean['Diem_So'].max()}")
print("Các tỉnh thành trong tập dữ liệu sạch:", df_clean['Tinh_Thanh'].unique().tolist())


# ==============================================================================
# PHẦN 4: THU HOẠCH BẮT BUỘC (KẾT QUẢ ĐỐI VỚI MSSV 2251010027)
# ==============================================================================
print("\n" + "=" * 70)
print("--- PHẦN THU HOẠCH BẮT BUỘC ---")
print(f"Câu hỏi 1:")
print(f"- Số dòng trùng lặp hệ thống quét được: {num_duplicates} dòng")
print(f"- Số ô khuyết ở cột Tuoi: {missing_tuoi} ô")
print(f"- Số ô khuyết ở cột Diem_So: {missing_diem} ô")

print(f"\nCâu hỏi 2:")
print(f"- Giải thích: Cột Tuoi chứa các giá trị ngoại lai cực đoan (âm: -12, quá lớn: 195).")
print(f"  Trung bình (Mean) rất nhạy cảm với ngoại lai và sẽ bị kéo lệch đáng kể khỏi độ tuổi thực tế.")
print(f"  Trong khi đó, Trung vị (Median = {tuoi_median:.1f}) là thước đo có tính bền vững (Robust statistics),")
print(f"  đại diện chính xác cho độ tuổi trung tâm của nhóm sinh viên mà không bị chi phối bởi các giá trị bất thường.")

print(f"\nCâu hỏi 3:")
print(f"- Tệp dữ liệu sạch cuối cùng ({clean_filename}) còn lại chính xác: {len(df_clean)} dòng.")
print("=" * 70)
