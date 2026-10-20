import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Nạp dữ liệu và chuyển các cột sang kiểu số 
df = pd.read_csv('car.csv', sep='|')
df['car_year'] = pd.to_numeric(df['car_year'], errors='coerce')
df['km'] = pd.to_numeric(df['km'].str.replace(',', '').str.replace(' Km', ''), errors='coerce')

stds = df[['car_year', 'km']].std()
max_col = stds.idxmax()
print("Cột có độ lệch chuẩn (std) lớn nhất:", max_col, f"với std = {stds[max_col]:,.2f}")
print("Ý nghĩa: Cột km có std lớn nhất thể hiện sự phân tán dữ liệu rất mạnh giữa các dòng xe (từ xe mới 0 km đến xe chạy lâu năm) và có chứa nhiều ngoại lai (outlier).")

# 1. Lấp đầy ô trống bằng giá trị Trung vị (Median)
df['car_year'] = df['car_year'].fillna(df['car_year'].median())
print("Số giá trị khuyết ở car_year sau khi điền median:", df['car_year'].isnull().sum())

# 2. Xóa bỏ các dòng dữ liệu bị khuyết tại cột quan trọng
df = df.dropna(subset=['car_model'])
print("Kích thước dữ liệu sau khi dropna:", df.shape)

# 1. Biểu đồ hộp (Boxplot) tìm Outliers
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['car_year'])
plt.title('Biểu đồ hộp (Boxplot) Năm sản xuất')
plt.xlabel('Năm sản xuất')
plt.tight_layout()
plt.show()

# 2. Biểu đồ phân tán (Scatter Plot) biểu diễn mối tương quan giữa 2 cột số
plt.figure(figsize=(8, 5))
sns.scatterplot(x='car_year', y='km', data=df)
plt.title('Biểu đồ phân tán: Tương quan giữa Năm sản xuất và Số km')
plt.xlabel('Năm sản xuất')
plt.ylabel('Số Km')
plt.tight_layout()
plt.show()
