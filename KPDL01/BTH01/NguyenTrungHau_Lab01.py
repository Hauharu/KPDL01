import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

df = pd.read_csv('car.csv', sep='|')

print("Kích thước tập dữ liệu (Số dòng, Số cột):", df.shape)

df['car_year'] = pd.to_numeric(df['car_year'], errors='coerce')

print("\n--- THÔNG TIN CHI TIẾT CÁC CỘT ---")
df.info()

print("\n--- 5 DÒNG ĐẦU TIÊN CỦA DATASET ---")
print(df.head())

print("\n--- THỐNG KÊ MÔ TẢ ĐỐI VỚI CÁC BIẾN SỐ ---")
print(df.describe())

print("\nSố lượng giá trị bị khuyết trong từng thuộc tính:")
print(df.isnull().sum())

df['car_year'] = df['car_year'].fillna(df['car_year'].mean())

print("\nSố lượng khuyết thiếu sau khi xử lý:")
print(df.isnull().sum())

plt.figure(figsize=(8, 5))
sns.histplot(df['car_year'], kde=True, color='teal')
plt.title('Biểu đồ phân phối tần suất thuộc tính số')
plt.xlabel('Năm sản xuất (car_year)')
plt.ylabel('Tần suất xuất hiện')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
df['car_type'].value_counts().plot(kind='bar', color='coral')
plt.title('Biểu đồ thống kê số lượng theo từng nhóm')
plt.xlabel('Kiểu dáng xe (car_type)')
plt.ylabel('Số lượng (Dòng)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
