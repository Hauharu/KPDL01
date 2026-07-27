import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
CSV_PATH = 'Telco_Customer_Churn.csv'
plt.style.use('seaborn-v0_8-whitegrid')

print('CÂU 1: LÀM SẠCH DỮ LIỆU')
df = pd.read_csv(CSV_PATH)

print(df.info())

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df['TotalCharges'] = df['TotalCharges'].fillna(0)
df['InternetService'] = df['InternetService'].fillna(df['InternetService'].mode()[0])
df['PaymentMethod'] = df['PaymentMethod'].fillna(df['PaymentMethod'].mode()[0])
df['MonthlyCharges'] = df['MonthlyCharges'].fillna(df['MonthlyCharges'].median())

df = df.drop_duplicates()

print(df)
print(f"Kích thước sau làm sạch: {df.shape}")
print(f"Số lượng giá trị khuyết còn lại: {df.isnull().sum().sum()}")

print('CÂU 2: THỐNG KÊ')
print(f"Số lượng khách hàng: {len(df)}")
print("\nGiới tính:\n", df['gender'].value_counts())
print("\nLoại hợp đồng:\n", df['Contract'].value_counts())
print("\nDịch vụ Internet:\n", df['InternetService'].value_counts())

stats = df['MonthlyCharges'].agg(['mean', 'median', 'std'])
mode_val = df['MonthlyCharges'].mode()[0]
print(f"\nMonthlyCharges: Mean={stats['mean']:.2f}, Median={stats['median']:.2f}, Mode={mode_val:.2f}, Std={stats['std']:.2f}")

plt.figure(figsize=(10, 5))
sns.histplot(df['MonthlyCharges'], bins=30, kde=True)
plt.title('Phân bố Monthly Charges')
plt.show()

print('CÂU 3: PHÂN LOẠI TENURE')
def classify(row):
    if row['Churn'] == 'Yes': return 'Đã rời bỏ'
    if row['tenure'] >= 48: return 'Trung thành'
    if 24 <= row['tenure'] < 48: return 'Ổn định'
    return 'Mới'

df['Segment'] = df.apply(classify, axis=1)
counts = df['Segment'].value_counts()
print(counts)

plt.figure(figsize=(10, 5))
sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette='viridis', legend=False)
plt.title('Phân đoạn khách hàng')
plt.show()

print('CÂU 4: CHUẨN HÓA')
df_ml = df.drop(columns=['customerID', 'Segment']).copy()

for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']:
    if col in df_ml.columns: df_ml[col] = df_ml[col].map({'Yes': 1, 'No': 0})
df_ml['gender'] = df_ml['gender'].map({'Female': 1, 'Male': 0})

df_ml = pd.get_dummies(df_ml)
scaler = MinMaxScaler()
df_ml[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(df_ml[['tenure', 'MonthlyCharges', 'TotalCharges']])

OUTPUT = 'Telco_Customer_Churn_Preprocessed.csv'
df_ml.to_csv(OUTPUT, index=False)
print(f"Đã lưu file: {OUTPUT}")