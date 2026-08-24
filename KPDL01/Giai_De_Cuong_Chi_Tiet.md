# HƯỚNG DẪN GIẢI CHI TIẾT CÁC DẠNG BÀI TẬP (CẬP NHẬT FULL LAB 02, 03, 04, 05, TEST 01, TEST 02)

Dựa trên toàn bộ các file PDF đề bài, mình đã gom nhóm tất cả các yêu cầu thành **5 DẠNG BÀI TOÁN CHÍNH**. Dưới đây là yêu cầu chi tiết và code mẫu "ăn điểm" cho từng dạng.

---

## DẠNG 1: BÀI TOÁN LÀM SẠCH VÀ PHÂN TÍCH THỐNG KÊ (Dựa theo Test 01 - Telco Churn)

### Yêu cầu 1.1: Kiểm tra cấu trúc và Làm sạch dữ liệu
**Đề hỏi:** Kiểm tra dòng, cột, kiểu dữ liệu. Xử lý giá trị khuyết (giải thích lý do), giá trị trùng lặp. Kiểm tra miền giá trị.
```python
import pandas as pd
import numpy as np

df = pd.read_csv("Telco-Customer-Churn.csv")

# 1. Kiểm tra cấu trúc
print("Số dòng, số cột:", df.shape)
print("Tên cột:\n", df.columns.tolist())
print("Kiểu dữ liệu:\n", df.dtypes)

# 2. Xử lý kiểu dữ liệu chưa phù hợp (vd TotalCharges đang là object)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# 3. Xử lý bản ghi trùng lặp
print("Số dòng trùng:", df.duplicated().sum())
df = df.drop_duplicates()

# 4. Xử lý giá trị khuyết (Missing Values)
print("Giá trị khuyết:\n", df.isnull().sum())
# Giải thích: Vì số lượng dòng bị khuyết rất nhỏ so với tổng dữ liệu nên ta chọn phương pháp xóa bỏ (dropna) để không làm méo mó phân bố dữ liệu. (Hoặc có thể dùng median).
df = df.dropna().reset_index(drop=True)

# 5. Kiểm tra miền giá trị
for col in ['gender', 'Partner', 'InternetService']:
    print(f"Miền giá trị {col}:", df[col].unique())
```

### Yêu cầu 1.2: Thống kê mô tả và Biểu đồ phân bố
**Đề hỏi:** Đếm số lượng theo điều kiện (giới tính, hợp đồng...). Tính Mean, Median, Mode, Std. Vẽ biểu đồ phân bố (Histogram).
```python
import matplotlib.pyplot as plt

# 1. Đếm số lượng
print("Tổng khách hàng:", len(df))
print("Giới tính:\n", df['gender'].value_counts())

# 2. Tính toán đặc trưng của cột số (MonthlyCharges)
print("Trung bình:", df['MonthlyCharges'].mean())
print("Trung vị:", df['MonthlyCharges'].median())
print("Mode:", df['MonthlyCharges'].mode()[0])
print("Độ lệch chuẩn:", df['MonthlyCharges'].std())

# 3. Vẽ biểu đồ phân bố (Histogram)
plt.figure(figsize=(10, 5))
plt.hist(df['MonthlyCharges'], bins=20, color='skyblue', edgecolor='black')
plt.title("Phân bố chi phí hàng tháng (MonthlyCharges)")
plt.xlabel("Chi phí"); plt.ylabel("Số lượng khách hàng")
plt.show()
```

---

## DẠNG 2: THAO TÁC NGHIỆP VỤ PHỨC TẠP BẰNG PANDAS (Dựa theo Lab 02 - Điểm Thi THPT)

### Yêu cầu 2.1: Tính toán tổ hợp cột phức tạp
**Đề hỏi:** Kiểm tra sinh viên thi cả KHTN và KHXH. Xây dựng phổ điểm tổng cộng (Toán+Văn+Ngoại ngữ+KHXH/KHTN).
```python
# 1. Lọc sinh viên thi cả KHTN và KHXH
# Điều kiện: Cả 2 cột KHTN và KHXH đều có giá trị (notna) và > 0
ca_hai = df[(df['KHTN'].notna() & (df['KHTN'] > 0)) & 
            (df['KHXH'].notna() & (df['KHXH'] > 0))]
print(f"Số sinh viên thi cả 2 bài tổ hợp: {len(ca_hai)}")

# 2. Tạo cột điểm tổ hợp (Ưu tiên KHTN, nếu rỗng thì lấy KHXH)
df['KH_chon'] = df['KHTN'].fillna(df['KHXH'])

# 3. Tính tổng 4 môn (Chỉ tính khi sinh viên có đủ 4 cột điểm - min_count=4)
df['Tong'] = df[['Toan', 'Nguvan', 'Ngoaingu', 'KH_chon']].sum(axis=1, min_count=4)
```

### Yêu cầu 2.2: Phân loại bằng hàm tự định nghĩa `.apply(axis=1)`
**Đề hỏi:** Xếp loại tốt nghiệp (Giỏi, Khá, Trung bình, Không tốt nghiệp) dựa trên nhiều điều kiện điểm liệt (<1) và điểm trung bình.
```python
def xep_loai_tot_nghiep(row):
    # Lấy các môn bắt buộc
    bai_thi = [row['Toan'], row['Nguvan'], row['Ngoaingu'], row['KH_chon']]
    
    # 1. Nếu thiếu bất kỳ môn nào -> Khong tot nghiep
    if any(pd.isna(d) for d in bai_thi): return 'Khong tot nghiep'
    
    # 2. Xóa các môn thành phần bị NaN để kiểm tra điểm liệt
    khtn_cols = ['Vatli', 'Hoahoc', 'Sinhhoc']
    khxh_cols = ['Diali', 'LichSu', 'GDCD']
    
    if pd.notna(row['KHTN']):
        thanh_phan = [row[m] for m in khtn_cols if pd.notna(row[m])]
    else:
        thanh_phan = [row[m] for m in khxh_cols if pd.notna(row[m])]
        
    tat_ca_diem = bai_thi + thanh_phan
    
    # 3. Điều kiện điểm liệt (<= 1 điểm)
    if any(d <= 1 for d in tat_ca_diem): return 'Khong tot nghiep'
    
    # 4. Tính điểm xếp loại
    diem_xl = sum(bai_thi) / len(bai_thi)
    if diem_xl < 5: return 'Khong tot nghiep'
    if diem_xl >= 8.0 and all(d >= 7.0 for d in bai_thi): return 'Gioi'
    if diem_xl >= 6.5 and all(d >= 6.0 for d in bai_thi): return 'Kha'
    return 'Trung binh'

# Áp dụng hàm lên từng dòng (axis=1)
df['Xep_loai'] = df.apply(xep_loai_tot_nghiep, axis=1)

# Vẽ biểu đồ Pie tỉ lệ
counts = df['Xep_loai'].value_counts()
plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Tỉ lệ xếp loại tốt nghiệp')
plt.show()
```

---

## DẠNG 3: BÀI TOÁN PHÂN LOẠI & LỰA CHỌN ĐẶC TRƯNG (Dựa theo Lab 05 - Bank)

### Yêu cầu 3.1: Ma trận tương quan (Heatmap)
```python
import seaborn as sns 
df_numeric = df.select_dtypes(include=['int64', 'float64'])
corr_matrix = df_numeric.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Ma trận tương quan")
plt.show()
```

### Yêu cầu 3.2: Lựa chọn đặc trưng (SelectKBest & Random Forest)
```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

X = df.drop(columns=['y'])
y = df['y']

# Phương pháp 1: SelectKBest (Kiểm định thống kê ANOVA)
selector = SelectKBest(score_func=f_classif, k=5)
selector.fit(X_train, y_train)
kbest_cols = X_train.columns[selector.get_support()]

# Phương pháp 2: Random Forest Feature Importance
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
rf_cols = importances.head(5).index
```

---

## DẠNG 4: BÀI TOÁN TRAIN/TEST SPLIT, CROSS VALIDATION & KNN, NAIVE BAYES (Dựa theo Lab 04)

### Yêu cầu 4.1: Tách tập dữ liệu và lưu ra file
**Đề hỏi:** Tách Train (90%) - Test (10%), phải đảm bảo tỷ lệ các lớp phân bổ đều (Stratify). Lưu ra 2 file riêng.
```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=['nhan'])
y = df['nhan']

# stratify=y đảm bảo tỉ lệ Yes/No trong Train và Test đều giống nhau
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, stratify=y, random_state=42)

# Ghép X và y lại để lưu ra file CSV
TrainData = pd.concat([X_train, y_train], axis=1)
TestData = pd.concat([X_test, y_test], axis=1)

TrainData.to_csv("TrainData.csv", index=False)
TestData.to_csv("TestData.csv", index=False)
```

### Yêu cầu 4.2: Đánh giá mô hình KNN bằng Cross Validation và % Split để tìm K tốt nhất
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict

k_values = [1, 3, 5, 7, 9, 11]

# 1. Tìm K bằng % Split (Trích 30% từ tập Train ra làm Validation)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.3, stratify=y_train, random_state=42)
best_k_split, best_acc_split = 1, 0
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_tr, y_tr)
    acc = accuracy_score(y_val, knn.predict(X_val))
    if acc > best_acc_split:
        best_acc_split, best_k_split = acc, k

# 2. Tìm K bằng 10-Fold Cross Validation (Thử trên toàn bộ X_train)
best_k_cv, best_acc_cv = 1, 0
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    # cross_val_predict tự động chia X_train làm 10 phần, train 9 phần, test 1 phần
    y_pred_cv = cross_val_predict(knn, X_train, y_train, cv=10) 
    acc = accuracy_score(y_train, y_pred_cv)
    if acc > best_acc_cv:
        best_acc_cv, best_k_cv = acc, k

print(f"K tốt nhất theo % Split: {best_k_split}")
print(f"K tốt nhất theo 10-Fold CV: {best_k_cv}")

# Cuối cùng, dùng best_k_cv để test trên tập TestData (X_test)
final_knn = KNeighborsClassifier(n_neighbors=best_k_cv)
final_knn.fit(X_train, y_train)
y_pred_test = final_knn.predict(X_test)
```

---

## DẠNG 5: BÀI TOÁN PHÂN CỤM K-MEANS & PCA (Dựa theo Test 02, Lab 03)

### Yêu cầu 5.1: Tìm K tối ưu (Phương pháp Elbow) và chạy K-Means
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# BẮT BUỘC PHẢI SCALE TRƯỚC KHI CHẠY K-MEANS
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []
for k in range(2, 7):
    # Dùng n_init=10 hoặc K-Means++ để tránh cảnh báo
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(2, 7), wcss, marker='o')
plt.title('Phương pháp Elbow')
plt.show() # Chọn K tại vị trí uốn cong

# Gom cụm thực tế với K=3
kmeans_opt = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans_opt.fit_predict(X_scaled)

# Tìm Centroid của cụm (Tọa độ trung tâm)
print("Centroids:\n", kmeans_opt.cluster_centers_)
```

### Yêu cầu 5.2: Trực quan hóa dữ liệu N-chiều bằng PCA
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='Set1', alpha=0.6, edgecolors='w')
plt.title('Trực quan hóa Phân cụm K-Means bằng PCA (2D)')
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.colorbar(label='Cluster')
plt.show()
```

### Yêu cầu 5.3: Lập bảng chéo (Crosstab) so sánh cụm với nhãn thực tế
```python
# Lập bảng chéo Crosstab
crosstab = pd.crosstab(df['Cluster'], df['is_canceled'])
print(crosstab)

# Tính tỉ lệ phần trăm để đánh giá (xem code ở Dạng 1 phần Tính tỉ lệ rời bỏ)
```
