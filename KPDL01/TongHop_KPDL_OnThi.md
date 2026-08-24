# TỔNG HỢP KIẾN THỨC VÀ CODE MẪU ÔN THI KHAI PHÁ DỮ LIỆU (BẢN CHI TIẾT NHẤT)

Tài liệu này tổng hợp **toàn bộ** các kỹ thuật từ cơ bản đến nâng cao trong tất cả các bài thực hành của bạn. Phân loại rõ ràng cái nào là kỹ thuật **dùng chung** (bài nào cũng có), cái nào là kỹ thuật **dùng riêng** (chỉ xuất hiện ở một số dạng bài cụ thể).

---

## PHẦN 1: KỸ THUẬT DÙNG CHUNG - TIỀN XỬ LÝ DỮ LIỆU

Đây là khung xương của mọi bài toán. Bước này luôn thực hiện đầu tiên để làm sạch dữ liệu trước khi đưa vào phân tích hay học máy.

### 1.1. Đọc và Gộp nhiều file dữ liệu
*Dùng khi:* Dữ liệu bị chia nhỏ thành nhiều file (Ví dụ: `DSSV1.xlsx`, `DSSV2.xlsx`).
```python
import pandas as pd
import numpy as np

# Đọc file CSV (header=None nếu file không có tên cột)
df = pd.read_csv("data.csv", low_memory=False)

# Gộp nhiều Dataframe lại với nhau
df1 = pd.read_excel("DSSV1.xlsx")
df2 = pd.read_excel("DSSV2.xlsx")
df_tong = pd.concat([df1, df2], ignore_index=True)
```

### 1.2. Làm sạch Chuỗi, Tên Cột và Kiểu Dữ Liệu
*Dùng khi:* Tên cột lộn xộn, chuỗi dư khoảng trắng, cột số nhưng bị hiểu nhầm là chữ.
```python
# Đổi tên cột
df.columns = ['STT', 'MaSV', 'Toan', 'NguVan']

# Loại bỏ khoảng trắng 2 đầu ở các cột chuỗi
df['Location'] = df['Location'].fillna('').str.strip()

# Sửa lỗi chính tả hàng loạt bằng Dictionary
replacements = {'Ngũyn': 'Nguyễn', 'Hong': 'Hoàng'}
for wrong, right in replacements.items():
    df['Ten'] = df['Ten'].astype(str).str.replace(wrong, right)

# Ép kiểu dữ liệu về dạng số (báo lỗi sẽ chuyển thành NaN)
df['Toan'] = pd.to_numeric(df['Toan'], errors='coerce')
```

### 1.3. Xử lý giá trị Trống (NaN) và Trùng lặp (Duplicates)
```python
# Xóa dòng trùng lặp dựa trên 1 cột định danh
df = df.drop_duplicates(subset='MaSV').copy()

# Cách 1: Xóa toàn bộ dòng chứa NaN
df = df.dropna().reset_index(drop=True)

# Cách 2: Lấp đầy (Imputation) bằng Trung vị hoặc Mode
df['Toan'] = df['Toan'].fillna(df['Toan'].median())
df['GioiTinh'] = df['GioiTinh'].fillna(df['GioiTinh'].mode()[0])

# Cách 3: Lấp đầy bằng giá trị của cột khác
# (Nếu cột KHTN rỗng, lấy giá trị cột KHXH điền vào)
df['KH_chon'] = df['KHTN'].fillna(df['KHXH'])
```

---

## PHẦN 2: THỐNG KÊ VÀ XỬ LÝ LOGIC NGHIỆP VỤ (DÙNG RIÊNG TÙY BÀI)

Phần này linh hoạt tùy yêu cầu đề bài: xét điểm tốt nghiệp, tính tuổi, tạo bảng Pivot...

### 2.1. Xử lý logic phức tạp (Sử dụng hàm `.apply(axis=1)`)
*Dùng khi:* Tính toán dựa trên nhiều điều kiện rẽ nhánh phức tạp (như bài Xét Tốt Nghiệp, Phân khúc khách hàng).
```python
def xep_loai(row):
    # Lấy điểm
    bai_thi = [row['Toan'], row['Van'], row['NgoaiNgu']]
    
    # Kiem tra co mon nao <= 1 khong
    if any(diem <= 1 for diem in bai_thi):
        return 'Khong tot nghiep'
        
    diem_tb = sum(bai_thi) / 3
    if diem_tb >= 8 and all(d >= 7 for d in bai_thi): return 'Gioi'
    return 'Trung binh'

df['XepLoai'] = df.apply(xep_loai, axis=1)
```

### 2.2. Tính tổng có điều kiện
*Dùng khi:* Cần cộng dồn các cột nhưng yêu cầu phải đủ dữ liệu.
```python
# Tính tổng điểm, chỉ tính nếu cả 4 môn đều không bị rỗng (min_count=4)
df['Tong'] = df[['Toan','Van','NgoaiNgu','KH_chon']].sum(axis=1, min_count=4)
```

### 2.3. Bảng chéo (Pivot Table & Crosstab)
*Dùng khi:* So sánh 2 hoặc nhiều thuộc tính với nhau (vd: Sinh viên sinh Tháng x Năm y, Đối chiếu Cụm K-Means với Hủy phòng).
```python
# Lập bảng chéo đếm số lượng (Crosstab)
crosstab = pd.crosstab(df['Cluster'], df['is_canceled'])

# Lập Pivot Table (vd: Đếm sinh viên theo Năm và Tháng sinh)
pivot = df.pivot_table(index='Nam', columns='Thang', aggfunc='size', fill_value=0)
# Bổ sung tổng theo hàng/cột
pivot['Tổng hàng'] = pivot.sum(axis=1)
```

---

## PHẦN 3: MÃ HÓA VÀ CHUẨN HÓA DỮ LIỆU (DÙNG CHUNG CHO MACHINE LEARNING)

Các thuật toán Máy học (Đặc biệt là KNN, SVM, K-Means) rất nhạy cảm với kiểu dữ liệu chữ và sự chênh lệch thang đo. **Bắt buộc phải làm phần này trước khi đưa vào Model.**

```python
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler

# 1. Map (Nhị phân hoặc danh mục ít)
df['GioiTinh'] = df['GioiTinh'].map({'Male': 0, 'Female': 1})

# 2. Label Encoding (Cho cột chuỗi cơ bản)
le = LabelEncoder()
df['ThanhPho'] = le.fit_transform(df['ThanhPho'])

# 3. Ordinal Encoding (Danh mục có thứ bậc: Cấp 3 < CĐ < ĐH)
ordinal = OrdinalEncoder(categories=[['HighSchool', 'College', 'University']], handle_unknown='use_encoded_value', unknown_value=-1)
df['HocVan'] = ordinal.fit_transform(df[['HocVan']]) # Nhớ 2 ngoặc vuông

# 4. One-Hot Encoding (Danh mục không thứ bậc)
df = pd.get_dummies(df, columns=['City'], dtype=int)

# 5. Scaling (Chuẩn hóa)
# - MinMaxScaler (Phổ biến, đưa về 0-1)
minmax = MinMaxScaler()
df[['Tuoi', 'Luong']] = minmax.fit_transform(df[['Tuoi', 'Luong']])
# - StandardScaler (Dùng nhiều trong phân cụm, PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## PHẦN 4: LỰA CHỌN ĐẶC TRƯNG - FEATURE SELECTION (DÙNG RIÊNG CHO PHÂN LOẠI)

*Dùng khi:* Dữ liệu có quá nhiều cột, làm mô hình chậm hoặc bị nhiễu.

```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

X = df.drop(columns=['y'])
y = df['y']

# Phương pháp 1: SelectKBest (Dựa trên kiểm định ANOVA)
selector = SelectKBest(score_func=f_classif, k=5)
selector.fit(X_train, y_train)
list_kbest = X_train.columns[selector.get_support()]

# Phương pháp 2: Mức độ quan trọng từ Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
list_rf = importances.head(5).index # Chọn 5 cột đầu

# Lấy dữ liệu đã lọc
X_train_filtered = X_train[list_kbest]
X_test_filtered = X_test[list_kbest]
```

---

## PHẦN 5: CÁC MÔ HÌNH HỌC MÁY & CÁCH ĐÁNH GIÁ (MACHINE LEARNING)

### 5.1. Mô hình Phân Cụm (K-Means)
*Dùng khi:* Không có nhãn (vd: Phân khúc khách hàng).
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Tìm K tối ưu bằng WCSS (Elbow) & Silhouette Score
wcss, sil_scores = [], []
for k in range(2, 6):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled) # Nhớ dùng X_scaled
    
    wcss.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

# (Vẽ 2 biểu đồ ra để nhìn K) ...
# Khởi tạo mô hình K-Means với K=3
kmeans_opt = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans_opt.fit_predict(X_scaled)
```

### 5.2. Mô hình Phân Loại & Đánh Giá Đa Năng
*Dùng khi:* Đã có nhãn y. Bao gồm các kỹ thuật: Chia tập có phân tầng (`stratify`), Cross Validation.

```python
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Chia Train/Test (Luôn thêm stratify=y để cân bằng số lượng các lớp)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Xây dựng HÀM ĐÁNH GIÁ DÙNG CHUNG
# (Giúp tái sử dụng khi muốn test nhiều tập thuộc tính, hoặc nhiều thuật toán)
def evaluate_model(model, Xtr, ytr, Xte, yte, ten_mo_hinh="Model"):
    # Chuẩn hóa bên trong hàm để linh hoạt
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    
    # Train và Đoán
    model.fit(Xtr_s, ytr)
    y_pred = model.predict(Xte_s)
    
    # In kết quả
    print(f"--- Kết quả: {ten_mo_hinh} ---")
    print(confusion_matrix(yte, y_pred))
    print(classification_report(yte, y_pred)) # Tự in ra Precision, Recall, F1 đầy đủ
    return accuracy_score(yte, y_pred)

# 3. Sử dụng hàm:
# So sánh các tập feature khác nhau trên SVM
evaluate_model(SVC(kernel='rbf'), X_train, y_train, X_test, y_test, "SVM Full Features")
evaluate_model(SVC(kernel='rbf'), X_train[list_kbest], y_train, X_test[list_kbest], "SVM KBest Features")

# 4. Kỹ thuật Cross Validation (Xác thực chéo K-Fold)
# Dùng để đảm bảo tính khách quan của mô hình trên tập Train (Bài BTH4_Bai1)
y_pred_cv = cross_val_predict(SVC(), X_train, y_train, cv=10)
print("10-Fold CV Accuracy:", accuracy_score(y_train, y_pred_cv))
```

---

## PHẦN 6: TRỰC QUAN HÓA (VISUALIZATION NÂNG CAO)

*Mẹo chung:* Đừng quên `plt.tight_layout()` và `plt.savefig()` ở cuối.

### 6.1. Vẽ Ma Trận Tương Quan (Heatmap)
```python
corr_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.show()
```

### 6.2. PCA (Giảm chiều dữ liệu xuống 2D để vẽ Cụm)
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='Set1')
plt.xlabel('PC 1'); plt.ylabel('PC 2')
plt.title('PCA Phân Cụm')
plt.show()
```

### 6.3. Biểu đồ 2 trục Y kép (Song song WCSS và Silhouette)
```python
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(K_range, wcss, 'b-', label='WCSS')
ax1.set_ylabel('WCSS', color='b')

ax2 = ax1.twinx() # Tạo trục Y thứ 2 dùng chung trục X
ax2.plot(K_range, sil_scores, 'r--', label='Silhouette')
ax2.set_ylabel('Silhouette', color='r')
plt.show()
```

### 6.4. Biểu đồ thanh 3D (3D Bar)
*(Thường dùng khi có 2 biến độc lập, vd: Năm và Tháng ảnh hưởng tới Số lượng sinh viên).*
```python
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig3d = plt.figure(figsize=(12, 7))
ax3d = fig3d.add_subplot(111, projection='3d')

# X, Y là tọa độ mặt đáy. Z thường bắt đầu từ 0. 
# dx, dy là độ rộng cột. dz là độ cao cột.
ax3d.bar3d(xpos, ypos, np.zeros_like(xpos), dx, dy, dz_heights)
plt.show()
```
