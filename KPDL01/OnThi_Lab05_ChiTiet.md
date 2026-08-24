
Bài thi hôm nay của bạn sẽ tương tự Lab 05, tức là tập trung vào **tiền xử lý phức tạp, vẽ Heatmap, dùng các thuật toán lựa chọn đặc trưng (SelectKBest, Random Forest) và so sánh nhiều mô hình máy học (SVM, Decision Tree, Random Forest)**. 

Dưới đây là một bộ code mẫu hoàn chỉnh và liền mạch từ trên xuống dưới, bạn có thể copy-paste và thay đổi tên cột cho phù hợp với đề thi của mình.

---

## BƯỚC 1: TIỀN XỬ LÝ DỮ LIỆU & EDA CƠ BẢN (Yêu cầu 1 & 2)

> [!TIP] 
> **MẸO DÙNG CHUNG BẤT CHẤP MỌI ĐỀ BÀI:**
> Bạn chỉ cần thay đổi 2 biến `TEN_FILE` và `COT_NHAN` dưới đây cho khớp với đề bài. Toàn bộ code từ Bước 2 trở đi sẽ tự động chạy đúng mà không cần sửa thêm chữ nào!

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# ==========================================
# CHỈ CẦN THAY ĐỔI 2 GIÁ TRỊ NÀY KHI THI:
TEN_FILE = 'bank-full.csv'
COT_NHAN = 'y'
# ==========================================

# 1. Đọc dữ liệu
df = pd.read_csv(TEN_FILE)

# 2. Xử lý NaN và Trùng lặp
df = df.dropna().drop_duplicates()

# 3. Mã hóa dữ liệu TỰ ĐỘNG (Không cần viết tay map)
# Đoạn code này sẽ tự tìm tất cả các cột chứa chữ và ép về số
categorical_cols = df.select_dtypes(include=['object', 'string']).columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Xóa các cột định danh hoặc ít ý nghĩa (Nếu đề yêu cầu, vd: ID, Ngày tháng)
# df = df.drop(columns=['id_khach_hang'])


# 4. Thống kê số lượng khách hàng
print('Tổng số lượng khách hàng: ', len(df))
print(f'Số lượng khách hàng từng lớp {COT_NHAN}:\n', df[COT_NHAN].value_counts())

# Vẽ biểu đồ Bar Chart cho lớp kết quả
class_counts = df[COT_NHAN].value_counts()
plt.figure(figsize=(7, 5))
plt.bar(class_counts.index.astype(str), class_counts.values, color=['skyblue', 'salmon'])
plt.title(f'Số lượng khách hàng theo lớp {COT_NHAN}')
plt.xlabel('Lớp phân loại')
plt.ylabel('Số lượng')
plt.show()
```

---

## BƯỚC 2: MA TRẬN TƯƠNG QUAN & HEATMAP (Yêu cầu 3)

Bước này giúp ta nhìn bằng mắt thường xem những cột nào có ảnh hưởng mạnh nhất tới cột kết quả, từ đó đề xuất ra một bộ thuộc tính.

```python
# Tách X và y tự động dựa vào biến COT_NHAN đã khai báo ở trên
X = df.drop(columns=[COT_NHAN])
y = df[COT_NHAN]

# Tính ma trận tương quan (chỉ tính trên các cột số)
corr_matrix = df.corr(numeric_only=True)

# Vẽ đồ thị Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Ma trận tương quan Pearson')
plt.tight_layout()
plt.show()

# NHẬN XÉT (Câu hỏi lý thuyết hay gặp):
# - Nhìn vào dòng/cột của biến 'y', những biến nào có hệ số tương quan (độ lớn tuyệt đối) 
#   càng gần 1 thì càng có mức độ ảnh hưởng lớn.
# - Dựa vào Heatmap, ta thấy các biến có tương quan cao với y là: 'duration', 'previous', 'housing'...
```

---

## BƯỚC 3: CHIA TẬP VÀ LỰA CHỌN ĐẶC TRƯNG TỰ ĐỘNG (Yêu cầu 4)

Việc chạy model với quá nhiều cột sẽ làm chậm và gây nhiễu. Ta dùng thuật toán để máy tự động chọn ra `K` cột tốt nhất.

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

# 1. Chia Train/Test (Luôn nhớ stratify=y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Lựa chọn bằng SelectKBest (Kiểm định ANOVA)
# Lấy 5 thuộc tính tốt nhất
selector = SelectKBest(score_func=f_classif, k=5)
selector.fit(X_train, y_train)
kbest_features = X_train.columns[selector.get_support()].tolist()

print("\n--- 5 thuộc tính tốt nhất theo SelectKBest ---")
print(kbest_features)

# 3. Lựa chọn bằng Feature Importances của Random Forest
rf_fs = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_fs.fit(X_train, y_train)

# Lấy điểm số mức độ quan trọng và sắp xếp giảm dần
importances = pd.Series(rf_fs.feature_importances_, index=X_train.columns).sort_values(ascending=False)
rf_top_features = importances.head(5).index.tolist()

print("\n--- 5 thuộc tính quan trọng nhất theo Random Forest ---")
print(importances.head(5))

# 4. Lưu lại tập dữ liệu đã được thu gọn (Giả sử ta tin tưởng thuật toán Random Forest hơn)
X_train_selected = X_train[rf_top_features]
X_test_selected = X_test[rf_top_features]
```
*(Nếu thầy cô yêu cầu so sánh:* `SelectKBest` đánh giá sự phụ thuộc tuyến tính của từng biến độc lập với `y`. Trong khi đó, `Random Forest` phát hiện được cả quan hệ phi tuyến và sự kết hợp của nhiều biến với nhau. Thường RF sẽ cho kết quả thực tế tốt hơn).*

---

## BƯỚC 4: ĐÁNH GIÁ MÔ HÌNH TRƯỚC VÀ SAU CHỌN LỌC (Yêu cầu 5, 6, 7)

Đây là bước cực kỳ dài dòng nếu viết chay. Bạn hãy **viết 1 hàm duy nhất**, sau đó truyền model và tập dữ liệu vào để code chạy tự động.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Viết hàm đánh giá chung để xài lại nhiều lần
def evaluate_and_print(model, X_tr, y_tr, X_te, y_te, title):
    # Luôn Scale dữ liệu trước khi train (Cần cho SVM)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    # Huấn luyện
    model.fit(X_tr_s, y_tr)
    
    # Dự đoán
    y_pred = model.predict(X_te_s)
    
    # Đánh giá
    print(f"\n========== {title} ==========")
    print("Confusion Matrix:\n", confusion_matrix(y_te, y_pred))
    print(f"Accuracy : {accuracy_score(y_te, y_pred):.4f}")
    print(f"Precision: {precision_score(y_te, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_te, y_pred, zero_division=0):.4f}")
    print(f"F1-score : {f1_score(y_te, y_pred, zero_division=0):.4f}")

# ----- CHẠY TEST CÁC MÔ HÌNH -----

# 1. Thuật toán SVM
# Lưu ý: Nếu dữ liệu quá lớn, SVM có thể chạy mất vài phút, hãy kiên nhẫn
evaluate_and_print(SVC(kernel='rbf', random_state=42), X_train, y_train, X_test, y_test, "SVM - FULL ĐẶC TRƯNG")
evaluate_and_print(SVC(kernel='rbf', random_state=42), X_train_selected, y_train, X_test_selected, y_test, "SVM - SAU CHỌN LỌC (TOP 5)")

# 2. Thuật toán Decision Tree
evaluate_and_print(DecisionTreeClassifier(random_state=42), X_train, y_train, X_test, y_test, "DECISION TREE - FULL ĐẶC TRƯNG")
evaluate_and_print(DecisionTreeClassifier(random_state=42), X_train_selected, y_train, X_test_selected, y_test, "DECISION TREE - SAU CHỌN LỌC (TOP 5)")

# 3. Thuật toán Random Forest
evaluate_and_print(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), X_train, y_train, X_test, y_test, "RANDOM FOREST - FULL ĐẶC TRƯNG")
evaluate_and_print(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), X_train_selected, y_train, X_test_selected, y_test, "RANDOM FOREST - SAU CHỌN LỌC (TOP 5)")

# NHẬN XÉT CHUNG VỀ KẾT QUẢ:
# - Các mô hình khi chạy trên bộ dữ liệu đã được chọn lọc đặc trưng (chỉ 5 cột) chạy NHANH HƠN rất nhiều (đặc biệt là SVM).
# - Độ chính xác (Accuracy) và F1-Score khi chỉ dùng 5 cột thường gần bằng hoặc thậm chí cao hơn lúc dùng Full cột do đã loại bỏ được các cột nhiễu (Noise).
```
