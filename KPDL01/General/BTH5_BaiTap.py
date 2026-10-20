import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Đọc dữ liệu
df = pd.read_csv('bank-full.csv')

# --- Yêu cầu khúc đầu (Tiền xử lý giống trong ảnh) ---
print("--- Tiền xử lý dữ liệu ---")
# Xử lý giá trị khuyết
print("Số giá trị khuyết mỗi cột:\n", df.isnull().sum())
df = df.dropna()

# Xử lý dòng trùng
print("\nSố dòng bị trùng lặp:", df.duplicated().sum())
df = df.drop_duplicates()

# Loại bỏ cột định danh (job)
if 'job' in df.columns:
    df = df.drop(columns=['job'])

if 'marital' in df.columns:
    print("\nValue counts của 'marital' trước khi map:\n", df['marital'].value_counts())
    marital_map = {
        'single': 0,
        'married': 1,
        'divorced': 2
    }
    df['marital'] = df['marital'].map(marital_map)

if 'education' in df.columns:
    print("\nValue counts của 'education' trước khi map:\n", df['education'].value_counts())
    edu_map = {
        'unknown': 0,
        'primary': 1,
        'secondary': 2,
        'tertiary': 3
    }
    df['education'] = df['education'].map(edu_map)

if 'month' in df.columns:
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    df['month'] = df['month'].map(month_map)

# Ánh xạ thuộc tính nhị phân
binary_map = {
    'yes': 1,
    'no': 0
}
binary_cols = ['housing', 'loan', 'y']
for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].map(binary_map)

print("\nSố khách hàng: ", len(df))
print("Số lượng khách hàng của từng lớp:\n", df['y'].value_counts())

# Chuyển đổi các cột phân loại (object/string) còn lại sang số để đưa vào mô hình
categorical_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# --- Yêu cầu 3 ---
print("\n" + "="*40)
print("--- Yêu cầu 3: Phân tích & Lựa chọn thuộc tính thủ công ---")
# Phân tích: Loại bỏ những thuộc tính có thông tin trùng lặp hoặc ít ý nghĩa.
# Ví dụ: 'day', 'month', 'contact' ít có ý nghĩa về mặt tài chính hơn so với balance, housing, loan.
# 'pdays', 'previous' chứa nhiều giá trị 0 hoặc -1.
# Ở đây ta đề xuất loại bỏ 'day', 'month', 'contact'.
cols_to_drop_y3 = ['day', 'month', 'contact']
features_y3 = [col for col in df.columns if col not in cols_to_drop_y3 and col != 'y']
print("Các thuộc tính bị loại bỏ thủ công:", cols_to_drop_y3)
print("Tập thuộc tính được đề xuất:", features_y3)

# --- Yêu cầu 4 ---
print("\n" + "="*40)
print("--- Yêu cầu 4: Chia dữ liệu và Lựa chọn đặc trưng tự động ---")

X = df.drop(columns=['y'])
y = df['y']

# Chia dữ liệu thành 80% huấn luyện và 20% kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Kích thước tập huấn luyện:", X_train.shape)
print("Kích thước tập kiểm tra:", X_test.shape)

# Lựa chọn đặc trưng bằng SelectKBest (chọn 10 đặc trưng tốt nhất)
k = 10
selector = SelectKBest(score_func=f_classif, k=k)
selector.fit(X_train, y_train)
kbest_mask = selector.get_support()
kbest_features = X_train.columns[kbest_mask].tolist()
print("\nCác thuộc tính được chọn bởi SelectKBest:\n", kbest_features)

# Lựa chọn đặc trưng bằng Feature Importance của Random Forest
rf_fs = RandomForestClassifier(random_state=42)
rf_fs.fit(X_train, y_train)
feature_importances = pd.Series(rf_fs.feature_importances_, index=X_train.columns).sort_values(ascending=False)

print("\nCác thuộc tính sắp xếp theo mức độ quan trọng (Random Forest):\n", feature_importances)

# Vẽ biểu đồ Feature Importances
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.values, y=feature_importances.index, hue=feature_importances.index, palette='viridis', legend=False)
plt.title('Mức độ quan trọng của các đặc trưng (Random Forest)')
plt.xlabel('Mức độ quan trọng')
plt.ylabel('Đặc trưng')
plt.tight_layout()
plt.show()

# Lấy các đặc trưng quan trọng nhất từ Random Forest (chọn 10 đặc trưng top)
rf_top_features = feature_importances.head(k).index.tolist()

print("\nSo sánh kết quả lựa chọn đặc trưng:")
print("- Yêu cầu 3 (Thủ công): Chọn", len(features_y3), "thuộc tính.")
print("- SelectKBest: Chọn", len(kbest_features), "thuộc tính ->", kbest_features)
print("- Random Forest (Top 10): Chọn", len(rf_top_features), "thuộc tính ->", rf_top_features)

print("\nNhận xét sự khác nhau:")
print("-> Yêu cầu 3 dựa trên đánh giá cảm tính và ý nghĩa thực tế của nghiệp vụ (Domain Knowledge).")
print("-> SelectKBest sử dụng kiểm định thống kê (ANOVA F-value) để đánh giá mối quan hệ tuyến tính giữa từng đặc trưng và nhãn.")
print("-> Random Forest Feature Importance đánh giá dựa trên mức độ giảm Gini impurity khi sử dụng đặc trưng đó để phân nhánh, có thể bắt được các mối quan hệ phi tuyến và tương tác giữa các đặc trưng (ví dụ: duration thường luôn có mức quan trọng cao nhất).")

# Sử dụng top 10 features của RF cho phần "sau khi chọn lọc đặc trưng" ở YC5, 6, 7
X_train_selected = X_train[rf_top_features]
X_test_selected = X_test[rf_top_features]

# Chuẩn hóa dữ liệu (cần thiết cho SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_sel_scaled = scaler.fit_transform(X_train_selected)
X_test_sel_scaled = scaler.transform(X_test_selected)

def evaluate_and_print(model, X_tr, y_tr, X_te, y_te, title):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    print(f"\n[{title}]")
    print("Confusion Matrix:\n", confusion_matrix(y_te, y_pred))
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision_score(y_te, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_te, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_te, y_pred):.4f}")
    return acc

# Lưu trữ accuracy để vẽ biểu đồ cuối cùng
accuracies = {}

# --- Yêu cầu 5: SVM ---
print("\n" + "="*40)
print("--- Yêu cầu 5: SVM ---")
# Ghi chú: SVM chạy trên dữ liệu lớn có thể mất nhiều thời gian, ta dùng tập scaled
accuracies['SVM (Trước)'] = evaluate_and_print(SVC(), X_train_scaled, y_train, X_test_scaled, y_test, "SVM (Trước chọn lọc)")
accuracies['SVM (Sau)'] = evaluate_and_print(SVC(), X_train_sel_scaled, y_train, X_test_sel_scaled, y_test, "SVM (Sau chọn lọc)")
print("\nNhận xét SVM: Mô hình SVM tốn rất nhiều thời gian để huấn luyện trên không gian đặc trưng lớn. Sau khi chọn lọc đặc trưng, thời gian huấn luyện giảm đáng kể, trong khi các chỉ số (Accuracy, F1-score) có thể không giảm nhiều, hoặc đôi khi tăng lên do loại bỏ bớt nhiễu.")

# --- Yêu cầu 6: Decision Tree ---
print("\n" + "="*40)
print("--- Yêu cầu 6: Decision Tree ---")
accuracies['Tree (Trước)'] = evaluate_and_print(DecisionTreeClassifier(random_state=42), X_train, y_train, X_test, y_test, "Decision Tree (Trước chọn lọc)")
accuracies['Tree (Sau)'] = evaluate_and_print(DecisionTreeClassifier(random_state=42), X_train_selected, y_train, X_test_selected, y_test, "Decision Tree (Sau chọn lọc)")
print("\nNhận xét Decision Tree: Decision Tree có cơ chế tự lựa chọn đặc trưng khi chia nhánh (splitting). Dù vậy, việc lọc đặc trưng trước giúp cây bớt phức tạp, giảm rủi ro Overfitting trên dữ liệu nhiễu.")

# --- Yêu cầu 7: Random Forest ---
print("\n" + "="*40)
print("--- Yêu cầu 7: Random Forest ---")
accuracies['RF (Trước)'] = evaluate_and_print(RandomForestClassifier(random_state=42), X_train, y_train, X_test, y_test, "Random Forest (Trước chọn lọc)")
accuracies['RF (Sau)'] = evaluate_and_print(RandomForestClassifier(random_state=42), X_train_selected, y_train, X_test_selected, y_test, "Random Forest (Sau chọn lọc)")
print("\nNhận xét Random Forest: Random Forest bản chất là thuật toán Ensemble cực kỳ mạnh mẽ, không bị ảnh hưởng nhiều bởi số lượng đặc trưng lớn và có khả năng chống nhiễu tốt. Kết quả trước và sau khi chọn lọc thường xấp xỉ nhau.")

# Vẽ biểu đồ so sánh Accuracy
plt.figure(figsize=(10, 6))
models = list(accuracies.keys())
acc_scores = list(accuracies.values())
sns.barplot(x=models, y=acc_scores, hue=models, palette='coolwarm', legend=False)
plt.ylim(0.7, 1.0) # Đặt giới hạn trục Y từ 0.7 đến 1.0 để dễ thấy sự khác biệt
plt.title('So sánh độ chính xác (Accuracy) của các mô hình')
plt.ylabel('Accuracy')
for i, v in enumerate(acc_scores):
    plt.text(i, v + 0.005, f"{v:.4f}", ha='center')
plt.tight_layout()
plt.show()

