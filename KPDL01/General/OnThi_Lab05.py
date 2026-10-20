import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ==========================================
# CHỈ CẦN THAY ĐỔI 2 GIÁ TRỊ NÀY KHI THI:
TEN_FILE = 'bank-full.csv'
COT_NHAN = 'y'
# ==========================================

# --- BƯỚC 1: TIỀN XỬ LÝ DỮ LIỆU & EDA ---

# 1. Đọc dữ liệu
df = pd.read_csv(TEN_FILE)

# 2. Xử lý NaN và Trùng lặp
df = df.dropna().drop_duplicates()

# 3. Mã hóa dữ liệu TỰ ĐỘNG
categorical_cols = df.select_dtypes(include=['object', 'string']).columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# 4. Thống kê số lượng khách hàng
print('Tổng số lượng khách hàng: ', len(df))
print(f'Số lượng khách hàng từng lớp {COT_NHAN}:\n', df[COT_NHAN].value_counts())

# Vẽ biểu đồ Bar Chart
class_counts = df[COT_NHAN].value_counts()
plt.figure(figsize=(7, 5))
plt.bar(class_counts.index.astype(str), class_counts.values, color=['skyblue', 'salmon'])
plt.title(f'Số lượng khách hàng theo lớp {COT_NHAN}')
plt.xlabel('Lớp phân loại')
plt.ylabel('Số lượng')
plt.show()

# --- BƯỚC 2: MA TRẬN TƯƠNG QUAN & HEATMAP ---

X = df.drop(columns=[COT_NHAN])
y = df[COT_NHAN]

corr_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Ma trận tương quan Pearson')
plt.tight_layout()
plt.show()

# --- BƯỚC 3: CHIA TẬP VÀ LỰA CHỌN ĐẶC TRƯNG TỰ ĐỘNG ---

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# SelectKBest
selector = SelectKBest(score_func=f_classif, k=5)
selector.fit(X_train, y_train)
kbest_features = X_train.columns[selector.get_support()].tolist()
print("\n--- 5 thuộc tính tốt nhất theo SelectKBest ---")
print(kbest_features)

# Random Forest Feature Importance
rf_fs = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_fs.fit(X_train, y_train)
importances = pd.Series(rf_fs.feature_importances_, index=X_train.columns).sort_values(ascending=False)
rf_top_features = importances.head(5).index.tolist()
print("\n--- 5 thuộc tính quan trọng nhất theo Random Forest ---")
print(importances.head(5))

# Lưu lại tập dữ liệu đã chọn lọc (Dùng của RF)
X_train_selected = X_train[rf_top_features]
X_test_selected = X_test[rf_top_features]


# --- BƯỚC 4: ĐÁNH GIÁ MÔ HÌNH TRƯỚC VÀ SAU CHỌN LỌC ---

def evaluate_and_print(model, X_tr, y_tr, X_te, y_te, title):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    model.fit(X_tr_s, y_tr)
    y_pred = model.predict(X_te_s)
    
    print(f"\n========== {title} ==========")
    print("Confusion Matrix:\n", confusion_matrix(y_te, y_pred))
    print(f"Accuracy : {accuracy_score(y_te, y_pred):.4f}")
    print(f"Precision: {precision_score(y_te, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_te, y_pred, zero_division=0):.4f}")
    print(f"F1-score : {f1_score(y_te, y_pred, zero_division=0):.4f}")

# Chạy SVM
# Lưu ý: Nếu dữ liệu lớn SVM chạy hơi chậm
evaluate_and_print(SVC(kernel='rbf', random_state=42), X_train, y_train, X_test, y_test, "SVM - FULL ĐẶC TRƯNG")
evaluate_and_print(SVC(kernel='rbf', random_state=42), X_train_selected, y_train, X_test_selected, y_test, "SVM - SAU CHỌN LỌC (TOP 5)")

# Chạy Decision Tree
evaluate_and_print(DecisionTreeClassifier(random_state=42), X_train, y_train, X_test, y_test, "DECISION TREE - FULL ĐẶC TRƯNG")
evaluate_and_print(DecisionTreeClassifier(random_state=42), X_train_selected, y_train, X_test_selected, y_test, "DECISION TREE - SAU CHỌN LỌC (TOP 5)")

# Chạy Random Forest
evaluate_and_print(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), X_train, y_train, X_test, y_test, "RANDOM FOREST - FULL ĐẶC TRƯNG")
evaluate_and_print(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), X_train_selected, y_train, X_test_selected, y_test, "RANDOM FOREST - SAU CHỌN LỌC (TOP 5)")
