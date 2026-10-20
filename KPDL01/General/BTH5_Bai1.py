# Nguyễn Thị Ngọc Ngoan - 2351050114
# Võ Thị Ngọc Tiên - 2351050176
# Nguyễn Huỳnh Như Ý - 2351050211

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as mpl
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, \
    classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

pd.set_option('display.max_columns', None) #khong gioi han so cot hien thi trong kq

df = pd.read_csv('bank-full.csv')
print(df.isnull().sum()) #tong so hang co gtri rong
df = df.dropna()

print(df.duplicated().sum()) #tong so dong trung
df = df.drop_duplicates()

print(df.info()) #kieu DL cua cot

binary_cols = ['housing', 'loan', 'y'] #cot y phai dua ve dang so
binary_map = {
    'yes': 1,
    'no': 0
}
for c in binary_cols:
    df[c] = df[c].map(binary_map)

marital_map = {
    'single': 0,
    'married': 1,
    'divorced': 2
}
df['marital'] = df['marital'].map(marital_map)

#print(df['education'].value_counts()) # xem cot co bao nhieu loai gtri
edu_map = {
    'unknown': 0,
    'primary': 1,
    'secondary': 2,
    'tertiary': 3
}
df['education'] = df['education'].map(edu_map)

month_map = {
    'jan':0,
    'feb':1,
    'mar':2,
    'apr':3,
    'may':4,
    'jun':5,
    'jul':6,
    'aug':7,
    'sep':8,
    'oct':9,
    'nov':10,
    'dec':11
}
df['month'] = df['month'].map(month_map)

df = df.drop(columns=['job']) #xoa cot

print('So luong khach hang: ', len(df))
print('So luong khach hang cua tung lop: ')
print(df['y'].value_counts())

# Vẽ biểu đồ thể hiện số lượng khách hàng thuộc hai lớp.
class_counts = df['y'].value_counts()
plt.figure(figsize=(7, 5))
bars = plt.bar(class_counts.index, class_counts.values) #ten lop, so luong
plt.title('Bieu do so luong tung lop')
plt.xlabel('Ten lop')
plt.ylabel('So luong')
plt.show()

# Tách tập dữ liệu thành dữ liệu đầu vào X và nhãn y.
X = df.drop(columns=['y'])
y = df['y']

# Ma trận tương quan giữa các thuộc tính số
corr_matrix = df.corr(numeric_only=True)
print('Ma tran tuong quan')
print(corr_matrix)
# Ma tran tuong quan: duration, previous, balance, education, month

plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(label='He so tuong quan Pearson')
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha='right')
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title('Ma tran tuong quan Pearson')
plt.tight_layout()
plt.show()

# Chia dữ liệu thành 80% huấn luyện và 20% kiểm tra.
# SelectKBest
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

selector = SelectKBest(score_func=f_classif, k=5)
selector.fit(X_train, y_train)
selected_kbest = X_train.columns[selector.get_support()]
kbest_scores = pd.Series(selector.scores_, index=X_train.columns).sort_values(ascending=False)
print("SelectKBest")
print(selected_kbest)
print(kbest_scores)
# SelectKBest: duration, housing, previous, campaign, loan

# Feature Importance của Random Forest
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_importance = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False) #sap xep giam dan
print('Random Forest')
print(rf_importance)
# Random Forest: duration, balance, month, age, day

def evaluate_svm(Xtr, Xte, ytr, yte):
    scaler = StandardScaler()
    Xtr_scaled = scaler.fit_transform(Xtr)
    Xte_scaled = scaler.transform(Xte)

    model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    model.fit(Xtr_scaled, ytr)
    y_pred = model.predict(Xte_scaled)

    cm = confusion_matrix(yte, y_pred)
    print(cm)
    print('Accuracy: ', accuracy_score(yte, y_pred))
    print('Precision: ', precision_score(yte, y_pred))
    print('Recall: ', recall_score(yte, y_pred))
    print('F1: ', f1_score(yte, y_pred))
    print(classification_report(yte, y_pred))

def evaluate_dt(Xtr, Xte, ytr, yte):
    scaler = StandardScaler()
    Xtr_scaled = scaler.fit_transform(Xtr)
    Xte_scaled = scaler.transform(Xte)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(Xtr_scaled, ytr)
    y_pred = model.predict(Xte_scaled)

    cm = confusion_matrix(yte, y_pred)
    print(cm)
    print('Accuracy: ', accuracy_score(yte, y_pred))
    print('Precision: ', precision_score(yte, y_pred))
    print('Recall: ', recall_score(yte, y_pred))
    print('F1: ', f1_score(yte, y_pred))
    print(classification_report(yte, y_pred))

def evaluate_rf(Xtr, Xte, ytr, yte):
    scaler = StandardScaler()
    Xtr_scaled = scaler.fit_transform(Xtr)
    Xte_scaled = scaler.transform(Xte)

    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(Xtr_scaled, ytr)
    y_pred = model.predict(Xte_scaled)

    cm = confusion_matrix(yte, y_pred)
    print(cm)
    print('Accuracy: ', accuracy_score(yte, y_pred))
    print('Precision: ', precision_score(yte, y_pred))
    print('Recall: ', recall_score(yte, y_pred))
    print('F1: ', f1_score(yte, y_pred))
    print(classification_report(yte, y_pred))

print("Ket qua SVM voi 11 cot: ")
evaluate_svm(X_train, X_test, y_train, y_test)

corr_list = ['duration', 'previous', 'balance', 'education', 'month']
print("Ket qua SVM voi ma tran tuong quan: ")
evaluate_svm(X_train[corr_list], X_test[corr_list], y_train, y_test)

selectk_list = ['duration', 'housing', 'previous', 'campaign', 'loan']
print("Ket qua SVM voi SelectKBest: ")
evaluate_svm(X_train[selectk_list], X_test[selectk_list], y_train, y_test)

rf_list = ['duration', 'balance', 'month', 'age', 'day']
print("Ket qua SVM voi RF: ")
evaluate_svm(X_train[rf_list], X_test[rf_list], y_train, y_test)

custom_list = ['duration', 'balance', 'month', 'age', 'day', 'housing', 'previous', 'education']
print("Ket qua SVM voi custom: ")
evaluate_svm(X_train[custom_list], X_test[custom_list], y_train, y_test)