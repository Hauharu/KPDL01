import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler

def evaluate_model(y_true, y_pred, model_name="Mô hình"): 
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    tp_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    print(f"--- Kết quả đánh giá {model_name} ---")
    print("Confusion matrix:\n", cm)
    print("TP Rate:   ", round(tp_rate, 4))
    print("FP Rate:   ", round(fp_rate, 4))
    print("Accuracy:  ", round(accuracy, 4))
    print("Precision: ", round(precision, 4))
    print("Recall:    ", round(recall, 4))
    print("F1-score:  ", round(f1, 4))
    print("-" * 40)
    return accuracy

def find_best_knn(X_train, y_train):
    print("\n" + "="*50)
    print("KHẢO SÁT THUẬT TOÁN KNN TRÊN TẬP TRAIN (90%)")
    print("="*50)

    k_values = [1, 3, 5, 7, 9, 11, 13, 15]
    
    # 1. Phương pháp % Split (Trích ra 30% từ tập Train để validation nội bộ)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.3, stratify=y_train, random_state=42)
    
    best_k_split = 1
    best_acc_split = 0
    
    print("\n[ Phương pháp 1: % Split (70/30 của TrainData) ]")
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_tr, y_tr)
        y_pred = knn.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        if acc > best_acc_split:
            best_acc_split = acc
            best_k_split = k
            
    print(f"-> % Split - K tốt nhất = {best_k_split} (Accuracy: {round(best_acc_split, 4)})")

    # 2. Phương pháp 10-Fold CV trên tập TrainData
    best_k_cv = 1
    best_acc_cv = 0
    print("\n[ Phương pháp 2: 10-Fold Cross Validation trên TrainData ]")
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        y_pred_cv = cross_val_predict(knn, X_train, y_train, cv=10)
        acc = accuracy_score(y_train, y_pred_cv)
        if acc > best_acc_cv:
            best_acc_cv = acc
            best_k_cv = k
            
    print(f"-> 10-Fold CV - K tốt nhất = {best_k_cv} (Accuracy: {round(best_acc_cv, 4)})")

    # So sánh mô hình tốt nhất trong lúc huấn luyện
    print("\n=> KẾT LUẬN MÔ HÌNH TỐT NHẤT LÚC HUẤN LUYỆN:")
    if best_acc_split > best_acc_cv:
        print(f"Chọn K = {best_k_split} (từ % Split) vì có accuracy cao hơn ({round(best_acc_split,4)} > {round(best_acc_cv,4)})")
        return best_k_split
    elif best_acc_cv > best_acc_split:
        print(f"Chọn K = {best_k_cv} (từ 10-Fold CV) vì có accuracy cao hơn ({round(best_acc_cv,4)} > {round(best_acc_split,4)})")
        return best_k_cv
    else:
        print(f"Cả hai đều cho kết quả tốt ngang nhau. Chọn K = {best_k_cv}")
        return best_k_cv

if __name__ == "__main__":
    # Đọc tập tin dữ liệu
    df = pd.read_csv("bank-data.csv")

    # 1. Xóa vùng định danh (cột 'id')
    if 'id' in df.columns: 
        df = df.drop(columns=['id'])

    # 2. Mã hóa dữ liệu phân loại (categorical)
    sex_map = {'FEMALE': 0, 'MALE': 1}
    df['sex'] = df['sex'].map(sex_map)

    binary_map = {'YES': 1, 'NO': 0}
    binary_cols = ['married', 'car', 'save_act', 'current_act', 'mortgage', 'pep']
    for col in binary_cols: 
        df[col] = df[col].map(binary_map)
        
    region_map = {'INNER_CITY': 0, 'TOWN': 1, 'RURAL': 2, 'SUBURBAN': 3} 
    df['region'] = df['region'].map(region_map)

    df = df.dropna()

    # Phân tách đặc trưng (X) và nhãn mục tiêu (y)
    X = df.drop('pep', axis=1)
    y = df['pep']

    # 3. TÁCH TẬP TIN (Train 90% - Test 10%)
    # Sử dụng stratify=y để đảm bảo tỷ lệ YES/NO cân bằng ở cả 2 tập tin
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, stratify=y, random_state=42)

    # (Tùy chọn) Lưu ra file để đáp ứng đúng câu chữ "tách thành 2 tập tin"
    TrainData = pd.concat([X_train, y_train], axis=1)
    TestData = pd.concat([X_test, y_test], axis=1)
    TrainData.to_csv("TrainData.csv", index=False)
    TestData.to_csv("TestData.csv", index=False)
    print("Đã tách và lưu thành 2 file: TrainData.csv (90%) và TestData.csv (10%).")

    # 4. CHUẨN HÓA DỮ LIỆU
    # LƯU Ý: Chỉ fit scaler trên tập Train, sau đó transform cho cả Train và Test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Khôi phục thành DataFrame (tùy chọn, để code nhất quán)
    X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

    # 5. KHẢO SÁT & TÌM MÔ HÌNH KNN TỐT NHẤT TRÊN TẬP TRAIN
    best_k = find_best_knn(X_train, y_train)

    # 6. KIỂM TRA MÔ HÌNH KNN TỐT NHẤT TRÊN TẬP TESTDATA
    print("\n" + "="*50)
    print(f"KIỂM TRA TRÊN TẬP TESTDATA (10%)")
    print("="*50)
    
    # Train lại mô hình bằng toàn bộ 90% TrainData với Best K
    final_knn = KNeighborsClassifier(n_neighbors=best_k)
    final_knn.fit(X_train, y_train)
    y_pred_knn = final_knn.predict(X_test)
    
    print("\n1. Đánh giá thuật toán KNN:")
    evaluate_model(y_test, y_pred_knn, model_name=f"KNN (K={best_k})")

    # 7. THỰC HIỆN TƯƠNG TỰ VỚI NAIVE BAYES
    print("\n2. Đánh giá thuật toán Naive Bayes:")
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)
    
    evaluate_model(y_test, y_pred_nb, model_name="Naive Bayes (GaussianNB)")

    print("\n=> NHẬN XÉT:")
    print("- Mô hình KNN và Naive Bayes được huấn luyện hoàn toàn độc lập với tập TestData.")
    print("- Dựa trên các chỉ số (Accuracy, F1-score...), ta có thể dễ dàng thấy mô hình nào hoạt động tốt hơn trên dữ liệu thực tế ở bảng đánh giá phía trên.")
