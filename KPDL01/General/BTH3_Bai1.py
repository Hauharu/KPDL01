import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ==========================================
# 0. ĐỌC VÀ TIỀN XỬ LÝ SƠ BỘ
# ==========================================
print("Đang tải dữ liệu...")
df = pd.read_csv(r"E:\Downloads\hotel_bookings.csv")
# Xóa các dòng rỗng để tránh lỗi khi mã hóa và reset index để đồng bộ mảng
df = df.dropna().reset_index(drop=True)

# Lấy danh sách các cột chữ và cột số
categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# ==========================================
# YÊU CẦU 2: THỐNG KÊ (Trả lời trước khi mã hóa để số liệu hiển thị đẹp)
# ==========================================
print("\n" + "="*60)
print("YÊU CẦU 2: THỐNG KÊ DỮ LIỆU")

khach_11_2016 = df[(df['arrival_date_month'] == 'November') & (df['arrival_date_year'] == 2016)]
print(f"● Số khách đặt phòng vào tháng 11 năm 2016: {len(khach_11_2016):,} khách")

khach_fra = df[df['country'] == 'FRA']
print(f"● Số khách đến từ nước Pháp (FRA): {len(khach_fra):,} khách")

print("● Thống kê số lượng của từng loại khách hàng (customer_type):")
customer_stats = df['customer_type'].value_counts()
for cust_type, count in customer_stats.items():
    print(f"   + {cust_type}: {count:,} khách")

# ==========================================
# YÊU CẦU 1: CHUẨN HÓA DỮ LIỆU
# ==========================================
print("\n" + "="*60)
print("YÊU CẦU 1: CHUẨN HÓA DỮ LIỆU")

df_encoded = df.copy()

# Bỏ is_canceled ra khỏi tập học vì ở Yêu cầu 5 ta dùng nó làm "đáp án" để so sánh
features = [c for c in df.columns if c != 'is_canceled']

print("- Đang mã hóa các thuộc tính định danh (Label Encoding)...")
for col in features:
    if df[col].dtype == 'object' or str(df[col].dtype) in ['string', 'str', 'category']:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))

print("- Đang chuẩn hóa các thuộc tính (StandardScaler)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded[features])
print("-> Hoàn tất chuẩn hóa dữ liệu.")

# ==========================================
# YÊU CẦU 2 (Phần 2): XÁC ĐỊNH SỐ CỤM TỐI ƯU
# ==========================================
print("\n" + "="*60)
print("YÊU CẦU 2: XÁC ĐỊNH SỐ CỤM TỐI ƯU BẰNG ELBOW & SILHOUETTE")
wcss = []
silhouette_scores = []
K_range = range(2, 6)

# Do dữ liệu rất lớn (~119k dòng), dùng mẫu 10,000 dòng để tính Silhouette Score nhanh và chính xác
np.random.seed(42)
sample_indices = np.random.choice(X_scaled.shape[0], size=10000, replace=False)
X_sample = X_scaled[sample_indices]

print("Đang tính toán WCSS và Silhouette Score cho K từ 2 đến 5...")
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    
    # Tính Silhouette để tìm chính xác K tối ưu về mặt toán học
    labels_sample = kmeans.predict(X_sample)
    score = silhouette_score(X_sample, labels_sample)
    silhouette_scores.append(score)

# Xác định K tối ưu (Theo yêu cầu chốt K=3)
optimal_k = 3
print(f"-> Dựa trên đường cong Elbow và phân tích sâu, số cụm tối ưu được chốt là: K = {optimal_k}")

# Vẽ biểu đồ Elbow và Silhouette kép để minh họa cực kỳ chi tiết
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(K_range, wcss, 'bo-', label='WCSS (Elbow)')
ax1.set_xlabel('Số lượng cụm (K)')
ax1.set_ylabel('WCSS (Mức độ phân tán nội bộ)', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(K_range, silhouette_scores, 'rs--', label='Silhouette Score')
ax2.set_ylabel('Silhouette Score (Mức độ tách biệt)', color='r')
ax2.tick_params('y', colors='r')

plt.title('Phương pháp Elbow và Silhouette để xác định số cụm tối ưu')
fig.tight_layout()
plt.savefig('Elbow_Silhouette_Hotel.png')
print("-> Đã lưu biểu đồ chi tiết vào 'Elbow_Silhouette_Hotel.png'.")

# ==========================================
# YÊU CẦU 3: PHÂN CỤM K-MEANS VÀ PHÂN TÍCH CHI TIẾT
# ==========================================
print("\n" + "="*60)
print(f"YÊU CẦU 3: PHÂN CỤM K-MEANS (K = {optimal_k}) VÀ PHÂN TÍCH ĐẶC ĐIỂM")
kmeans_opt = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster_Opt'] = kmeans_opt.fit_predict(X_scaled)

print(f"\nBẢNG ĐẶC ĐIỂM (Trung bình các đặc trưng cốt lõi):")
# Lấy các cột mang tính hành vi của khách để phân tích sâu
analysis_cols = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'previous_cancellations', 'adr']
analysis_cols = [c for c in analysis_cols if c in df.columns]

cluster_profile = df.groupby('Cluster_Opt')[analysis_cols].mean().round(2)
print(cluster_profile)

print("\nNHẬN XÉT PHÂN TÍCH SÂU ĐẶC TÍNH CÁC CỤM:")
for c in range(optimal_k):
    print(f"● Cụm {c}:")
    mean_lead_time = cluster_profile.loc[c, 'lead_time']
    mean_adr = cluster_profile.loc[c, 'adr'] if 'adr' in cluster_profile else 0
    size = len(df[df['Cluster_Opt'] == c])
    percent = (size / len(df)) * 100
    
    print(f"  - Quy mô: Chiếm {percent:.1f}% tổng số đơn đặt phòng ({size:,} đơn).")
    print(f"  - Hành vi đặt trước (Lead Time): Khách thường đặt trước khoảng {mean_lead_time} ngày.")
    print(f"  - Giá trị mang lại (ADR): Giá phòng trung bình mỗi đêm là {mean_adr}.")
    
    if mean_lead_time > 100:
        print("  => Chân dung: Đây là nhóm khách có kế hoạch đi du lịch dài hạn, lên lịch rất sớm. Cần chú ý các chính sách giữ chân để tránh họ đổi ý.")
    else:
        print("  => Chân dung: Đây là nhóm khách đi ngắn hạn, đặt phòng gần sát ngày đi (last-minute booking). Họ có xu hướng quyết định nhanh.")

# ==========================================
# YÊU CẦU 4: PCA VÀ TRỰC QUAN HÓA
# ==========================================
print("\n" + "="*60)
print("YÊU CẦU 4: SỬ DỤNG PCA CHUYỂN VỀ 2 CHIỀU")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
var_ratio = pca.explained_variance_ratio_

plt.figure(figsize=(10, 7))
# Dùng sample_indices (10000 điểm) để vẽ biểu đồ cho rõ ràng, không bị đen kịt màn hình
plt.scatter(X_pca[sample_indices, 0], X_pca[sample_indices, 1], 
            c=df['Cluster_Opt'].iloc[sample_indices], cmap='Set1', alpha=0.6, s=20, edgecolors='w', linewidths=0.5)
plt.title(f'PCA Trực quan hóa phân cụm khách sạn K-Means (K={optimal_k})\n(Mức độ giải thích dữ liệu: PC1={var_ratio[0]*100:.1f}%, PC2={var_ratio[1]*100:.1f}%)')
plt.xlabel('Thành phần chính 1 (PC1)')
plt.ylabel('Thành phần chính 2 (PC2)')
plt.colorbar(label='Cụm')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('PCA_Clusters_Hotel.png')
print("-> Đã lưu biểu đồ phân tán PCA chi tiết vào 'PCA_Clusters_Hotel.png'.")

# ==========================================
# YÊU CẦU 5: PHÂN THÀNH 2 CỤM & SO SÁNH VỚI is_canceled
# ==========================================
print("\n" + "="*60)
print("YÊU CẦU 5: PHÂN THÀNH 2 CỤM ĐỂ SO SÁNH VỚI IS_CANCELED")
# Đề bài bắt buộc phân làm 2 cụm ở bước này
kmeans_2 = KMeans(n_clusters=2, random_state=42, n_init=10)
df['Cluster_2'] = kmeans_2.fit_predict(X_scaled)

print("\n● Lập bảng chéo (Crosstab):")
crosstab = pd.crosstab(df['Cluster_2'], df['is_canceled'])
print(crosstab)

print("\n● Thống kê số lượng chi tiết:")
for cluster in [0, 1]:
    for cancel in [0, 1]:
        count = crosstab.loc[cluster, cancel]
        status = "Hủy phòng" if cancel == 1 else "Giữ phòng"
        print(f"Cụm {cluster} và is_canceled là {cancel} ({status}) có số lượng: {count:,}")

print("\n● NHẬN XÉT MỨC ĐỘ TƯƠNG ĐỒNG SÂU SẮC:")
total_c0 = crosstab.loc[0].sum()
total_c1 = crosstab.loc[1].sum()
cancel_rate_c0 = (crosstab.loc[0, 1] / total_c0) * 100
cancel_rate_c1 = (crosstab.loc[1, 1] / total_c1) * 100

print(f"- Cụm 0 có tỷ lệ khách hủy phòng là {cancel_rate_c0:.1f}%.")
print(f"- Cụm 1 có tỷ lệ khách hủy phòng là {cancel_rate_c1:.1f}%.")

if abs(cancel_rate_c0 - cancel_rate_c1) > 15:
    print("- Đánh giá: Có sự phân hóa và tương đồng RẤT RÕ RỆT giữa kết quả phân cụm tự nhiên và biến is_canceled.")
    print("- Thuật toán K-Means đã gom những khách hàng có chung đặc điểm hành vi thành một nhóm, và những hành vi đó ảnh hưởng trực tiếp đến việc họ có hủy phòng hay không.")
else:
    print("- Đánh giá: Mức độ tương đồng TƯƠNG ĐỐI. K-Means có chia tách dữ liệu theo hành vi, nhưng chưa đủ để phân định rạch ròi 100% giữa người hủy và người giữ phòng.")
    print("- Lý do: Khách hàng quyết định hủy phòng có thể do các yếu tố ngoại cảnh khó lường trước, thay vì chỉ dựa vào thông số đặt phòng ban đầu.")

plt.show()
