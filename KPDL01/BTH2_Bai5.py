import pandas as pd
# Thư viện scikit-learn (sklearn) cung cấp các công cụ tiền xử lý dữ liệu mạnh mẽ.
# - OrdinalEncoder: Dùng để mã hóa các biến phân loại có thứ tự (ví dụ: Học vấn).
# - MinMaxScaler: Dùng để chuẩn hóa các cột số về cùng một khoảng giá trị từ 0 đến 1.
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler 

# Đọc file dữ liệu CSV từ đường dẫn được chỉ định và lưu vào một cấu trúc bảng gọi là DataFrame (df).
df = pd.read_csv("D:/Mixed_Data.csv")


# =====================================================================
# BƯỚC 1: MÃ HÓA CÁC CỘT BIẾN PHÂN LOẠI NHỊ PHÂN (BINARY CATEGORICAL)
# =====================================================================
# Các cột này chỉ có đúng 2 trạng thái (Male/Female, No/Yes). 
# Hàm .map() sẽ dò từng dòng trong cột và thay thế chữ thành số theo "từ điển" {Key: Value} được truyền vào.

# Tìm chữ 'Male' đổi thành 0, 'Female' đổi thành 1. Ghi đè lại vào cột 'Gender'.
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Tìm chữ 'No' đổi thành 0, 'Yes' đổi thành 1. Ghi đè lại vào cột 'Married'.
df['Married'] = df['Married'].map({'No': 0, 'Yes': 1})


# =====================================================================
# BƯỚC 2: ONE-HOT ENCODING CHO DỮ LIỆU ĐỊNH DANH KHÔNG THỨ TỰ (NOMINAL)
# =====================================================================
# Cột 'City' chứa tên các thành phố (ví dụ: Hanoi, HCM, Danang). Chúng không có tính hơn kém.
# Nếu đổi thành số 0, 1, 2, mô hình sẽ hiểu nhầm là 2 > 1 > 0 (HCM > Hanoi).
# Hàm pd.get_dummies() sẽ tách 1 cột 'City' thành N cột mới tương ứng với N thành phố.
# Dòng nào thuộc thành phố nào thì cột đó sẽ mang giá trị 1 (True), các cột khác mang giá trị 0 (False).

nominal_columns = ['City'] # Khai báo danh sách các cột cần làm One-Hot
df = pd.get_dummies(df, columns=nominal_columns) # Tiến hành tách cột và gán lại vào df


# =====================================================================
# BƯỚC 3: ORDINAL ENCODING CHO DỮ LIỆU ĐỊNH DANH CÓ THỨ TỰ (ORDINAL)
# =====================================================================
# Cột 'Education' có tính chất thứ bậc tăng dần: Cấp 3 < Cao đẳng < Đại học < Thạc sĩ.
# Do đó, ta cần gán số tăng dần (0, 1, 2, 3) để giữ nguyên mối quan hệ cấp bậc này.

ordinal_encoder = OrdinalEncoder(
    # categories: Định nghĩa chính xác thứ tự từ thấp đến cao. Thằng đứng trước ăn số nhỏ, đứng sau ăn số lớn.
    categories=[['HighSchool', 'College', 'University', 'Master']],
    
    # handle_unknown='use_encoded_value': Phòng hờ nếu trong file CSV xuất hiện từ lạ (ví dụ: 'PhD' hoặc gõ sai chính tả).
    # Thay vì làm chương trình bị sập (báo lỗi), nó sẽ tự động bỏ qua và gán cho một giá trị mặc định.
    handle_unknown='use_encoded_value',
    
    # unknown_value=-1: Quy định giá trị mặc định cho những từ lạ/lỗi ở trên là số -1.
    unknown_value=-1
)

# fit_transform(): Bắt đầu quét qua cột 'Education' để học thứ tự (fit) và tiến hành đổi chữ thành số (transform).
# Lưu ý: Thao tác df[['Education']] (2 dấu ngoặc vuông) là bắt buộc để biến cột này thành một bảng 2 chiều, đúng định dạng mà Sklearn yêu cầu.
df['Education'] = ordinal_encoder.fit_transform(df[['Education']])


# =====================================================================
# BƯỚC 4: XỬ LÝ GIÁ TRỊ KHUYẾT THIẾU (MISSING VALUES / NaN)
# =====================================================================
# Các mô hình toán học và hàm MinMaxScaler KHÔNG THỂ tính toán nếu dữ liệu có ô trống (NaN).
# Do đó, ta cần lấp đầy (điền) các ô trống này trước khi scale.

numeric_columns = ['Age', 'Income', 'Score', 'Education'] # Gom các cột dạng số cần xử lý vào một danh sách

# df[numeric_columns].median(): Tính toán giá trị trung vị (giá trị nằm ở chính giữa tập dữ liệu) của từng cột số.
# .fillna(): Tìm tất cả các ô bị trống (NaN) trong các cột này và điền giá trị trung vị vừa tính được vào đó.
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())


# =====================================================================
# BƯỚC 5: CHUẨN HÓA DỮ LIỆU SỐ (FEATURE SCALING / MINMAXSCALER)
# =====================================================================
# Cột 'Income' (Thu nhập) có thể lên tới hàng triệu, trong khi 'Age' (Tuổi) chỉ từ 20-60.
# Nếu giữ nguyên, các thuật toán sẽ bị đánh lừa rằng cột Income quan trọng hơn Age vì số nó to hơn.
# MinMaxScaler sẽ đưa tất cả các cột số này về chung một cái "thước đo" từ 0 đến 1 bằng công thức: (x - min) / (max - min)

minmax = MinMaxScaler() # Khởi tạo công cụ MinMaxScaler với cấu hình mặc định là đưa về khoảng [0, 1]

# fit_transform(): Tìm ra giá trị lớn nhất (Max) và nhỏ nhất (Min) của từng cột (fit), 
# sau đó áp dụng công thức để biến đổi toàn bộ dữ liệu trong các cột đó về khoảng 0 và 1 (transform).
# Giải thích cú pháp: Vì numeric_columns bản thân nó đã là một List ['Age', 'Income',...], 
# nên ta chỉ cần viết df[numeric_columns] (1 cặp ngoặc vuông đại diện cho việc chọn cột trong DataFrame).
df[numeric_columns] = minmax.fit_transform(df[numeric_columns])


# =====================================================================
# BƯỚC 6: XUẤT KẾT QUẢ CUỐI CÙNG
# =====================================================================
# In toàn bộ bảng dữ liệu (DataFrame) sau khi đã trải qua tất cả các bước làm sạch và chuẩn hóa ở trên.
print(df)