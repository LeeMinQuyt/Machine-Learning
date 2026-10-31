# ML Algorithms VN

Dự án học Machine Learning qua các thuật toán nền tảng tự cài đặt bằng Python và NumPy.

## Thuật toán
- Linear Regression: hồi quy tuyến tính bằng gradient descent.
- Logistic Regression: phân loại nhị phân bằng sigmoid.
- K-Nearest Neighbors (KNN): phân loại theo láng giềng gần nhất.
- K-Means: phân cụm với khởi tạo k-means++ và thuật toán Lloyd.

## Cài đặt
Yêu cầu Python 3.10 trở lên.
```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e .
```

## Chạy ví dụ
```bash
python examples/linear_regression.py
python examples/logistic_regression.py
python examples/knn.py
python examples/kmeans.py
```

Chạy từ thư mục gốc. Các mô hình chỉ phụ thuộc NumPy và không dùng thư viện ML dựng sẵn.

## Cấu trúc
```text
src/ml_algorithms_vn/supervised/   # Linear/Logistic Regression, KNN
src/ml_algorithms_vn/unsupervised/ # K-Means
examples/                          # Ví dụ độc lập
```
