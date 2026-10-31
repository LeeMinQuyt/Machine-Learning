"""Phân loại nhị phân theo số giờ học."""
import numpy as np
from ml_algorithms_vn import LogisticRegression

X = np.array([[1],[2],[3],[4],[5],[6]], dtype=float)
y = np.array([0,0,0,1,1,1])
model = LogisticRegression(learning_rate=0.1, n_iterations=3000).fit(X,y)
print("Xác suất lớp 1:", np.round(model.predict_proba([[2.5],[4.5]]),3))
print("Nhãn dự đoán:", model.predict([[2.5],[4.5]]))
