"""Phân loại trái cây từ hai đặc trưng."""
import numpy as np
from ml_algorithms_vn import KNNClassifier

X = np.array([[1,2],[1.5,1.8],[5,8],[6,9],[1,0.6],[6,7]])
y = np.array(["táo","táo","cam","cam","táo","cam"])
model = KNNClassifier(n_neighbors=3).fit(X,y)
print("Nhãn cho điểm [2, 2]:", model.predict([[2,2]])[0])
