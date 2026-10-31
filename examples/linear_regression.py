"""Dự đoán điểm số theo số giờ học."""
import numpy as np
from ml_algorithms_vn import LinearRegression

X = np.array([[1],[2],[3],[4],[5]], dtype=float)
y = np.array([52,58,65,71,78], dtype=float)
model = LinearRegression(learning_rate=0.01, n_iterations=5000).fit(X,y)
print("Dự đoán điểm khi học 6 giờ:", round(float(model.predict([[6]])[0]),2))
print("MSE cuối:", round(model.loss_history[-1],4))
