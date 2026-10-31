"""Hồi quy tuyến tính bằng batch gradient descent."""
import numpy as np


def _validate_xy(X, y):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[0] != y.size:
        raise ValueError("X phải là ma trận không rỗng và số hàng bằng độ dài y")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("X và y chỉ được chứa giá trị hữu hạn")
    return X, y


class LinearRegression:
    """Mô hình y = Xw + b."""
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        if learning_rate <= 0 or n_iterations < 1:
            raise ValueError("learning_rate phải > 0 và n_iterations phải >= 1")
        self.learning_rate, self.n_iterations = learning_rate, n_iterations
        self.weights, self.bias, self.loss_history = None, 0.0, []

    def fit(self, X, y):
        X, y = _validate_xy(X, y)
        n, d = X.shape
        self.weights, self.bias, self.loss_history = np.zeros(d), 0.0, []
        for _ in range(self.n_iterations):
            error = X @ self.weights + self.bias - y
            self.weights -= self.learning_rate * (2 / n) * (X.T @ error)
            self.bias -= self.learning_rate * 2 * float(error.mean())
            self.loss_history.append(float(np.mean(error ** 2)))
        return self

    def predict(self, X):
        if self.weights is None:
            raise RuntimeError("Hãy gọi fit() trước predict()")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.weights.size:
            raise ValueError("X cần có cùng số đặc trưng như dữ liệu huấn luyện")
        return X @ self.weights + self.bias
