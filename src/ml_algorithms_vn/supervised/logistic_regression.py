"""Hồi quy logistic cho phân loại nhị phân."""
import numpy as np
from .linear_regression import _validate_xy


class LogisticRegression:
    """Phân loại nhị phân với sigmoid và binary cross-entropy."""
    def __init__(self, learning_rate=0.1, n_iterations=1000, threshold=0.5):
        if learning_rate <= 0 or n_iterations < 1 or not 0 < threshold < 1:
            raise ValueError("Tham số không hợp lệ")
        self.learning_rate, self.n_iterations, self.threshold = learning_rate, n_iterations, threshold
        self.weights, self.bias, self.loss_history = None, 0.0, []

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        X, y = _validate_xy(X, y)
        if not np.isin(y, [0, 1]).all():
            raise ValueError("y chỉ được chứa nhãn 0 và 1")
        n, d = X.shape
        self.weights, self.bias, self.loss_history = np.zeros(d), 0.0, []
        for _ in range(self.n_iterations):
            p = self._sigmoid(X @ self.weights + self.bias)
            error = p - y
            self.weights -= self.learning_rate * (X.T @ error) / n
            self.bias -= self.learning_rate * float(error.mean())
            p = np.clip(p, 1e-12, 1 - 1e-12)
            self.loss_history.append(float(-np.mean(y * np.log(p) + (1-y) * np.log(1-p))))
        return self

    def predict_proba(self, X):
        if self.weights is None:
            raise RuntimeError("Hãy gọi fit() trước predict_proba()")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.weights.size:
            raise ValueError("X cần có cùng số đặc trưng như dữ liệu huấn luyện")
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X):
        return (self.predict_proba(X) >= self.threshold).astype(int)
