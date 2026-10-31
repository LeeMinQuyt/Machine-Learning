"""K-Nearest Neighbors cho phân loại."""
import numpy as np


class KNNClassifier:
    """Dự đoán nhãn theo đa số trong k láng giềng gần nhất."""
    def __init__(self, n_neighbors=3):
        if n_neighbors < 1:
            raise ValueError("n_neighbors phải >= 1")
        self.n_neighbors, self._X, self._y = n_neighbors, None, None

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y).reshape(-1)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[0] != y.size:
            raise ValueError("X phải là ma trận không rỗng và số hàng bằng độ dài y")
        if not np.isfinite(X).all() or self.n_neighbors > X.shape[0]:
            raise ValueError("Dữ liệu không hợp lệ hoặc k lớn hơn số mẫu")
        self._X, self._y = X.copy(), y.copy()
        return self

    def predict(self, X):
        if self._X is None:
            raise RuntimeError("Hãy gọi fit() trước predict()")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self._X.shape[1]:
            raise ValueError("X cần có cùng số đặc trưng như dữ liệu huấn luyện")
        result = []
        for point in X:
            idx = np.argsort(np.linalg.norm(self._X - point, axis=1))[:self.n_neighbors]
            labels, counts = np.unique(self._y[idx], return_counts=True)
            result.append(labels[np.argmax(counts)])
        return np.asarray(result)
