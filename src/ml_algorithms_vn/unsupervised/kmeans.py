"""K-Means với khởi tạo k-means++ và thuật toán Lloyd."""
import numpy as np


class KMeans:
    """Chia dữ liệu thành k cụm; nhãn cụm không mang ý nghĩa thứ tự."""
    def __init__(self, n_clusters=3, max_iterations=300, random_state=42):
        if n_clusters < 1 or max_iterations < 1:
            raise ValueError("n_clusters và max_iterations phải >= 1")
        self.n_clusters, self.max_iterations, self.random_state = n_clusters, max_iterations, random_state
        self.centroids, self.inertia = None, 0.0

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not X.size or not np.isfinite(X).all() or self.n_clusters > X.shape[0]:
            raise ValueError("X cần là ma trận hữu hạn, không rỗng và có ít nhất k mẫu")
        rng = np.random.default_rng(self.random_state)
        centers = [X[rng.integers(len(X))]]
        while len(centers) < self.n_clusters:
            d2 = np.min(np.stack([np.sum((X-c)**2, axis=1) for c in centers]), axis=0)
            if d2.sum() == 0:
                remaining = [r for r in X if not any(np.array_equal(r,c) for c in centers)]
                centers.append(remaining[0] if remaining else X[rng.integers(len(X))])
            else:
                centers.append(X[rng.choice(len(X), p=d2/d2.sum())])
        centers = np.asarray(centers, dtype=float)
        for _ in range(self.max_iterations):
            d2 = np.sum((X[:,None,:]-centers[None,:,:])**2, axis=2)
            labels = np.argmin(d2, axis=1)
            updated = centers.copy()
            for k in range(self.n_clusters):
                members = X[labels == k]
                if len(members):
                    updated[k] = members.mean(axis=0)
                else:
                    updated[k] = X[np.argmax(np.min(d2, axis=1))]
            if np.allclose(centers, updated):
                centers = updated
                break
            centers = updated
        self.centroids = centers
        labels = self.predict(X)
        self.inertia = float(np.sum((X-centers[labels])**2))
        return self

    def predict(self, X):
        if self.centroids is None:
            raise RuntimeError("Hãy gọi fit() trước predict()")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.centroids.shape[1]:
            raise ValueError("X cần có cùng số đặc trưng như dữ liệu huấn luyện")
        d2 = np.sum((X[:,None,:]-self.centroids[None,:,:])**2, axis=2)
        return np.argmin(d2, axis=1)
