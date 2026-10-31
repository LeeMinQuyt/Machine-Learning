"""Gom điểm dữ liệu thành ba cụm."""
import numpy as np
from ml_algorithms_vn import KMeans

rng = np.random.default_rng(7)
X = np.vstack([rng.normal(loc=(-4,-3),scale=0.6,size=(30,2)),
               rng.normal(loc=(0,4),scale=0.6,size=(30,2)),
               rng.normal(loc=(4,-2),scale=0.6,size=(30,2))])
model = KMeans(n_clusters=3,random_state=7).fit(X)
print("Tâm cụm:\n",np.round(model.centroids,2))
print("Inertia:",round(model.inertia,2))
print("Nhãn cho [0,4]:",model.predict([[0,4]])[0])
