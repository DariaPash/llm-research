from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Загрузка данных
iris = load_iris()
X = iris.data

# Стандартизация признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Параметры для подбора k
k_range = [2, 3, 4, 5, 6]
best_k = None
best_score = -1

# Подбор оптимального числа кластеров k
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)

    if score > best_score:
        best_score = score
        best_k = k

# Итоговая кластеризация с оптимальным k
final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
final_labels = final_kmeans.fit_predict(X_scaled)
final_silhouette = silhouette_score(X_scaled, final_labels)

print(f"METRIC: {final_silhouette}")
