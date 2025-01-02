import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 生成一个示例数据集
X, y = make_blobs(n_samples=300, centers=4, random_state=42)

# 进行KMeans聚类
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

# 获取每个数据点所属的聚类标签和质心
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# 计算每个类内样本到质心的平均距离
def compute_within_cluster_distance(X, labels, centroids):
    distances = []
    for i in range(len(centroids)):
        cluster_points = X[labels == i]
        centroid = centroids[i]
        # 计算该类所有点到质心的距离
        cluster_distances = np.linalg.norm(cluster_points - centroid, axis=1)
        mean_distance = np.mean(cluster_distances)
        distances.append((i, mean_distance))
    return distances

# 获取每个类的平均类内距离
distances = compute_within_cluster_distance(X, labels, centroids)

# 按类内距离排序
distances.sort(key=lambda x: x[1])

# 输出排序后的聚类结果
print("Classes sorted by mean within-cluster distance:")
for cluster, mean_distance in distances:
    print(f"Cluster {cluster}: Mean Distance = {mean_distance:.2f}")

# 可视化聚类结果和质心
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', label='Centroids')
plt.title('KMeans Clustering with Centroids')
plt.legend()
plt.show()
