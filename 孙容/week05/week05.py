import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# 示例数据
np.random.seed(42)
data = np.random.rand(100, 2)  # 100个二维点

# 聚类数目
n_clusters = 3

# 使用KMeans聚类
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(data)

# 获取每个点的聚类标签和质心
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# 计算类内距离
intra_distances = []

for cluster_id in range(n_clusters):
    cluster_points = data[labels == cluster_id]  # 取出属于该聚类的点
    centroid = centroids[cluster_id]  # 该聚类的质心
    distances = cdist(cluster_points, [centroid])  # 计算每个点到质心的距离
    for point, distance in zip(cluster_points, distances):
        intra_distances.append((cluster_id, point, distance[0]))

# 按距离排序
intra_distances_sorted = sorted(intra_distances, key=lambda x: x[2])

# 输出排序结果
print("聚类内点距离排序结果：")
for cluster_id, point, distance in intra_distances_sorted:
    print(f"聚类: {cluster_id}, 点: {point}, 距离: {distance:.4f}")
