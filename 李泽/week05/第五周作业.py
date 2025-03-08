import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


def calculate_intra_cluster_distances(X, labels, centroids):
    """
    计算每个类内数据点到其所属类中心的平均距离。

    参数:
    X (numpy.ndarray): 特征数据矩阵，形状为 (n_samples, n_features)。
    labels (numpy.ndarray): 聚类标签数组，形状为 (n_samples,)，每个元素对应一个数据点所属的类标签。
    centroids (numpy.ndarray): 聚类中心坐标矩阵，形状为 (n_clusters, n_features)。

    返回:
    dict: 以类标签为键，类内平均距离为值的字典。
    """
    intra_cluster_distances = {}
    for label in np.unique(labels):
        class_mask = labels == label
        class_data = X[class_mask]
        centroid = centroids[label]
        distances = np.linalg.norm(class_data - centroid, axis=1)
        intra_cluster_distances[label] = np.mean(distances)
    return intra_cluster_distances


def sort_clusters_by_intra_cluster_distance(intra_cluster_distances):
    """
    根据类内平均距离对聚类进行排序。

    参数:
    intra_cluster_distances (dict): 以类标签为键，类内平均距离为值的字典。

    返回:
    list: 按照类内平均距离从小到大排序后的类标签列表。
    """
    return sorted(intra_cluster_distances, key=lambda x: intra_cluster_distances[x])


# 生成示例数据（这里使用make_blobs函数生成模拟数据，你可以替换成真实数据）
X, true_labels = make_blobs(n_samples=300, centers=4, random_state=42)

# 执行K-Means聚类（这里指定聚类数量为4，你可以根据实际情况调整）
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

# 获取聚类标签和聚类中心
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# 计算类内距离
intra_cluster_distances = calculate_intra_cluster_distances(X, labels, centroids)

# 基于类内距离对聚类进行排序
sorted_clusters = sort_clusters_by_intra_cluster_distance(intra_cluster_distances)

# 输出排序后的结果
print("按照类内平均距离从小到大排序后的聚类顺序:", sorted_clusters)

# 可视化（可选，只是为了直观展示聚类结果，这里简单绘制散点图）
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='black')
plt.title("K-Means Clustering Result")
plt.show()
