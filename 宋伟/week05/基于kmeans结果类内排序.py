# -*- coding: utf-8 -*-
# @Date    :2024-12-26 11:22:04
# @Author  :SongWei (songweiof@gmail.com)
# @Link    :https://github.com/Finder12345
# @Version :python 3.11
# @Software:Sublime Text


import numpy as np
from sklearn.cluster import KMeans

# 示例数据
np.random.seed(16)
data = np.random.rand(100, 2)  # 100个点，2个特征

# 1. 使用KMeans进行聚类
kmeans = KMeans(n_clusters=5, random_state=16)  # 假设分成5类
kmeans.fit(data)

# 聚类结果
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# 2. 计算每个类的类内距离
class_distances = {}
for i in range(len(centers)):
    # 选取属于类i的点
    class_points = data[labels == i]
    # 计算每个点到类中心的距离
    distances = np.linalg.norm(class_points - centers[i], axis=1)
    # 累计类内距离
    class_distances[i] = np.sum(distances)
    # 类内拼接距离
    # class_distances[i] = np.sum(distances)/len(class_points)

# 3. 对类内距离进行排序
sorted_distances = sorted(class_distances.items(), key=lambda x: x[1])

# 输出排序结果
print("类内距离排序结果（类编号, 距离）：")
for cluster_id, distance in sorted_distances:
    print(f"类 {cluster_id}: {distance:.4f}")

"""
类内距离排序结果（类编号, 距离）：
类 3: 2.2631
类 1: 2.8020
类 0: 2.8853
类 4: 3.7017
类 2: 4.3656
"""