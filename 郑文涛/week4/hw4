import numpy as np
import sys
import random
class KMeansClusterer:
    def __init__(self, ndarray, cluster_num):
        self.ndarray = ndarray
        self.cluster_num = cluster_num
        self.points = self.__pick_start_point(ndarray, cluster_num)
    def cluster(self):
        result = [[] for _ in range(self.cluster_num)]
        for item in self.ndarray:
            distance_min = sys.maxsize
            index = -1
            for i in range(len(self.points)):
                distance = self.__distance(item, self.points[i])
                if distance < distance_min:
                    distance_min = distance
                    index = i
            result[index].append(item.tolist())
        new_centers = [self.__center(cluster) for cluster in result]
        # 如果中心点未改变（或者达到某个迭代次数/误差阈值），则结束递归
        if np.allclose(self.points, new_centers):
            self.cluster_distances = self.__calculate_cluster_distances(result, new_centers)
            return result, new_centers, self.cluster_distances
        self.points = np.array(new_centers)
        return self.cluster()
    def __calculate_cluster_distances(self, result, centers):
        # 计算每个聚类中点到聚类中心的距离，并返回每个聚类的平均距离
        cluster_distances = []
        for i, cluster in enumerate(result):
            cluster_distances.append(np.mean([self.__distance(point, centers[i]) for point in cluster]))
        return cluster_distances
    def __distance(self, p1, p2):
        # 计算两点之间的欧几里得距离
        return np.linalg.norm(np.array(p1) - np.array(p2))
    def __center(self, list_of_points):
        # 计算一组点的中心点（均值）
        return np.array(list_of_points).mean(axis=0)
    def __pick_start_point(self, ndarray, cluster_num):
        # 随机选择初始中心点
        if cluster_num < 0 or cluster_num > ndarray.shape[0]:
            raise Exception("簇数设置有误")
        indexes = random.sample(range(ndarray.shape[0]), cluster_num)
        return np.array([ndarray[i] for i in indexes])
# 使用示例
x = np.random.rand(100, 8)  # 生成100个8维的随机点
kmeans = KMeansClusterer(x, 3)  # 聚类成3个簇
result, cente
