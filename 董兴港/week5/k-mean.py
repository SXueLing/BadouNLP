import numpy as np
import random
import sys
'''
Kmeans算法实现
原文链接：https://blog.csdn.net/qingchedeyongqi/article/details/116806277
'''

class KMeansClusterer:  # k均值聚类
    def __init__(self, ndarray, cluster_num):
        self.ndarray = ndarray
        self.cluster_num = cluster_num
        self.points = self.__pick_start_point(ndarray, cluster_num)

    def cluster(self, sort_by_distance=False):
        result = []
        for i in range(self.cluster_num):
            result.append([])
        
        for item in self.ndarray:
            distance_min = sys.maxsize
            index = -1
            for i in range(len(self.points)):
                distance = self.__distance(item, self.points[i])
                if distance < distance_min:
                    distance_min = distance
                    index = i
            result[index] = result[index] + [item.tolist()]
        
        new_center = []
        for item in result:
            new_center.append(self.__center(item).tolist())
        
        # 中心点未改变，说明达到稳态，结束递归
        if (self.points == new_center).all():
            sum_dist = self.__sumdis(result)
            if sort_by_distance:
                sorted_clusters = self.sort_clusters_by_internal_distance(result)
                return sorted_clusters, self.points, sum_dist
            return result, self.points, sum_dist
        
        self.points = np.array(new_center)
        return self.cluster(sort_by_distance)

    def __sumdis(self,result):
        #计算总距离和
        sum=0
        for i in range(len(self.points)):
            for j in range(len(result[i])):
                sum+=self.__distance(result[i][j],self.points[i])
        return sum

    def __center(self, list):
        # 计算每一列的平均值
        return np.array(list).mean(axis=0)

    def __distance(self, p1, p2):
        #计算两点间距
        tmp = 0
        for i in range(len(p1)):
            tmp += pow(p1[i] - p2[i], 2)
        return pow(tmp, 0.5)

    def __pick_start_point(self, ndarray, cluster_num):
        if cluster_num < 0 or cluster_num > ndarray.shape[0]:
            raise Exception("簇数设置有误")
        # 取点的下标
        indexes = random.sample(np.arange(0, ndarray.shape[0], step=1).tolist(), cluster_num)
        points = []
        for index in indexes:
            points.append(ndarray[index].tolist())
        return np.array(points)

    def sort_clusters_by_internal_distance(self, clusters):
        """
        计算并排序每个聚类的内部距离
        返回: [(cluster_index, internal_distance, cluster_data)]
        """
        cluster_distances = []
        
        for i, cluster in enumerate(clusters):
            if not cluster:  # 跳过空聚类
                continue
            
            # 计算该聚类的中心点
            center = self.__center(cluster)
            
            # 计算该聚类内所有点到中心的平均距离
            total_distance = 0
            for point in cluster:
                total_distance += self.__distance(point, center)
            avg_distance = total_distance / len(cluster) if cluster else 0
            
            cluster_distances.append((i, avg_distance, cluster))
        
        # 按照内部距离排序
        sorted_clusters = sorted(cluster_distances, key=lambda x: x[1])
        
        return sorted_clusters

x = np.random.rand(100, 8)
kmeans = KMeansClusterer(x, 10)

# 使用排序功能
sorted_clusters, centers, distances = kmeans.cluster(sort_by_distance=True)

# 打印排序后的聚类结果
for i, (cluster_idx, internal_dist, cluster_data) in enumerate(sorted_clusters):
    print(f"Cluster {cluster_idx}: {len(cluster_data)} points, Internal distance: {internal_dist:.4f}")
