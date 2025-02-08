import numpy as np
import random
import sys

class KMeansClusterer:
    def __init__(self, ndarray, cluster_num):
        self.ndarray = ndarray
        self.cluster_num = cluster_num
        self.points = self.__pick_start_point(ndarray, cluster_num)

    def cluster(self):
        result = []
        for i in range(self.cluster_num):
            result.append([])  # 初始化簇列表

        # 分配每个数据点到最近的簇
        for item in self.ndarray:
            distance_min = sys.maxsize
            index = -1
            for i in range(len(self.points)):
                distance = self.__distance(item, self.points[i])
                if distance < distance_min:
                    distance_min = distance
                    index = i
            result[index] = result[index] + [item.tolist()]

        # 更新每个簇的聚类中心
        new_center = []
        for item in result:
            new_center.append(self.__center(item).tolist())

        # 如果聚类中心没有变化，表示已收敛
        if (self.points == new_center).all():
            sum = self.__sumdis(result)
            sorted_result = self.__get_sorted_distances(result)  # 对每个簇内的点按距离排序
            return result, self.points, sum, sorted_result

        # 否则更新聚类中心并递归
        self.points = np.array(new_center)
        return self.cluster()

    def __sumdis(self, result):
        sum = 0
        for i in range(len(self.points)):
            for j in range(len(result[i])):
                sum += self.__distance(result[i][j], self.points[i])
        return sum

    def __center(self, list):
        return np.array(list).mean(axis=0)

    def __distance(self, p1, p2):
        tmp = 0
        for i in range(len(p1)):
            tmp += pow(p1[i] - p2[i], 2)
        return pow(tmp, 0.5)

    def __pick_start_point(self, ndarray, cluster_num):
        if cluster_num < 0 or cluster_num > ndarray.shape[0]:
            raise Exception("簇数设置有误")
        indexes = random.sample(np.arange(0, ndarray.shape[0], step=1).tolist(), cluster_num)
        points = []
        for index in indexes:
            points.append(ndarray[index].tolist())
        return np.array(points)

    # 新增：对每个簇内的点按到聚类中心的距离排序
    def __get_sorted_distances(self, result):
        sorted_result = []
        for i in range(len(result)):
            # 计算每个簇内的数据点到中心的距离
            distances = []
            for item in result[i]:
                distance = self.__distance(item, self.points[i])  # 计算当前点到该簇中心的距离
                distances.append((item, distance))

            # 按照距离进行升序排序
            distances.sort(key=lambda x: x[1])  # 排序：按距离升序排列
            sorted_result.append([x[0] for x in distances])  # 只保留排序后的点
        return sorted_result


# 测试代码
x = np.random.rand(100, 8)  # 生成一个 100 个点，每个点有 8 个维度的随机数据
kmeans = KMeansClusterer(x, 10)  # 创建 KMeansClusterer 实例，指定簇数为 10
result, centers, distances, sorted_result = kmeans.cluster()  # 执行聚类并排序

print("聚类结果（每个簇中的数据点）:")
print(result)

print("最终聚类中心:")
print(centers)

print("总距离和:")
print(distances)

print("排序后的类内距离:")
print(sorted_result)
