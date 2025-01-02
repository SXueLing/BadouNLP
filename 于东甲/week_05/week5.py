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

    def cluster(self):
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
            sum,max,min = self.__sumdis(result)
            return result, self.points, sum,max,min
        self.points = np.array(new_center)
        return self.cluster()

    def __sumdis(self,result):
        #计算总距离和
        sum=[]
        max=[]
        min=[]
        for i in range(len(self.points)):
            sum_1=0
            min_1=sys.maxsize
            max_1=0
            for j in range(len(result[i])):
                distance=self.__distance(self.points[i], result[i][j])
                max_1=distance if distance > max_1 else max_1
                min_1=distance if distance < min_1 else min_1
                sum_1+=distance
            max.append(max_1)
            min.append(min_1)
            sum.append(sum_1)
        return sum,max,min

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
        #print("indexes" ,indexes)
        points = []
        for index in indexes:
            points.append(ndarray[index].tolist())
        return np.array(points)

x = np.random.rand(100, 8)

kmeans = KMeansClusterer(x, 10)
result, centers, distances,max,min = kmeans.cluster()
print(result)
print(len(result))
print(centers)
print(distances)
print(max)
print(min)
