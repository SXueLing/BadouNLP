import numpy as np
import random
import sys
from functools import partial
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
            # print("[item.tolist()]:",[item.tolist()])
        # print("result[index]:", result[index])
        new_center = []
        for item in result:
            new_center.append(self.__center(item).tolist())
        # 中心点未改变，说明达到稳态，结束递归
        if (self.points == new_center).all():
            sum = self.__sumdis(result)
            return result, self.points, sum
        self.points = np.array(new_center)
        return self.cluster()

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
        #print("bbb:",np.array(points))
        return np.array(points)
    
    def sort(self,array,points):
        print("shape.array:",len(array))
        sort_result = []
        for i in range(len(points)):
            r1 = []
            print("第",i,"个p1:",points[i])
            num = 1
            for p2 in array[i]:
                print("p2-",num,":",p2)
                dis = self.__distance(points[i],p2)
                r1.append(dis)
                num +=1
            # print("len:",len(r1))
            sort_result.append(r1)
        print("--------------------------------------")
        return sort_result

def o_distance(p1, p2):
        #计算两点间距
        tmp = 0
        for i in range(len(p1)):
            tmp += pow(p1[i] - p2[i], 2)
        return pow(tmp, 0.5)


if __name__ == '__main__':
    x = np.random.rand(100, 8)
    kmeans = KMeansClusterer(x, 10)
    result, centers, distances = kmeans.cluster()

    # print("--------before sort--------")
    # for i in range(len(result)):
    #     print("第",i,"个,num:",len(result[i]),"  ",result[i])

    # print("--------after sort--------")

    for i in range(len(result)):
        result[i].sort(key=partial(o_distance,centers[i]))

    # for i in range(len(result)):
    #     print("第",i,"个,num:",len(result[i]),"  ",result[i])

    
