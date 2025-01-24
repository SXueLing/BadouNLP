"""
改用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类

"""


import  numpy as np
import  torch
import  torch.nn as nn

def softmax(matrix):
    return np.exp(matrix)/np.sum(np.exp(matrix))

numbers=np.random.random(5)
print("numbers.type:",type(numbers))

def to_one_hot(number):
    y_class=np.zeros(number.shape)
    y_class[np.argmax(number)]=1
    return y_class

def crossEntropy(target):
    pred=softmax(target)
    print("pred:",pred)
    y_target=to_one_hot(target)
    print("y_target:",y_target)

    return  -np.sum(np.log(pred)*y_target)

# print(softmax(numbers))
print("crossEntropy:",crossEntropy(numbers))


