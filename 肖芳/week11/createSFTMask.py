import numpy as np
import torch

def createSFTMask(s_len, index):
  arr = torch.ones((s_len, s_len))

  arr[:index, index:] = 0.0
  i = index + 1
  for row in arr[index:]:
    row[i:] = 0.0
    i += 1
  return arr

def createSFTMaskForY(ys):

  tensor = torch.empty(ys.shape[0], ys.shape[1], ys.shape[1])
  for (i,y) in enumerate(ys):
    index = (y == -100).sum() + 1
    tensor[i] = createSFTMask(len(y), index)
  
  return tensor


# print(createSFTMask(10, 5))
# 打印结果
# tensor([[1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
#         [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
#         [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
#         [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
#         [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
#         [1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
#         [1., 1., 1., 1., 1., 1., 1., 0., 0., 0.],
#         [1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
#         [1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
#         [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])


