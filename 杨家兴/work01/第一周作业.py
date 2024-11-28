import torch
b1 = torch.tensor([[1,2,3], [1,2,3]])
b2 = torch.tensor([[2,3,4],[2,3,4]])
print(b1 + b2)
print(2 * b1)
print(b1 * b2)
print(b1 / b2)
a5 = torch.exp(b1)
print(a5)
