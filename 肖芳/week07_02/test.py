import csv
from config import Config
from torch.utils.data import Dataset, DataLoader
from loader import DataGenerator

# dg = DataGenerator("data/test_datas.csv", Config)
# print(dg[1])

# X = []
# Y = []

# def read_csv_datas(path):
#   with open(path, 'r') as f:
#       csv_reader = csv.reader(f)
#       next(csv_reader)
#       for row in csv_reader:
#           X.append(row[1])
#           Y.append(row[0])
#   return X, Y

# X, Y = read_csv_datas("data/test_datas.csv")
# print(X)
# print(Y)

print(1e-5)