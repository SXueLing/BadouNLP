import torch
# config.py

# 文件路径
TRAIN_DATA_PATH = "./文本分类练习.csv"
VAL_DATA_PATH = "./文本分类练习.csv"

# 超参数
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
NUM_CLASSES = 2  # 根据数据集的类别数调整
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
MAX_LEN = 100  # 文本序列的最大长度
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
