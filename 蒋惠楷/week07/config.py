# config.py

import torch
# 模型配置
FILE_PATH = "./data/文本分类练习.csv"
MODEL_NAME = 'XLNet'
MODEL_PATH = f"./models/{MODEL_NAME}_model.pth"
MAX_LENGTH = 256  # 最大输入长度
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
DROUPOUT = 0.2
EMBEDDING_DIM = 128
HIDDEN_DIM = 128

# cnn模型
NUM_FILTERS = 100
FILTER_SIZE = [3, 4, 5]

# 数据集配置
RANDOM_SEED = 42

# 设备配置
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
