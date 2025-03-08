
"""
配置参数
"""

Config = {
    "model_path": "output",
    "train_data_path": "./train.csv",
    "valid_data_path": "./test.csv",
    "vocab_path": "./chars.txt",
    "pretrain_model_path": r"D:\BaiduNetdiskDownload\八斗NLP课程\week6 语言模型\bert-base-chinese",
    "model_type": "bert",
    "pooling_style": "max",
    "optimizer": "adam",
    "hidden_size": 256,
    "kernel_size": 3,
    "epoch": 15,
    "max_length": 25,
    "batch_size": 128,
    "learning_rate": 1e-5,
    "num_layers": 2,
    "seed":987,
    "class_nums": 2
}
