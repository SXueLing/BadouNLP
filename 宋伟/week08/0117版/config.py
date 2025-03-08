# 项目目录结构
# text_matching_project
# ├── config.py
# ├── evaluate.py
# ├── loader.py
# ├── main.py
# ├── model.py
# ├── requirements.txt
# ├── train.json
# └── README.md

# config.py
# 配置文件，保存模型、训练参数和数据路径。

CONFIG = {
    "model_name": "bert-base-chinese",
    "embedding_dim": 128,
    "learning_rate": 2e-5,
    "batch_size": 32,
    "num_epochs": 5,
    "margin": 1.0,
    "data_path": "train.json",
    "save_model_path": "best_model.pth"
}

