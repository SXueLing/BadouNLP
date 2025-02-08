"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "week8\data\schema.json",
    "train_data_path": "week8\data\\train.json",
    "valid_data_path": "week8\data\\valid.json",
    "vocab_path":"week8\chars.txt",
    "max_length": 20,
    "hidden_size": 128,
    "epoch": 20,
    "batch_size": 32,
    "epoch_data_size": 300,     
    "positive_sample_rate":0.5,  
    "optimizer": "adam",
    "learning_rate": 1e-3,
}
