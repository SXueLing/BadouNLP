"""
配置参数信息
"""
Config = {
    "model_path": "./output/",
    "model_name": "model.pt",
    "schema_path": r"D:\NLP\video\第八周\week8 文本匹配问题\data\schema.json",
    "train_data_path": r"D:\NLP\video\第八周\week8 文本匹配问题\data\data.json",
    "valid_data_path": r"D:\NLP\video\第八周\week8 文本匹配问题\data\valid.json",
    "vocab_path": r"D:\NLP\video\第七周\data\vocab.txt",
    "positive_sample_rate": 0.5,
    "char_dim": 32,
    "max_len": 20,
    "hidden_size": 128,
    "epoch_size": 15,
    "batch_size": 32,
    "simple_size": 300,
    "lr": 1e-3,
    "optimizer": "adam",
}


