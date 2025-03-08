Config = {
    "model_path": "./model.bin",
    "train_data_path": "news.json",
    "data_num": 50000,
    "vocab_path": "vocab.txt",
    "model_type": "bert",
    "max_length_content": 260,
    "max_length_title": 30,
    "bridge_token": "以上新闻标题是：",
    "hidden_size": 256,
    "num_layers": 2,
    "epochs": 25,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "pretrain_model_path": "../bert-base-chinese/bert-base-chinese",
    "seed": 987,
    "sampling": 0.1
}
