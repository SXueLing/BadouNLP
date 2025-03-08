

Config = {
    "model_path": "./model.bin",
    "train_data_path": "./dataset/train.csv",
    "eval_data_path": "./dataset/eval.csv",
    "vocab_path": "chars.txt",
    "model_type": "bert",
    "max_length": 100,
    "hidden_size": 8,
    "kernel_size": 3,
    "num_layers": 1,
    "epochs": 15,
    "batch_size": 128,
    "pooling_style": "max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path": "./bert-base-chinese/bert-base-chinese",
    "seed": 987,
    "train_index": 0.9,
    "predict_num": 2
}
