Config = {
    "model_path": "./model/",
    "out_csv_path": "./result/",
    "config_path": "config.py",
    "train_data_path": "./data/train",
    "valid_data_path": "./data/test",
    "schema_path": "./data/schema.json",
    "vocab_path": "./data/chars.txt",
    "model_type": "bert",
    "task_type": "train",
    "model_type_list": [
        "bert"
    ],
    "droup_out_pro": 0.5,
    "vocab_size": 4622,
    "max_length": 110,
    "len_check": 0.95,
    "compare_pen": 0.8,
    "use_crf": True,
    "dic_length": 30,
    "class_num": 9,
    "hidden_size": 768,
    "hidden_size_list": [
        256
    ],
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 20,
    "epoch_list": [
        20
    ],
    "batch_size": 128,
    "batch_size_list": [
        128
    ],
    "pooling_style": "max",
    "pooling_style_list": [
        "max"
    ],
    "optimizer": "adam",
    "learning_rate": 0.0001,
    "learning_rate_list": [
        0.0001,
        0.0005
    ],
    "pretrain_model_path": "D:\\JianGuo\\AI\\八斗\\课件资料\\第六周 语言模型\\bert-base-chinese",
    "seed": 5663
}
