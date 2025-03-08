Config = {
    "model_path": "./model/",
    "out_csv_path": "./result/练习结果.csv",
    "config_path": "config.py",
    "train_data_all_path": "./work/文本分类练习.csv",
    "train_data_path": "./work/文本分类练习-训练.csv",
    "valid_data_path": "./work/文本分类练习-测试.csv",
    "vocab_path": "./data/chars.txt",
    "model_type": "bert",
    "model_type_list": [
        "bert",
        "rnn",
        "cnn",
        "lstm",
        "gru",
        "fast_text",
        "text_rnn",
        "text_cnn",
        "gated_cnn",
        "text_rcnn",
        "stack_gated_cnn",
        "bert_lstm",
        "bert_rnn",
        "bert_cnn",
        "bert_gcnn",
        "bert_mid_layer"
    ],
    "droup_out_pro": 0.5,
    "vocab_size": 4622,
    "max_length": 75,
    "len_check": 0.95,
    "compare_pen": 0.8,
    "dic_length": 30,
    "class_num": 2,
    "hidden_size": 768,
    "hidden_size_list": [
        256
    ],
    "kernel_size": 3,
    "num_layers": 3,
    "epoch": 10,
    "epoch_list": [
        15,
        20
    ],
    "batch_size": 128,
    "batch_size_list": [
        256
    ],
    "pooling_style": "max",
    "pooling_style_list": [
        "max",
        "avg"
    ],
    "optimizer": "adam",
    "learning_rate": 0.0001,
    "learning_rate_list": [
        0.001,
        0.0001
    ],
    "pretrain_model_path": "C:\Users\zixu\Documents\深度学习\第七周 文本分类\week7 文本分类问题\work\bert-base-chinese",
    "seed": 987
}
