# -*- coding: utf-8 -*-
# @Date    :2025-02-11 22:12:08
# @Author  :SongWei (songweiof@gmail.com)
# @Link    :https://github.com/Finder12345
# @Version :python 3.11
# @Software:Sublime Text


Config = {
    "model_path":"output",
    "data_path":"../文本分类练习.csv",

    "vocab_path":"../chars.txt",
    "model_type":"lstm",
    "max_length":30,
    "hidden_size":256,
    "kernel_size":3,
    "num_layers":2,
    "epoch":10,
    "batch_size":128,
    "pooling_style":"max",
    "optimizer":"adam",
    "learning_rate":1e-3,
    "pretrain_model_path":r"J:\\八斗课堂学习\\第六周 语言模型\\bert-base-chinese\\bert-base-chinese",
    "seed":987,
    "train_rate":0.8,
}