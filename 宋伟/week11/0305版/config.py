# -*- coding: utf-8 -*-
# @Date    :2025-03-05 15:17:24
# @Author  :SongWei (songweiof@gmail.com)
# @Link    :https://github.com/Finder12345
# @Version :python 3.11
# @Software:Sublime Text




Config = {
    "model_path":'./output',
    'bert_path':r"J:\\pretraining_model\\bert-base-chinese",
    'corpus_path':'./sample_data.json',
    'max_length' : 50,
    'batch_size': 32,
    'optimizer' : 'adam',
    'learning_rate' :1e-3,
    'seed':12,
    'epoch_num':20,
}