# -*- coding: utf-8 -*-

Config = {
    'bert_path' : 'bert-base-chinese',
    'corpus_path' : 'data/corpus.txt',
    'window_size' : 10,
    'input_max_length' : 100,
    'output_max_length' : 30,
    'epoch' : 20,
    'batch_times' : 1000,
    'batch_size' : 16,
    'train_sample' : 50000,
    "optimizer" : "adam",
    "learning_rate" : 1e-3,
    'model_path' : 'saved_model'
}
