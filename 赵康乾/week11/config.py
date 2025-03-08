# -*- coding: utf-8 -*-

Config = {
    'bert_path' : 'bert-base-chinese',
    'corpus_path' : 'data/sample_data.json',
    'window_size' : 10,
    'input_max_length' : 256,
    'output_max_length' : 30,
    'epoch' : 200,
    'batch_times' : 1000,
    'batch_size' : 16,
    'train_sample' : 50000,
    "optimizer" : "adam",
    "learning_rate" : 1e-4,
    'model_path' : 'saved_model'
}
