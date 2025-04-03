
# 参数配置

Config = {
    "model_path":'./model_output',
    "bert_path": r'J:\\pretraining_model\\bert-base-chinese',
    "schema_path":'J:\\DataSet\\序列标注\\ner_data\\schema.json',
    "train_data":'J:\\DataSet\\序列标注\\ner_data\\train',
    "test_data":'J:\\DataSet\\序列标注\\ner_data\\test',
    "max_length":100,
    "batch_size":16,
    "class_num":9,
    'use_crf':True,
    'label_padding_token':-1,
    'learning_rate':1e-3,
    'seed':12,
    "epoch":5,

}