Config = {
    "model_path": "model_output",  # 记录训练模型
    "schema_path": "./ner_data/schema.json",  # 标签进行Index化
    "train_data_path": "./ner_data/train",
    "valid_data_path":"./ner_data/test",
    "vocab_path":"chars.txt",   # 字表或词表
    "max_length":100,
    "hidden_size":256,
    "num_layers":2,
    "epoch":20,
    "batch_size":16,
    "optimizer":"adam",
    "learning_rate":1e-3,
    "use_crf":False,
    "class_num":9,
    "use_bert":True,
    "bert_path":r"J:\\八斗课堂学习\\第六周 语言模型\\bert-base-chinese\\bert-base-chinese",
    "seed":42

}
