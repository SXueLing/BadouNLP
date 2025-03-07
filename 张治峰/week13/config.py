Config = {
    "model_save_path": "model_path",
    "pretrain_model_path" : "/Users/zhangzhifeng/workspace/PyCharm/model/bert-base-chinese",
    "num_labels" : 9,
    "tuning_tactics":"lora_tuning",
    "optimizer":"adam",
    "epoch": 20,
    "learning_rate": 1e-3,
    "train_data_path":"ner_data/train",
    "valid_data_path":"ner_data/test",
    "schema_path":"ner_data/schema.json",
    "max_length":100,
    "batch_size":8
}
