Config = {
    "model_path": "model_output",
    "schema_path": "../data/schema.json",
    "train_data_path": "../data/train.json",
    "volid_data_path": "../data/valid.json",
    "vocab_path": "../chars.txt",
    "max_length": 20,
    "hidden_size": 128,
    "epoch": 10,
    "batch_size": 32,
    "epoch_data_size": 200,  # 数据量不够大，，从训练集中取出200条数据
    "positive_sample_rate": 0.5,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "loss":"cos_loss" ,# triplet
    "seed":44
}
