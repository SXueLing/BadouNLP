class Config:
    # 基本参数
    model_name = "bert-base-uncased"  # 使用的预训练模型
    batch_size = 16
    max_len = 128
    num_labels = 9  # 对应NER任务的标签种类数（比如B-LOC，I-LOC，O等）

    # 训练参数
    learning_rate = 2e-5
    epochs = 3
    warmup_steps = 0
    weight_decay = 0.01
    device = "cuda" if torch.cuda.is_available() else "cpu"
