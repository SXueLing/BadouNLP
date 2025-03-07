import torch

class Config:
    model_name = "bert-base-uncased"
    train_file = "data/train.json"
    val_file = "data/val.json"
    test_file = "data/test.json"
    max_length = 128
    batch_size = 16
    learning_rate = 2e-5
    num_epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lora_r = 8  # LoRA 低秩矩阵的秩
    lora_alpha = 16  # LoRA alpha 参数
    lora_dropout = 0.1  # LoRA dropout
