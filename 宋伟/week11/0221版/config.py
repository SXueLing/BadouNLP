import os

class Config:
    # 训练参数
    epochs = 3
    batch_size = 16
    learning_rate = 2e-5
    max_length = 128
    warmup_steps = 0
    weight_decay = 0.01
    gradient_accumulation_steps = 1
    
    # BERT预训练模型路径
    model_name = "bert-base-uncased"  
    pretrained_model_path = os.path.join("bert-base-uncased")

    # 数据集路径
    train_data_path = "path_to_train_data.csv"
    val_data_path = "path_to_val_data.csv"
    test_data_path = "path_to_test_data.csv"

    # 输出路径
    output_dir = "./output"
    logging_dir = "./logs"
