"""
@Project ：cgNLPproject 
@File    ：config_08.py
@Date    ：2025/1/13 13:36 
"""
Config = {
            'train_data_path':'./data/train.json',
            'valid_data_path':'./data/valid.json',
            'vocab_type':'char',
            'word_vocab_path':'./words.txt',
            'char_vocab_path':'./chars.txt',
            'label_path':'./data/schema.json',
            'optimizer':'adam',
            'max_length':20,
            'batch_size':32,
            'epoch_size':200,
            'epoch_num':15,
            'hidden_size':128,
            "learning_rate": 1e-3,
            "margin": 2,
            'data_rate':0.5
}