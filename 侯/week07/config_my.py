"""
@Project ：cgNLPproject 
@File    ：config_my.py
@Date    ：2025/1/6 15:21 
"""
Config = {
    'hidden_size':256,
    'input_dim':128,
    'model_type':'cnn',
    'num_layers':3,
    'pretrain_model_path':'E:\\SOME_DOC\\bd\\bert-base-chinese',
    'vocab_path':'chars.txt',
    'train_data_path':'train_tag_news.json',
    'valid_data_path':'valid_tag_news.json',
    'pooling_style':'avg',
    'kernel_size':3,
    'max_length':30,
    'batch_size':128,
    'seed':789,
    'optimizer': 'adam',
    'lr': 1e-3,
    'epoch_num':10
}