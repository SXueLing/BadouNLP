# config.py
# -*- coding: utf-8 -*-

"""
配置参数信息
"""

from transformers import PretrainedConfig

class Config(PretrainedConfig):
    def __init__(
        self,
        model_path="output",
        train_data_path="../ner_data/train",
        valid_data_path="../ner_data/test",
        schema_path="../ner_data/schema.json",
        vocab_path="chars.txt",
        model_type="bert",
        max_length=100,
        class_num=9,
        use_crf=False,
        hidden_size=256,
        kernel_size=3,
        num_layers=2,
        epoch=10,
        batch_size=64,
        tuning_tactics="lora_tuning",
        pooling_style="max",
        optimizer="adam",
        learning_rate=1e-5,
        pretrain_model_path=r"D:\AI\machine-learning\nlp20\week06\bert-base-chinese",
        seed=987,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.train_data_path = train_data_path
        self.valid_data_path = valid_data_path
        self.schema_path = schema_path
        self.vocab_path = vocab_path
        self.model_type = model_type
        self.max_length = max_length
        self.class_num = class_num
        self.use_crf = use_crf
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.epoch = epoch
        self.batch_size = batch_size
        self.tuning_tactics = tuning_tactics
        self.pooling_style = pooling_style
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.pretrain_model_path = pretrain_model_path
        self.seed = seed
