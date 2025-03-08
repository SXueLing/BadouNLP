# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd

from test_ai.homework.week07.config.logging_config import logger
from test_ai.homework.week07.config.running_config import Config


class GetSequenceLengths:
    def __init__(self, data_path):
        self.data_path = data_path
        self.get_config_max_length = None

    def get_length(self):
        if self.data_path.lower().endswith(f".csv".lower()):
            df = pd.read_csv(self.data_path)
            sequences = df['review'].tolist()
            sequence_lengths = [len(sequence) for sequence in sequences]
            self.get_config_max_length = int(np.percentile(sequence_lengths, 95))
            logger.info(f"{self.data_path} 95%分位数约：{self.get_config_max_length}")  # 95%序列的长度小于该值
        else:
            logger.info("文件格式不支持")
        return self.get_config_max_length


if __name__ == "__main__":
    tmp = GetSequenceLengths(Config["train_data_path"])
    tmp.get_length()
