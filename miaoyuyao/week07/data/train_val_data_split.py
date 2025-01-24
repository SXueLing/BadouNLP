import pandas as pd
from sklearn.model_selection import train_test_split

from test_ai.homework.week07.config.logging_config import logger
from test_ai.homework.week07.config.running_config import Config


class TrainTestSplit:
    def __init__(self, config):
        self.config = config
        self.test_data_path = config["test_data_path"]

    def data_split(self):
        if self.test_data_path.lower().endswith(f".csv".lower()):
            df = pd.read_csv(self.test_data_path)

            x = df["review"]
            y = df["label"]
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

            train_data = pd.DataFrame({"label": y_train, "review": x_train})
            val_data = pd.DataFrame({"label": y_val, "review": x_val})

            train_data = train_data.reset_index(drop=True)
            val_data = val_data.reset_index(drop=True)

            train_data.to_csv(self.config["train_data_path"], index=False)
            val_data.to_csv(self.config["valid_data_path"], index=False)

            logger.info(f'训练集已保存至{self.config["train_data_path"]}')
            logger.info(f'验证集已保存至{self.config["valid_data_path"]}')

        else:
            logger.info("文件格式不支持")


if __name__ == "__main__":
    tmp = TrainTestSplit(Config)
    tmp.data_split()
