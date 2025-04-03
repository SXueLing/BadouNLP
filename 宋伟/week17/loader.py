import numpy as np
import pandas as pd
from collections import defaultdict
from config import Config

class DataLoader:
    def __init__(self):
        self.data_path = Config.DATA_PATH
        self.top_k = Config.TOP_K

    def load_data(self):
        df = pd.read_csv(self.data_path)
        df = df[df['rating'] >= Config.MIN_RATING]  # 过滤低评分
        return df

    def compute_item_similarity(self, df):
        user_items = defaultdict(set)
        item_users = defaultdict(set)

        for _, row in df.iterrows():
            user, item = row['userId'], row['movieId']
            user_items[user].add(item)
            item_users[item].add(user)

        item_sim = defaultdict(lambda: defaultdict(float))

        # 计算物品相似度
        for item1, users1 in item_users.items():
            for item2, users2 in item_users.items():
                if item1 == item2:
                    continue
                intersection = len(users1 & users2)
                if intersection == 0:
                    continue
                sim = intersection / np.sqrt(len(users1) * len(users2))
                item_sim[item1][item2] = sim

        # 只保留Top-K个相似物品
        for item in item_sim:
            sorted_items = sorted(item_sim[item].items(), key=lambda x: x[1], reverse=True)[:self.top_k]
            item_sim[item] = dict(sorted_items)

        np.save(Config.SIMILARITY_PATH, item_sim)  # 保存相似度矩阵
        return item_sim
