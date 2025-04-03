class Config:
    DATA_PATH = "data/ratings.csv"
    SIMILARITY_PATH = "data/item_similarity.npy"
    TOP_K = 10  # 计算相似物品时的Top-K
    N_RECOMMEND = 5  # 推荐的物品数量
    MIN_RATING = 3  # 过滤低评分数据
