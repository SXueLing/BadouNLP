import numpy as np
from loader import DataLoader
from model import ItemCF
from config import Config

def main():
    # 加载数据
    loader = DataLoader()
    df = loader.load_data()

    # 计算并加载相似度矩阵
    try:
        similarity_matrix = np.load(Config.SIMILARITY_PATH, allow_pickle=True).item()
    except FileNotFoundError:
        similarity_matrix = loader.compute_item_similarity(df)

    # 构建ItemCF模型
    model = ItemCF(similarity_matrix)

    # 获取测试用户的推荐
    user_id = 1  # 示例用户
    user_items = set(df[df["userId"] == user_id]["movieId"])
    recommendations = model.recommend(user_id, user_items)

    print(f"User {user_id} 推荐的电影ID: {recommendations}")

if __name__ == "__main__":
    main()
