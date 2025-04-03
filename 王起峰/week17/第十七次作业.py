import numpy as np
from typing import Dict, List, Tuple


def build_rating_matrix(
        rating_file_path: str,
        item_file_path: str
) -> Tuple[Dict[int, np.ndarray], Dict[int, str]]:
    """构建用户-物品评分矩阵和物品名称映射

    Args:
        rating_file_path: 评分文件路径
        item_file_path: 物品信息文件路径

    Returns:
        user_ratings: 用户评分矩阵 {user_id: np.array}
        item_names: 物品名称映射 {item_id: item_name}
    """
    # 读取物品名称
    item_names = {}
    with open(item_file_path, encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split("|")
            item_names[int(parts[0])] = parts[1]

    # 初始化用户评分矩阵
    num_items = len(item_names)
    user_ratings = {}

    # 读取评分数据
    with open(rating_file_path) as f:
        for line in f:
            user_id, item_id, rating, _ = line.strip().split("\t")
            user_id = int(user_id)
            item_id = int(item_id)

            if user_id not in user_ratings:
                user_ratings[user_id] = np.zeros(num_items, dtype=np.float32)

            user_ratings[user_id][item_id - 1] = float(rating)

    return user_ratings, item_names


def compute_item_similarities(
        user_ratings: Dict[int, np.ndarray],
        min_common_users: int = 5
) -> Dict[int, List[Tuple[int, float]]]:
    """计算物品之间的余弦相似度

    Args:
        user_ratings: 用户评分矩阵
        min_common_users: 最小共同评分用户数

    Returns:
        物品相似度字典 {item_id: [(similar_item_id, similarity), ...]}
    """
    num_items = len(next(iter(user_ratings.values())))
    item_sims = {}
    sim_cache = {}

    # 将用户评分转换为物品视角
    item_ratings = np.array([
        [user_ratings[user_id][item_id] for user_id in user_ratings]
        for item_id in range(num_items)
    ])

    # 计算物品相似度
    for item_a in range(num_items):
        ratings_a = item_ratings[item_a]
        similarities = []

        for item_b in range(num_items):
            if item_b == item_a:
                continue

            cache_key = (min(item_a, item_b), max(item_a, item_b))
            if cache_key in sim_cache:
                similarity = sim_cache[cache_key]
            else:
                ratings_b = item_ratings[item_b]
                # 只计算有共同评分的用户
                common_ratings_mask = (ratings_a != 0) & (ratings_b != 0)
                if np.sum(common_ratings_mask) < min_common_users:
                    similarity = 0.0
                else:
                    vec_a = ratings_a[common_ratings_mask]
                    vec_b = ratings_b[common_ratings_mask]
                    similarity = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-9)
                sim_cache[cache_key] = similarity

            similarities.append((item_b, similarity))

        # 按相似度降序排序
        item_sims[item_a] = sorted(similarities, key=lambda x: x[1], reverse=True)

    return item_sims


def predict_rating(
        user_id: int,
        item_id: int,
        item_similarities: Dict[int, List[Tuple[int, float]]],
        user_ratings: Dict[int, np.ndarray],
        top_n: int = 10
) -> float:
    """预测用户对物品的评分

    Args:
        user_id: 用户ID
        item_id: 物品ID (1-based)
        item_similarities: 物品相似度字典
        user_ratings: 用户评分矩阵
        top_n: 使用最相似的top_n个物品

    Returns:
        预测评分
    """
    item_idx = item_id - 1  # 转换为0-based索引
    total = 0.0
    sim_sum = 0.0

    for similar_item, similarity in item_similarities[item_idx][:top_n]:
        user_rating = user_ratings[user_id][similar_item]
        if user_rating > 0 and similarity > 0:
            total += user_rating * similarity
            sim_sum += similarity

    return total / (sim_sum + 1e-9) if sim_sum > 0 else 0.0


def generate_recommendations(
        user_id: int,
        item_similarities: Dict[int, List[Tuple[int, float]]],
        user_ratings: Dict[int, np.ndarray],
        item_names: Dict[int, str],
        top_n: int = 10
) -> List[Tuple[str, float]]:
    """为用户生成推荐列表

    Args:
        user_id: 用户ID
        item_similarities: 物品相似度字典
        user_ratings: 用户评分矩阵
        item_names: 物品名称映射
        top_n: 推荐数量

    Returns:
        推荐列表 [(item_name, predicted_rating), ...]
    """
    # 找出用户未评分的物品
    user_rating_vec = user_ratings[user_id]
    unseen_items = [i + 1 for i, rating in enumerate(user_rating_vec) if rating == 0]

    # 预测评分
    recommendations = []
    for item_id in unseen_items:
        score = predict_rating(user_id, item_id, item_similarities, user_ratings)
        recommendations.append((item_names[item_id], score))

    # 返回top_n推荐
    return sorted(recommendations, key=lambda x: x[1], reverse=True)[:top_n]


def main():
    # 文件路径配置
    RATING_FILE = "D:/BaiduNetdiskDownload/python学习资料/nlp深度学习/第十七周 推荐系统/week17 推荐系统/ml-100k/u.data"
    ITEM_FILE = "D:/BaiduNetdiskDownload/python学习资料/nlp深度学习/第十七周 推荐系统/week17 推荐系统/ml-100k/u.item"

    print("正在加载数据...")
    user_ratings, item_names = build_rating_matrix(RATING_FILE, ITEM_FILE)

    print("正在计算物品相似度...")
    item_similarities = compute_item_similarities(user_ratings)

    print("推荐系统准备就绪，输入用户ID获取推荐(输入q退出)")
    while True:
        user_input = input("请输入用户ID: ").strip()
        if user_input.lower() == 'q':
            break

        try:
            user_id = int(user_input)
            if user_id not in user_ratings:
                print(f"错误: 用户ID {user_id} 不存在")
                continue

            recommendations = generate_recommendations(
                user_id, item_similarities, user_ratings, item_names
            )

            print(f"\n为用户 {user_id} 推荐的电影:")
            for movie, score in recommendations:
                print(f"{score:.4f}\t{movie}")
            print()

        except ValueError:
            print("错误: 请输入有效的用户ID或q退出")


if __name__ == "__main__":
    main()
