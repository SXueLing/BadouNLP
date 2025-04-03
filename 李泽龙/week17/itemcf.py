import numpy as np

# 构建用户-物品评分矩阵和物品名称映射
def build_u2i_matrix(user_item_score_data_path, item_name_data_path, is_small=False):
    # 读取物品名称
    item_to_name = {}
    with open(item_name_data_path, encoding='latin-1') as f:
        for line in f:
            line = line.strip().split("|")
            item_to_name[int(line[0])] = line[1]

    # 读取用户-物品评分数据
    user_to_rating = {}
    with open(user_item_score_data_path) as f:
        for line in f:
            line = line.strip().split("\t")
            user_id = int(line[0])
            item_id = int(line[1])
            score = int(line[2])
            if user_id not in user_to_rating:
                user_to_rating[user_id] = [0] * len(item_to_name)
            user_to_rating[user_id][item_id - 1] = score

    return user_to_rating, item_to_name

# 计算余弦距离
def cosine_distance(x, y):
    dot_product = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    if norm_x == 0 or norm_y == 0:
        return 0
    return dot_product / (norm_x * norm_y)

# 找出相似的物品
def find_similar_item(user_to_rating):
    num_items = len(user_to_rating[1])
    item_to_similar_item = {}
    score_buffer = {}
    for item_a in range(num_items):
        similar_item = []
        for item_b in range(num_items):
            if item_b == item_a:
                continue
            # ab物品互换不用重新计算cos
            if "%d_%d" % (item_b, item_a) in score_buffer:
                similarity = score_buffer["%d_%d" % (item_b, item_a)]
            else:
                ratings_a = [user_to_rating[user_id][item_a] for user_id in user_to_rating]
                ratings_b = [user_to_rating[user_id][item_b] for user_id in user_to_rating]
                similarity = cosine_distance(np.array(ratings_a), np.array(ratings_b))
                score_buffer["%d_%d" % (item_a, item_b)] = similarity
            similar_item.append([item_b, similarity])
        similar_item = sorted(similar_item, reverse=True, key=lambda x: x[1])
        item_to_similar_item[item_a] = similar_item
    return item_to_similar_item

# 基于物品的协同过滤
def item_cf(user_id, item_id, item_to_similar_item, user_to_rating, topn=10):
    pred_score = 0
    count = 0
    for similar_item, similarity in item_to_similar_item[item_id - 1][:topn]:
        # 用户对相似物品的打分
        rating_by_user = user_to_rating[user_id][similar_item]
        # 分数*物品相似度，作为一种对分数的加权，越相似的物品评分越重要
        pred_score += rating_by_user * similarity
        # 如果用户没对这个相似物品打分，就不计算在总数内
        if rating_by_user != 0:
            count += 1
    pred_score /= count + 1e-5
    return pred_score

# 对于一个用户做完整的物品召回
def movie_recommand(user_id, item_to_similar_item, user_to_rating, item_to_name, topn=10):
    # 当前用户还没看过的所有电影id
    unseen_items = [item_id + 1 for item_id, rating in enumerate(user_to_rating[user_id]) if rating == 0]
    res = []
    for item_id in unseen_items:
        # item_cf打分
        score = item_cf(user_id, item_id, item_to_similar_item, user_to_rating)
        res.append([item_to_name[item_id], score])
    # 排序输出
    res = sorted(res, key=lambda x: x[1], reverse=True)
    return res[:topn]

if __name__ == "__main__":
    user_item_score_data_path = "E:/nlp_learn/第十七周 推荐系统/week17 推荐系统/ml-100k/u.data"
    item_name_data_path = "E:/nlp_learn/第十七周 推荐系统/week17 推荐系统/ml-100k/u.item"
    user_to_rating, item_to_name = build_u2i_matrix(user_item_score_data_path, item_name_data_path, False)

    # item-cf
    item_to_similar_item = find_similar_item(user_to_rating)

    # 为用户推荐电影
    while True:
        user_id = int(input("输入用户id："))
        recommands = movie_recommand(user_id, item_to_similar_item, user_to_rating, item_to_name)
        for recommand, score in recommands:
            print("%.4f\t%s" % (score, recommand))
    