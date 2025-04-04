import openpyxl
import numpy as np

'''
电影打分数据集
实现协同过滤
'''


# 为了好理解，将数据格式转化成user-item的打分矩阵形式
def build_u2i_matrix(user_item_score_data_path, item_name_data_path, write_file=False):
    # 获取item id到电影名的对应关系
    item_id_to_item_name = {}
    with open(item_name_data_path, encoding="ISO-8859-1") as f:
        for line in f:
            item_id, item_name = line.split("|")[:2]
            item_id = int(item_id)
            item_id_to_item_name[item_id] = item_name
    total_movie_count = len(item_id_to_item_name)
    print("total movie:", total_movie_count)

    # 读打分文件
    user_to_rating = {}
    with open(user_item_score_data_path, encoding="ISO-8859-1") as f:
        for line in f:
            user_id, item_id, score, time_stamp = line.split("\t")
            user_id, item_id, score = int(user_id), int(item_id), int(score)
            if user_id not in user_to_rating:
                user_to_rating[user_id] = [0] * total_movie_count
            user_to_rating[user_id][item_id - 1] = score
    print("total user:", len(user_to_rating))

    if not write_file:
        return user_to_rating, item_id_to_item_name

    # 写入excel便于查看
    workbook = openpyxl.Workbook()
    sheet = workbook.create_sheet(index=0)
    # 第一行：user_id, movie1, movie2...
    header = ["user_id"] + [item_id_to_item_name[i + 1] for i in range(total_movie_count)]
    sheet.append(header)
    for i in range(len(user_to_rating)):
        # 每行：user_id, rate1, rate2...
        line = [i + 1] + user_to_rating[i + 1]
        sheet.append(line)
    workbook.save("user_movie_rating.xlsx")
    return user_to_rating, item_id_to_item_name


# 向量余弦距离
def cosine_distance(vector1, vector2):
    ab = vector1.dot(vector2)
    a_norm = np.sqrt(np.sum(np.square(vector1)))
    b_norm = np.sqrt(np.sum(np.square(vector2)))
    return ab / (a_norm * b_norm)


def build_i2u_matrix(user_to_rating):
    item_to_vector = {}
    total_user = len(user_to_rating)
    for user, user_rating in user_to_rating.items():
        for movie_id, score in enumerate(user_rating):
            movie_id += 1
            if movie_id not in item_to_vector:
                item_to_vector[movie_id] = [0] * (total_user)
            item_to_vector[movie_id][user - 1] = score
    return item_to_vector


# topn为考虑多少相似的用户
# 取前topn相似用户对该电影的打分
# 对于一个用户做完整的item召回
# user_to_rating{id:[0,0,0]}
# item_to_rating{item:[0,0,0]}
# similar_items{item:[similar_item, score]}
# item_to_name{item:item_name}
def movie_recommand(user_id, user_to_rating, item_to_rating, similar_items, item_to_name, topn=10):
    movies = [] #遍历user_id的每个电影，没评分的做推荐处理
    for item, rating in item_to_rating.items():
        if rating[user_id - 1] == 0:
            final_score = 0
            for similar_item in similar_items[item]:
                final_score += user_to_rating[user_id][similar_item[0] - 1] * similar_item[1]
            movies.append([item_to_name[item], final_score])
            print(item)
    movies = sorted(movies, reverse=True, key=lambda x: x[1])
    return movies[:topn]


def find_similar_item(item_to_rating):
    item_to_similar_item = {}
    score_memory = {}
    for item_a, ratings_a in item_to_rating.items():
        similar_item = []
        for item_b, ratings_b in item_to_rating.items():
            # 全算比较慢，省去一部分用户
            if item_b == item_a or item_b > 100 or item_a > 100:
                continue
            # ab用户互换不用重新计算cos
            if "%d_%d" % (item_b, item_a) in score_memory:
                similarity = score_memory["%d_%d" % (item_b, item_a)]
            # 相似度计算采取cos距离
            else:
                similarity = cosine_distance(np.array(ratings_a), np.array(ratings_b))
                score_memory["%d_%d" % (item_a, item_b)] = similarity
            similar_item.append([item_b, similarity])
        similar_items = sorted(similar_item, reverse=True, key=lambda x: x[1])
        item_to_similar_item[item_a] = similar_items
    return item_to_similar_item


if __name__ == "__main__":
    user_item_score_data_path = "ml-100k/u.data"
    item_name_data_path = "ml-100k/u.item"
    user_to_rating, item_to_name = build_u2i_matrix(user_item_score_data_path, item_name_data_path,
                                                    False)

    # user-cf
    item_to_rating = build_i2u_matrix(user_to_rating)  # 根据user_to_rating得到item_to_rating
    similar_items = find_similar_item(item_to_rating)  # 得到相似电影

    # 为用户推荐电影
    while True:
        user_id = int(input("输入用户id："))
        recommands = movie_recommand(user_id, user_to_rating, item_to_rating, similar_items, item_to_name)
        for recommand, score in recommands:
            print("%.4f\t%s" % (score, recommand))
