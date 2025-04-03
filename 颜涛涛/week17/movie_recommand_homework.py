import openpyxl
import numpy as np
import time
from collections import defaultdict

'''
电影打分数据集
实现协同过滤
'''


# 为了好理解，将数据格式转化成user-item的打分矩阵形式
def build_u2i_matrix(item_user_score_data_path, item_name_data_path, write_file=False):
    # 获取item id到电影名的对应关系
    item_id_to_item_name = {}
    with open(item_name_data_path, encoding="ISO-8859-1") as f:
        for line in f:
            item_id, item_name = line.split("|")[:2]
            item_id = int(item_id)
            item_id_to_item_name[item_id] = item_name
    total_movie_count = len(item_id_to_item_name)
    # 读打分文件
    item_to_rating = {}
    with open(item_user_score_data_path, encoding="ISO-8859-1") as f:
        for line in f:
            user_id, item_id, score, time_stamp = line.split("\t")
            user_id, item_id, score = int(user_id), int(item_id), int(score)
            if item_id not in item_to_rating:
                item_to_rating[item_id] = [0] * 943  # 943个用户 此处可以读取文件来得到
            item_to_rating[item_id][user_id - 1] = score

    if not write_file:
        return item_to_rating, item_id_to_item_name

    # 写入excel便于查看
    # workbook = openpyxl.Workbook()
    # sheet = workbook.create_sheet(index=0)
    # # 第一行：user_id, movie1, movie2...
    # header = ["user_id"] + [item_id_to_item_name[i + 1] for i in range(total_movie_count)]
    # sheet.append(header)
    # for i in range(len(item_to_rating)):
    #     # 每行：user_id, rate1, rate2...
    #     line = [i + 1] + item_to_rating[i + 1]
    #     sheet.append(line)
    # workbook.save("user_movie_rating1.xlsx")
    # return item_to_rating, item_id_to_item_name


# 向量余弦距离
def cosine_distance(vector1, vector2):
    ab = vector1.dot(vector2)
    a_norm = np.sqrt(np.sum(np.square(vector1)))
    b_norm = np.sqrt(np.sum(np.square(vector2)))
    return ab / (a_norm * b_norm)


# 依照user对item的打分判断user之间的相似度
# def find_similar_user(user_to_rating):
#     user_to_similar_user = {}
#     score_buffer = {}
#     for user_a, ratings_a in user_to_rating.items():
#         similar_user = []
#         for user_b, ratings_b in user_to_rating.items():
#             if user_b == user_a or user_b > 100 or user_a > 100:
#                 continue
#             # ab用户互换不用重新计算cos
#             if "%d_%d" % (user_b, user_a) in score_buffer:
#                 similarity = score_buffer["%d_%d" % (user_b, user_a)]
#             else:
#                 similarity = cosine_distance(np.array(ratings_a), np.array(ratings_b))
#                 score_buffer["%d_%d" % (user_a, user_b)] = similarity
#
#             similar_user.append([user_b, similarity])
#         similar_user = sorted(similar_user, reverse=True, key=lambda x: x[1])
#         user_to_similar_user[user_a] = similar_user
#     return user_to_similar_user


# 依照不同的user对item的打分判断item之间的相似度
def find_similar_item(item_to_rating):
    item_to_similar_item = {}
    score_buffer = {}
    for item_a, ratings_a in item_to_rating.items():
        similar_item = []
        for item_b, ratings_b in item_to_rating.items():
            if item_b == item_a or item_b > 100 or item_a > 100:
                continue
            # ab商品互换不用重新计算cos
            if "%d_%d" % (item_b, item_a) in score_buffer:
                similarity = score_buffer["%d_%d" % (item_b, item_a)]
            else:
                similarity = cosine_distance(np.array(ratings_a), np.array(ratings_b))
                score_buffer["%d_%d" % (item_a, item_b)] = similarity

            similar_item.append([item_b, similarity])
        similar_item = sorted(similar_item, reverse=True, key=lambda x: x[1])
        item_to_similar_item[item_a] = similar_item
    return item_to_similar_item


# 基于user的协同过滤
# 输入user_id, item_id, 给出预测打分
# 有预测打分之后就可以对该用户所有未看过的电影打分，然后给出排序结果
# 所以实现打分函数即可
# topn为考虑多少相似的用户
# def user_cf(user_id, item_id, user_to_similar_user, user_to_rating, topn=30):
#     pred_score = 0
#     # 取前topn相似用户对该电影的打分
#     count = 0
#     for similar_user, similarity in user_to_similar_user[user_id][:topn]:
#         rating_by_similiar_user = user_to_rating[similar_user][item_id]
#         pred_score += rating_by_similiar_user * similarity
#         if rating_by_similiar_user != 0:
#             count += 1
#     pred_score /= count + 1e-5
#     return pred_score

# 基于item的协同过滤
# 输入user_id, item_id, 给出预测打分
# 有预测打分之后就可以对该用户所有未看过的电影打分，然后给出排序结果
# 所以实现打分函数即可
# topn为考虑多少相似的用户
def item_cf(user_id, item_id, similar_item, item_to_rating, topn=30):
    pred_score = 0
    # 取前topn相似用户对该电影的打分
    count = 0
    for similar_item, similarity in similar_item[item_id][:topn]:
        rating_by_similiar_item = item_to_rating[similar_item][user_id-1]
        pred_score += rating_by_similiar_item * similarity
        if rating_by_similiar_item != 0:
            count += 1
    pred_score /= count + 1e-5
    return pred_score


# 基于item的协同过滤
# 类似user_cf
# 自己尝试实现
# def item_cf(user_id, item_id, TODO):
#     favorite_items = get_user_favorites()  #TODO  获取这个用户喜欢的前n个电影
#     score = 0
#     for favo_item in favorite_items:
#         sim = get_item_similarity(favo_item, item_id) #TODO  对于两个电影，计算相似度
#         score += sim * get_item_rating_by_user(user_id, item_id)  #TODO  获取已知喜欢的电影得分
#     #TODO


# 对于一个用户做完整的item召回
# def movie_recommand1(user_id, similar_user, user_to_rating, item_to_name, topn=10):
#     # 找到当前用户没打过分的电影
#     unseen_items = [item_id for item_id, rating in enumerate(user_to_rating[user_id]) if rating == 0]
#     res = []
#     for item_id in unseen_items:
#         score = user_cf(user_id, item_id, similar_user, user_to_rating)
#         res.append([item_to_name[item_id], score])
#     res = sorted(res, key=lambda x: x[1], reverse=True)
#     return res[:topn]



def movie_recommand(user_id, similar_item, item_to_rating, item_to_name, topn=10):
    # 找到当前用户没打过分的电影
    unseen_items = []
    for item, rating in item_to_rating.items():
        if rating[user_id - 1] == 0:
            unseen_items.append(item)
    res = []
    for item_id in unseen_items:
        score = item_cf(user_id, item_id, similar_item, item_to_rating)
        res.append([item_to_name[item_id], score])
    res = sorted(res, key=lambda x: x[1], reverse=True)
    return res[:topn]


if __name__ == "__main__":
    item_user_score_data_path = "ml-100k/u.data"
    item_name_data_path = "ml-100k/u.item"
    item_to_rating, item_to_name = build_u2i_matrix(item_user_score_data_path, item_name_data_path, False)

    # item-cf
    # s = time.time()
    similar_item = find_similar_item(item_to_rating)

    # print("相似用户计算完成，耗时：", time.time() - s)
    # while True:
    #     user_id = int(input("输入用户id："))
    #     item_id = int(input("输入电影id："))
    #     res = user_cf(user_id, item_id, similar_user, user_to_rating)
    #     print(res)

    # recommands = movie_recommand(10, similar_user, user_to_rating, item_to_name)
    # for recommand, score in recommands:
    #     print("%.4f\t%s"%(score, recommand))

    # 为用户推荐电影
    while True:
        user_id = int(input("输入用户id："))
        recommands = movie_recommand(user_id, similar_item, item_to_rating, item_to_name)
        for recommand, score in recommands:
            print("%.4f\t%s"%(score, recommand))




# 2.7187	Fugitive, The (1993)
# 2.6589	Terminator 2: Judgment Day (1991)
# 2.6182	Apollo 13 (1995)
# 2.5095	Dances with Wolves (1990)
# 2.4885	Blade Runner (1982)
# 2.4107	Aladdin (1992)
# 2.4072	Lion King, The (1994)
# 2.3028	Sleepless in Seattle (1993)
# 2.2763	Quiz Show (1994)
# 2.2619	Crow, The (1994)



# 2.4351	Body Snatchers (1993)
# 2.4351	Sword in the Stone, The (1963)
# 2.4351	Christmas Carol, A (1938)
# 2.4297	Once Were Warriors (1994)
# 2.4297	Family Thing, A (1996)
# 2.4297	Kim (1950)
# 2.4297	Marlene Dietrich: Shadow and Light (1996)
# 2.4297	Maybe, Maybe Not (Bewegte Mann, Der) (1994)
# 2.3842	True Crime (1995)
# 2.3832	Bed of Roses (1996)