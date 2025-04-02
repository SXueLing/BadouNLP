"""
电影打分
实现协同过滤
"""
import numpy as np
import openpyxl


# 读取数据并处理 转化打分矩阵形式
def build_u2i_matrix(user_item_score_data_path, item_name_data_path, write_file = False):
    #获取item id到电影名的对应关系
    item_id_to_item_name = {}
    with open(item_name_data_path, encoding="ISO-8859-1") as f:
        for line in f:
            item_id, item_name = line.split("|")[:2]#取前两个字段
            item_id = int(item_id)
            item_id_to_item_name[item_id] = item_name
        #获取电影数量
    item_num = len(item_id_to_item_name)
    print("电影数量：", item_num)

    #读取打分文件
    user_to_rating = {}
    with open(user_item_score_data_path, encoding="ISO-8859-1") as f:
        for line in f:
            user_id, item_id, score, _ = line.split("\t")
            user_id, item_id, score = int(user_id), int(item_id), int(score)
            if user_id not in user_to_rating:#如果用户不在字典中，则新建一个列表
                user_to_rating[user_id] = [0]*item_num
            #将打分矩阵存入字典
            user_to_rating[user_id][item_id-1] = score
    print("用户数量：", len(user_to_rating))

    #将打分矩阵写入文件
    if not write_file:
        return user_to_rating, item_id_to_item_name

    #写入excel文件
    workbook = openpyxl.Workbook()#创建一个工作簿
    sheet = workbook.create_sheet(index=0)#创建一个工作表
    #第一行 user_id,movie1,movie2...
    header = ["user_id"]+[item_id_to_item_name[i] for i in range(1, item_num+1)]
    sheet.append(header)
    #写入数据
    for i in range(len(user_to_rating)):
        #每行 user_id, rate1, rate2...
        row = [i+1]+user_to_rating[i+1]
        sheet.append(row)
    workbook.save("user_movie_rating.xlsx")
    return user_to_rating, item_id_to_item_name


#根据用户打分计算item相似度
def item_similarity(user_to_rating):
    item_to_vector = {}
    total_user = len(user_to_rating)
    for user_id, rating in user_to_rating.items():
        #print(user_id,rating)
        for item_id, score in enumerate(rating):
            item_id+= 1#item_id从1开始
            if item_id not in item_to_vector:
                item_to_vector[item_id] = [0]*(total_user+1)#+1是因为要计算item_id与item_id的相似度
            item_to_vector[item_id][user_id] = score
    # item_to_vector记录了每个用户打分，数据结构和user_to_rating一样
    return user_similarity(item_to_vector) #计算item相似度复用代码

#根据用户打分计算user相似度
def user_similarity(user_to_rating):
    user_to_similar_user = {}#记录每个用户的相似用户
    score_buffer = {}
    for user_a, ratings_a in user_to_rating.items():
        similar_user = []
        for user_b, ratings_b in user_to_rating.items():
            if user_a == user_b or user_a >100 or user_b >100:
                continue
            if (user_a, user_b) in score_buffer:#如果已经计算过，则直接使用
                similarity = score_buffer[(user_a, user_b)]
            else:
                similarity = cosine_similarity(np.array(ratings_a), np.array(ratings_b))
                score_buffer[(user_a, user_b)] = similarity
            similar_user.append((user_b, similarity))
        similar_user = sorted(similar_user, key=lambda x: x[1], reverse=True)#按相似度排序
        user_to_similar_user[user_a] = similar_user
    return user_to_similar_user


# 计算余弦相似度
def cosine_similarity(vector1, vector2):
    ab = vector1.dot(vector2)
    a_norm = np.sqrt(np.sum(np.square(vector1)))
    b_norm = np.sqrt(np.sum(np.square(vector2)))
    return ab / (a_norm * b_norm)


#基于user的协同过滤
#输入user_id, item_id, 给出预测打分
#有预测打分之后就可以对该用户所有未看过的电影打分，然后给出排序结果
#所以实现打分函数即可
#topn为考虑多少相似的用户
#取前topn相似用户对该电影的打分
def user_cf(user_id, item_id, user_to_similar_user, user_to_rating, topn=10):
    pre_score = 0
    count = 0
    for similar_user_id, similarity in user_to_similar_user[user_id][:topn]:
        #相似用户对这部电影的打分
        rating_by_similar_user = user_to_rating[similar_user_id][item_id-1]#item_id从1开始
        # 分数*用户相似度，作为一种对分数的加权，越相似的用户评分越重要
        pre_score += rating_by_similar_user * similarity
        # 如果这个相似用户没看过，就不计算在总数内
        if rating_by_similar_user != 0:#
            count += 1
    pre_score /= count+1e-10 # 防止除0
    return pre_score

#基于item的协同过滤
#类似user_cf
def item_cf(user_id, item_id, user_to_rating, item_to_similar_item, topn=10):
    pre_score = 0
    count = 0
    #print(item_to_similar_item)
    for similar_item_id, similarity in item_to_similar_item[item_id][:topn]:
        # 用户对相似物品的评分（注意索引从0开始，物品ID从1开始）
        rating_by_similar_item = user_to_rating[user_id][similar_item_id-1]#item_id从1开始
        # 分数*物品相似度，作为一种对分数的加权

        if rating_by_similar_item != 0:  # 只处理用户有评分的相似物品
            pre_score += rating_by_similar_item * similarity
            count+=1#同时累加
    pre_score /= count+1e-10 # 防止除0
    return pre_score

#对于一个用户做完整的item召回
def movie_recommend(user_id, similar_user, similar_item, user_to_rating, item_to_name, topn=10):
    #当前用户还没看过的所有电影id
    unseen_items = [item_id +1 for item_id, rating in enumerate(user_to_rating[user_id]) if rating == 0]
    res =[]
    for item_id in unseen_items:
        #user_cf打分
        #score = user_cf(user_id, item_id, similar_user, user_to_rating)
        #item_cf打分
        score = item_cf(user_id, item_id, user_to_rating, similar_item)
        res.append((item_to_name[item_id], score))
    res = sorted(res, key=lambda x: x[1], reverse=True)
    return res[:topn]



if __name__ == "__main__":
    user_item_score_data_path = "../week17 推荐系统/ml-100k/u.data"
    item_name_data_path = "../week17 推荐系统/ml-100k/u.item"
    user_to_rating, item_to_name = build_u2i_matrix(user_item_score_data_path, item_name_data_path, False)

    similar_item = item_similarity(user_to_rating)
    similar_user = user_similarity(user_to_rating)
    #print(similar_item)
    # 为用户推荐电影
    while True:
        user_id = int(input("输入用户id："))
        recommends = movie_recommend(user_id, similar_user, similar_item, user_to_rating, item_to_name)
        for recommend, score in recommends:
            print("%.4f\t%s" % (score, recommend))