import numpy as np
import openpyxl
from collections import defaultdict
import time

def build_u2i_matrix(user_item_score_data_path, item_name_data_path, write_file=True):
    item_id_to_name = {}
    with open(item_name_data_path, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            item_id, item_name = line.split('|')[:2]
            item_id = int(item_id)
            item_id_to_name[item_id] = item_name
        total_movie_count = len(item_id_to_name)
        print("Total movie count:", total_movie_count)
    
    user_to_rating = {}
    with open(user_item_score_data_path, 'r', encoding='ISO-8859-1') as f:
        # count = 0
        for line in f:
            user_id, item_id, score, timestamp = line.split('\t')
            user_id, item_id, score = int(user_id), int(item_id), int(score)
            # count += 1
            # if count > 10:
            #     break
            if user_id not in user_to_rating:
                user_to_rating[user_id] = [0] * total_movie_count
            user_to_rating[user_id][item_id - 1] = score
        print("Total user count:", len(user_to_rating))
    if not write_file:
        return user_to_rating, item_id_to_name
    
def cosine_distance(vector1, vector2):
    ab = vector1.dot(vector2)
    a_norm = np.sqrt(np.sum(np.square(vector1)))
    b_norm = np.sqrt(np.sum(np.square(vector2)))
    return ab / (a_norm * b_norm)

def find_similar_item(user_to_rating):
    item_to_vector = {}
    total_user = len(user_to_rating)
    for user, user_rating in user_to_rating.items():
        for movie_id, score in enumerate(user_rating):
            movie_id += 1
            if movie_id not in item_to_vector:
                item_to_vector[movie_id] = [0] * (total_user + 1)
            item_to_vector[movie_id][user - 1] = score
            #这里要多返回一个item_to_vector，每一部电影不同的用户的评分，用于item_cf计算 
    return find_similar_user(item_to_vector), item_to_vector

def find_similar_user(user_to_rating):
    user_to_similar_user = {}
    score_buffer = {}
    for user_a, ratings_a in user_to_rating.items():
        similar_user = []
        for user_b, ratings_b in user_to_rating.items():
            if user_b == user_a or user_b > 100 or user_a > 100:
                continue
            if "%d_%d"%(user_b, user_a) in score_buffer:
                similarity = score_buffer["%d_%d"%(user_b, user_a)]
            else:
                similarity = cosine_distance(np.array(ratings_a), np.array(ratings_b))
                score_buffer["%d_%d"%(user_a, user_b)] = similarity
            similar_user.append([user_b, similarity])
        similar_user = sorted(similar_user, reverse=True, key=lambda x: x[1])
        user_to_similar_user[user_a] = similar_user
    return user_to_similar_user

def user_cf(user_id, item_id, user_to_similar_user, user_to_rating, topn=10):
    pred_score = 0
    count = 0
    for similar_user, similarity in user_to_similar_user[user_id][:topn]:
        rating_by_similar_user = user_to_rating[similar_user][item_id - 1]
        
        pred_score += rating_by_similar_user * similarity
        
        if rating_by_similar_user != 0:
            count += 1
    pred_score /= count + 1e-5
    return pred_score
    
def item_cf(user_id, item_id, similar_items, item_to_vector, topn=10):
    pred_score = 0
    count = 0
    for similar_item, similarity in similar_items[item_id][:topn]:
        rating_by_similar_item = item_to_vector[similar_item][user_id - 1]
        
        pred_score += rating_by_similar_item * similarity
        if rating_by_similar_item != 0:
            count += 1
    pred_score /= count + 1e-5
    # print(90, pred_score, count)
    return pred_score
        
def movie_recommend(user_id, similar_user, similar_items, user_to_rating, item_to_vector, item_to_name, topn=10):
    unseen_items = [item_id + 1 for item_id, rating in enumerate(user_to_rating[user_id]) if rating == 0]
    # print("Unseen items:", unseen_items)
    # print("similar_items:", similar_items)
    # print("similar_user:", similar_user)
    
    res = []
    for item_id in unseen_items:
        # score = user_cf(user_id, item_id, similar_user, user_to_rating)
        score = item_cf(user_id, item_id, similar_items, item_to_vector)
        res.append([item_to_name[item_id], score])
    
    res = sorted(res, key=lambda x: x[1], reverse=True)
    return res[:topn]
    
if __name__ == '__main__':
    user_item_score_data_path = 'ml-100k/u.data'
    item_name_data_path = 'ml-100k/u.item'
    user_to_rating, item_to_name = build_u2i_matrix(user_item_score_data_path, item_name_data_path, False)
    
    similar_user = find_similar_user(user_to_rating)
    similar_items, item_to_vector = find_similar_item(user_to_rating)
    
    while True:
        user_id = int(input("Enter user id: "))
        recommands = movie_recommend(user_id, similar_user, similar_items, user_to_rating, item_to_vector, item_to_name, 10)
        for recommand, score in recommands:
            print("%.4f\t%s"%(score, recommand))
