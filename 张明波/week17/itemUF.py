import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

class ItemUF:
    def __init__(self, k=20, sim_threshold=0.5):
        """
        初始化ItemUF推荐系统
        
        参数:
            k (int): 选择最相似的k个物品进行推荐
            sim_threshold (float): 相似度阈值，低于此值的物品不考虑
        """
        self.k = k
        self.sim_threshold = sim_threshold
        self.user_item_matrix = None
        self.item_similarity = None
        self.item_index = None
        self.user_index = None
        
    def fit(self, ratings):
        """
        训练模型，计算物品相似度矩阵
        
        参数:
            ratings (list): 评分数据，格式为[(user_id, item_id, rating), ...]
        """
        # 创建用户和物品的索引映射
        users = set([r[0] for r in ratings])
        items = set([r[1] for r in ratings])
        
        self.user_index = {u: i for i, u in enumerate(users)}
        self.item_index = {i: idx for idx, i in enumerate(items)}
        
        # 构建用户-物品评分矩阵
        num_users = len(users)
        num_items = len(items)
        
        self.user_item_matrix = np.zeros((num_users, num_items))
        
        for user, item, rating in ratings:
            user_idx = self.user_index[user]
            item_idx = self.item_index[item]
            self.user_item_matrix[user_idx, item_idx] = rating
        
        # 计算物品相似度矩阵 (使用余弦相似度)
        item_matrix = self.user_item_matrix.T  # 转置为物品-用户矩阵
        self.item_similarity = cosine_similarity(item_matrix)
        
        # 将对角线设为0(物品与自身的相似度不考虑)
        np.fill_diagonal(self.item_similarity, 0)
        
    def recommend(self, user_id, top_n=5):
        """
        为用户生成推荐
        
        参数:
            user_id: 要推荐的用户ID
            top_n: 返回前n个推荐物品
            
        返回:
            list: 推荐的物品ID列表，按评分从高到低排序
        """
        if user_id not in self.user_index:
            return []  # 新用户，无历史数据
            
        user_idx = self.user_index[user_id]
        user_ratings = self.user_item_matrix[user_idx]
        
        # 获取用户已评分的物品索引
        rated_items = [i for i, rating in enumerate(user_ratings) if rating > 0]
        
        # 计算每个未评分物品的预测评分
        scores = defaultdict(float)
        
        for item_idx in range(len(self.item_index)):
            if user_ratings[item_idx] > 0:
                continue  # 跳过已评分的物品
                
            # 找到与当前物品最相似的k个物品
            similar_items = np.argsort(self.item_similarity[item_idx])[::-1]
            similar_items = [i for i in similar_items 
                           if i in rated_items and 
                           self.item_similarity[item_idx][i] > self.sim_threshold][:self.k]
            
            if not similar_items:
                continue
                
            # 计算加权平均评分
            sum_sim = sum_ratings = 0.0
            for sim_item_idx in similar_items:
                similarity = self.item_similarity[item_idx][sim_item_idx]
                sum_sim += similarity
                sum_ratings += similarity * user_ratings[sim_item_idx]
                
            if sum_sim > 0:
                scores[item_idx] = sum_ratings / sum_sim
        
        # 获取评分最高的top_n个物品
        recommended_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # 将物品索引转换回原始ID
        index_to_item = {v: k for k, v in self.item_index.items()}
        recommendations = [(index_to_item[item_idx], score) for item_idx, score in recommended_items]
        
        return recommendations
    
    def predict(self, user_id, item_id):
        """
        预测用户对某个物品的评分
        
        参数:
            user_id: 用户ID
            item_id: 物品ID
            
        返回:
            float: 预测评分，如果无法预测则返回None
        """
        if user_id not in self.user_index or item_id not in self.item_index:
            return None
            
        user_idx = self.user_index[user_id]
        item_idx = self.item_index[item_id]
        
        user_ratings = self.user_item_matrix[user_idx]
        
        # 如果用户已经评分过该物品，直接返回实际评分
        if user_ratings[item_idx] > 0:
            return user_ratings[item_idx]
            
        # 找到与目标物品最相似的k个物品
        similar_items = np.argsort(self.item_similarity[item_idx])[::-1]
        similar_items = [i for i in similar_items 
                       if user_ratings[i] > 0 and 
                       self.item_similarity[item_idx][i] > self.sim_threshold][:self.k]
        
        if not similar_items:
            return None
            
        # 计算加权平均评分
        sum_sim = sum_ratings = 0.0
        for sim_item_idx in similar_items:
            similarity = self.item_similarity[item_idx][sim_item_idx]
            sum_sim += similarity
            sum_ratings += similarity * user_ratings[sim_item_idx]
            
        if sum_sim > 0:
            return sum_ratings / sum_sim
        else:
            return None


# 示例用法
if __name__ == "__main__":
    # 示例评分数据 (用户ID, 物品ID, 评分)
    ratings = [
        (1, 101, 5), (1, 102, 3), (1, 103, 4),
        (2, 101, 4), (2, 102, 2), (2, 104, 3),
        (3, 101, 3), (3, 103, 5), (3, 105, 4),
        (4, 102, 4), (4, 103, 3), (4, 105, 2),
        (5, 101, 5), (5, 104, 4), (5, 105, 3)
    ]
    
    # 创建并训练模型
    recommender = ItemUF(k=3, sim_threshold=0.1)
    recommender.fit(ratings)
    
    # 为用户1生成推荐
    user_id = 1
    recommendations = recommender.recommend(user_id, top_n=3)
    print(f"为用户 {user_id} 推荐的物品: {recommendations}")
    
    # 预测用户1对物品104的评分
    item_id = 104
    predicted_rating = recommender.predict(user_id, item_id)
    print(f"预测用户 {user_id} 对物品 {item_id} 的评分: {predicted_rating}")
