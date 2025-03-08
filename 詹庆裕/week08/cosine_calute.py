import numpy as np

def cosine_similarity(input_vector, seq_vectors):
    max_similarity = []
    for q_vector in seq_vectors:
        # 计算点积
        dot_product = np.dot(input_vector, q_vector)
        # 计算L2范数
        norm_a = np.linalg.norm(input_vector)
        norm_b = np.linalg.norm(q_vector)
        # 返回余弦相似度
        similarity = dot_product / (norm_a * norm_b)
        max_similarity.append(similarity)
    max_score = max(max_similarity)
    index = max_similarity.index(max_score)
    return index, max_score
