# 通过jaccard距离判断两个句子的相似度
def get_score(s1: str, s2: [str]):
    score = -1
    for s in s2:
        current_score = get_score_single(s1, s)
        if current_score > score:
            score = current_score
    return score

def get_score_single(s1: str, s2: str):
    set1 = set(s1)
    set2 = set(s2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

# 遍历slot（map）
def is_slot_filled(slot):
    for key, value in slot.items():
        if slot[key] is None:
            return False
    return True

# score = get_score_single('我买了', '我买了')
# print('score:', score)