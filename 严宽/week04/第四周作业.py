from typing import List

Dict = {"经常":0.1,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}


def all_cut(sentence, Dict):
    return partition(sentence,Dict)


def partition(sentence,Dict):
    '''
    递归用于纵向遍历
    for循环用于横向遍历
    当切割线迭代至字符串末尾，说明找到一种方法
    类似组合问题，为了不重复切割同一位置，需要start_index来做标记下一轮递归的起始位置(切割线)
    '''
    result = []
    backtracking(sentence, 0, [], result,Dict)
    return result


def backtracking(sentence, start_index, path, result,Dict):
    # Base Case
    if start_index == len(sentence):
        result.append(path[:])
        return

    # 单层递归逻辑
    for i in range(start_index, len(sentence)):
        # 判断被截取的这一段子串([start_index, i])是否在词表中
        if is_indict(sentence, start_index, i,Dict):
            path.append(sentence[start_index: i + 1])
            backtracking(sentence, i + 1, path, result,Dict)  # 递归纵向遍历：从下一处进行切割，判断其是否在词表中
            path.pop()  # 回溯


def is_indict(sentence, start, end,Dict):  # 判断[start, end]是否在词表中
    if sentence[start: end + 1] in Dict:
        return True
    else:
        return False

sentence = "经常有意见分歧"
path = all_cut(sentence, Dict)
print(path)
