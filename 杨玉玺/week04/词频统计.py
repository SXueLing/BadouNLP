#week3作业

#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
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

#待切分文本
sentence = "常经有意见分歧"

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    #TODO
    return target
# 杨玉玺的实现
def all_cut(sentence, Dict):
    # 辅助函数：用于递归地寻找所有可能的切分方式
    def find_all_segmentations(remaining_text, current_path):
        # 如果剩余文本为空，则表示找到一种完整的切分方式
        if not remaining_text:
            all_ways.append(str(current_path))
            return
        # 尝试每一个可能的前缀作为单词，并检查其是否在字典中
        for i in range(1, len(remaining_text) + 1):
            prefix = remaining_text[:i]
            if prefix in Dict:
                # 如果前缀在字典中，则递归地处理剩余文本
                find_all_segmentations(remaining_text[i:], current_path + [prefix])

    all_ways = []  # 存储所有可能的切分方式
    find_all_segmentations(sentence, [])  # 开始递归搜索
    result_target = '\n'.join(all_ways)
    return result_target

#jieba.cut
def calc_dag(sentence):
    DAG = {}   #DAG 空字典，用来存储DAG有向无环图
    N= len(sentence)
    for k in range(N):
        tmplist = []
        i = k
        frag = sentence[k]
        while i <N:
            if frag in Dict:
                tmplist.append(i)
            i += 1
            frag = sentence[k:i +1]
        if not tmplist:
            tmplist.append(k)
        DAG[k] = tmplist
    return DAG



# print(all_cut(sentence, Dict))



#目标输出;顺序不重要
target = [
    ['经常', '有意见', '分歧'],
    ['经常', '有意见', '分', '歧'],
    ['经常', '有', '意见', '分歧'],
    ['经常', '有', '意见', '分', '歧'],
    ['经常', '有', '意', '见分歧'],
    ['经常', '有', '意', '见', '分歧'],
    ['经常', '有', '意', '见', '分', '歧'],
    ['经', '常', '有意见', '分歧'],
    ['经', '常', '有意见', '分', '歧'],
    ['经', '常', '有', '意见', '分歧'],
    ['经', '常', '有', '意见', '分', '歧'],
    ['经', '常', '有', '意', '见分歧'],
    ['经', '常', '有', '意', '见', '分歧'],
    ['经', '常', '有', '意', '见', '分', '歧']
]

