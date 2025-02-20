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
sentence = "经常有意见分歧"

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
#动态规划
def all_cut(sentence, Dict):
    n = len(sentence)
    target = [[]for _ in range(n+1)]
    target[0] = [[]]
    for i in range(1,n+1):
        for j in range(0,i):
            word = sentence[j:i]
            if word in Dict:
                for k in target[j]:
                    target[i].append(k+[word]) 
    return target[n]

target = all_cut(sentence, Dict)
for i in target:
    print(i)
