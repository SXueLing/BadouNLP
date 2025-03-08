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

#实现全切分，保存切分结果
def all_cut(sentence, Dict, word=[]):
    #不满足切分条件即返回
    if sentence and not any(k.startswith(sentence[0]) for k in Dict.keys()):
        return
    if not sentence:
        #保存结果
        result.append(list(word))
        return
    for key in Dict:
        if sentence.startswith(key):
            word.append(key)
            all_cut(sentence[len(key):], Dict, word) #递归
            word.pop() #回溯

#初始化结果列表
result = []
#切分文本
all_cut(sentence, Dict)
#打印结果
for i in result:
    print(i,sep = '\n')
