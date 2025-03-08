#week3作业 实现基于词表的全切分
#词典
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
def all_cut(sentence, Dict):  # 使用递归实现  
    words = Dict.keys()
    target = []
    if len(sentence)==1: #最后一个字直接返回
       if sentence in words:
          return [[sentence]]
       else:
          return []
    for i in range(len(sentence)):
       word = sentence[0:i+1]  # 这里可以限制一下词的最大长度介绍计算次数 
       if word in words:
          if word==sentence:
            target.append([word])
            break
          for w in all_cut(sentence[i+1:len(sentence)],Dict):
             target.append([word]+ w)
    #TODO
    return target

#目标输出;顺序不重要
# target = [
#     ['经常', '有意见', '分歧'],
#     ['经常', '有意见', '分', '歧'],
#     ['经常', '有', '意见', '分歧'],
#     ['经常', '有', '意见', '分', '歧'],
#     ['经常', '有', '意', '见分歧'],
#     ['经常', '有', '意', '见', '分歧'],
#     ['经常', '有', '意', '见', '分', '歧'],
#     ['经', '常', '有意见', '分歧'],
#     ['经', '常', '有意见', '分', '歧'],
#     ['经', '常', '有', '意见', '分歧'],
#     ['经', '常', '有', '意见', '分', '歧'],
#     ['经', '常', '有', '意', '见分歧'],
#     ['经', '常', '有', '意', '见', '分歧'],
#     ['经', '常', '有', '意', '见', '分', '歧']
# ]

target = all_cut(sentence,Dict=Dict)
# 计算各种切分方式的得分  自定义一个计算规则 : 越短 分越高 相同长度 词频累加越大分越高（这个不一定对 ）
scope_dict = []
for index,item in enumerate(target):
   scope = 0
   for word in item:
      scope+=Dict[word] # 累加词频
   scope+=100-5*len(item)   # 随便写了一个和长度 负相关的 计算方式
   scope_dict.append([index,scope])
print(scope_dict)   
# 根据得分排序输出 
scope_dict = sorted(scope_dict,key = lambda item:item[1],reverse=True)
# 输出排序后的结果
for item in scope_dict:
   print(target[item[0]],item[1])

