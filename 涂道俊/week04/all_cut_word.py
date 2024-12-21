
# 第四周作业：实现基于词表的全切分

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

result = []
# 全切分函数
def all_cut_words(start_index, temp_result):
    # 递归函数
    def go_cut(start_index,temp_result):
        if start_index == len(sentence):
            #print("list",list(temp_result))
            #print("[]",temp_result)
            # 切分完毕，将切分结果添加到结果列表中
            result.append(list(temp_result))
            #result.append(temp_result)
            return
        # 遍历需要切分的字符串
        for i in range(start_index, len(sentence) +1):
                if sentence[start_index:i] in Dict:
                    temp_result.append(sentence[start_index:i])
                    # 递归调用（第一种切分方式确认之后，就可以通过递归的方式实现除了已经切分外的剩下所有字符的全部切分可能）
                    go_cut(i, temp_result)
                    # 回溯：第一种切分结束了，回溯到开始的位置，重新开始另一种新的切分方式，从头开始进行新的切分
                    temp_result.pop()

    go_cut(start_index, temp_result)
    return result

target=all_cut_words(0, [])
for target_result in target:
    print(target_result)



















