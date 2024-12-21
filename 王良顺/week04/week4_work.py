#week4作业

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
def all_cut(text,max_len):
    result = []
    def cut_word(start, str, tmp):
        # 若轮最后一个字时,代表完成了一次切分
        if start == len(str):
            # 将此次切分结果累积到结果中
            result.append(tmp[:])
            return
        # 1-4
        for i in range(1, max_len+1):
            # 切分结束位置 1/2/3/4
            cur_end = start + i
            # 字符切分项 = 开始位置:开始位置+1/+2/+3/+4
            key = str[start:cur_end]
            # 若切分项在字典中,并且切分结束位置未达到结尾,则继续递归
            if cur_end <= len(str) and key in list(Dict.keys()):
                # 中间件里添加上此次切分的字符
                tmp.append(key)
                cut_word(cur_end, str, tmp)
                # 此次递归完需要将中间件清空,待下一个此次位置+1处再使用
                tmp.pop()

    # 从0开始递归切分
    cut_word(0, text, [])
    return result

if __name__ == '__main__':
    # 若字典内容很多,查找字符的最长度
    max_len = 0
    for item in Dict.keys():
        if len(item) > max_len:
            max_len = len(item)
            
    print(all_cut(sentence,max_len))
