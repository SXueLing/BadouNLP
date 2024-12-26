#week3作业

#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常": 0.1,
        "经": 0.05,
        "有": 0.1,
        "常": 0.001,
        "有意见": 0.1,
        "歧": 0.001,
        "意见": 0.2,
        "分歧": 0.2,
        "见": 0.05,
        "意": 0.05,
        "见分歧": 0.05,
        "分": 0.1}

#待切分文本
sentence = "经常有意见分歧"


#实现全切分函数，输出根据字典能够切分出的所有的切分方式


def cut_times(s1, path, d1, results1):
    if not s1:
        results1.append(path[:])
        return
    for i in range(1, len(s1) + 1):
        word = s1[:i]
        if word in d1:
            path.append(word)
            cut_times(s1[i:], path, d1, results1)
            path.pop()


def all_cut(s2, d2):
    #TODO
    results = []
    cut_times(s2, [], d2, results)
    return results


def main(s, d):
    results = all_cut(s, d)
    for result in results:
        print("/".join(result))


#调用main函数
main(sentence, Dict)
