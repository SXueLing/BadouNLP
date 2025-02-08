#词典，每个词后方存储的是其词频，仅为示例，也可自行添加
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


#根据上方词典，对于输入文本，构造一个存储有所有切分方式的信息字典
#学术叫法为有向无环图，DAG（Directed Acyclic Graph），不理解也不用纠结，只当是个专属名词就好

text = "经常有意见分歧"
def all_cut(text, Dict, start=0, path=[]):
        if start == len(text):
                if path:  # 确保路径不为空
                        yield path
                return
        for end in range(start + 1, len(text) + 1):
                word = text[start:end]
                if word in Dict:
                        for result in all_cut(text, Dict, end, path + [word]):
                                if len(''.join(result)) == len(text):  # 确保结果包含所有字
                                        yield result

all_segments = list(all_cut(text, Dict))
result  = []
for segment in all_segments:
        result.append(segment)
print(result)
