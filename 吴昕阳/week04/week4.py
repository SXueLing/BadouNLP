
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
#这段代码直接来自于jieba分词

def generate_segmentations(input_text):
    n = len(input_text)
    segmentations = []

    def backtrack(start,path):
        if start == n:
            segmentations.append(list(path))
            return
        for end in range(start+1,n+1):
            word = input_text[start:end]
            if word in Dict:
                path.append(word)
                backtrack(end,path)
                path.pop()

    backtrack(0,[])

    segmentation_info = []
    for segmentation in segmentations:
        freq_sum = sum([Dict[word] for word in segmentation])
        segmentation_info.append({
                "segmentation":segmentation,
                "frequency_sum":freq_sum                  
                  })

    return segmentation_info

input_text = "经常有意见分歧"
info = generate_segmentations(input_text)

for i, entry in enumerate(info):
    print(f"切分方式 {i + 1}: {entry['segmentation']}, 总词频: {entry['frequency_sum']}")