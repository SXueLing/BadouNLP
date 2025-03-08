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

sentence = "经常有意见分歧"

def backtracking(sentence, start, path, result):
    if start == len(sentence):
        result.append(path[:])
        return 
    for i in range(start, len(sentence)):
        strs = sentence[start : i+1]
        if strs in Dict.keys():
            path.append(strs)
            backtracking(sentence, i+1, path, result)
            path.pop()
        
def all_cut(sentence, Dict):
    result = []
    path = []
    backtracking(sentence, 0, path, result)
    return result

        
all_cut(sentence, Dict)
