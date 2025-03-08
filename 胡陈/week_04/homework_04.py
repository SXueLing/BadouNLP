#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
dict = {"经常":0.1,
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

def cut_word(sentence, dict, path, result):
    
    if not sentence:
        re=" ".join(path)
        print(f"\n切分方式：{re}\n")
        result.append(re)  

    for index in range(1, len(sentence)+1):
        word = sentence[:index]
        if word in dict:
            new_path = path + [word]
            # print(f"\n path+word = {new_path}\n")
            cut_word(sentence[index:], dict, new_path, result)

def main():
    result=[]
    path = []
    cut_word(sentence, dict,path,result)
    print(result)

main()





