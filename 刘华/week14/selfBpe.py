def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus
def text_to_gbk_byte_array(text: str) -> list[int]:
    try:
        byte_data = text.encode('gbk')  # 使用 GBK 编码
        return list(byte_data)
    except UnicodeEncodeError:
        raise ValueError("文本包含无法用 GBK 编码的字符（如繁体字或特殊符号）")

corpus = load_corpus("corpus.txt")
byte_array = text_to_gbk_byte_array(corpus)[0:100000]
print(len(byte_array), 'len(byte_array)')

def findMaxTwo(arr: list[int]):
    tempTick = {}
    for i in range(len(arr) - 1):
        key = f"{arr[i]},{arr[i + 1]}"
        if key in tempTick: tempTick[key] += 1;
        else : tempTick[key] = 1;
    maxKey = max(tempTick, key=tempTick.get)
    return (maxKey, tempTick[maxKey])
def encodeBpe(array: list[int]) -> list[int]:
    tempArr = array[:]
    deCodeList = []
    key = 256
    while(True):
         (maxTow, count) = findMaxTwo(tempArr)
         if (count < 100): break;
         [a, b] = map(int, maxTow.split(','))
         print(a, b, 'maxTow', count)
         innerTempArr = []
         deCodeList.append((key, maxTow, count))
         j = 0
         while j < len(tempArr):

             if tempArr[j] == a and tempArr[j + 1] == b:
                 innerTempArr.append(key)
                 j = j + 2
             else:
                 innerTempArr.append(tempArr[j])
                 j = j + 1
         key = key + 1
         tempArr = innerTempArr[:]
    print(len(tempArr), deCodeList)
    return (tempArr, deCodeList)

def decodeBpe(array: list[int], deCodeList: list[(int, str)]) -> list[int]:
    tempArr = array[:]
    print(len(tempArr), deCodeList)
    while(len(deCodeList)):
        tail = deCodeList.pop()
        innerTemp = []
        for i in range(len(tempArr)):
            if tempArr[i] == tail[0]:
                innerTemp = innerTemp + list(map(int, tail[1].split(',')))
            else: innerTemp.append(tempArr[i]);
        tempArr = innerTemp[:]
        print(len(tempArr), tail)

(tempArr, deCodeList) = encodeBpe(byte_array)
decodeBpe(tempArr, deCodeList)