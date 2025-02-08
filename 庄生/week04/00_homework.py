# week3作业
import copy
import timeit
# 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {
        "经常": 0.1,
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
        "分": 0.1
    }

# 待切分文本
sentence = "经常有意见分歧"  # 7
"""
经 
    常
        有
        有意
        有意见
    常有
    常有意
经常

"""
diguiCount = 0


def getLastStr(arr, sentence):  # 获取剩余的字符串
    return sentence[getStrLenFromArr(arr):]


def getStrLenFromArr(arr):  # 获取数组中字符串拼接后总长度
    return len(''.join(arr))


def diguiFn(linshiArr, dictMaxLen, targetArr):
    global diguiCount
    diguiCount += 1
    linshiArrLength = len(linshiArr)
    for i in range(linshiArrLength):
        nowArr = linshiArr[i]
        nowlen = getStrLenFromArr(nowArr)
        lastlen = len(sentence) - getStrLenFromArr(nowArr)  # 剩余字符长度
        for subNum in range(min(dictMaxLen, lastlen)):
            subStr = sentence[nowlen:nowlen+subNum+1]  # 本轮要测试的字符串
            if subStr in Dict:
                # print(sentence[nowlen:nowlen+subNum+1], '有在字典里')
                newSubArr = copy.deepcopy(nowArr)
                newSubArr.append(subStr)
                if getStrLenFromArr(newSubArr) == len(sentence):
                    targetArr.append(newSubArr)
                else:
                    linshiArr.append(newSubArr)
    linshiArr=linshiArr[linshiArrLength:]

    if len(linshiArr) != 0:
        diguiFn(linshiArr, dictMaxLen, targetArr)


def initLinshiArr(dictMaxLen, sentence, curLen,linshiArr):
    # linshiArrLength = len(linshiArr)
    for i in range(min(dictMaxLen, len(sentence) - curLen)):
        curStrArr = [sentence[:i + 1]]
        # lastStr = sentence[i + 1:]
        if curStrArr[0] in Dict:
            linshiArr.append(curStrArr)
    # linshiArr = linshiArr[linshiArrLength:]
    # print("初始化linshiArr：", linshiArr)


# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    # 1、先确定窗口长度--Dict中最长的键长度
    dictMaxLen = 0
    for i in Dict:
        dictMaxLen = max(dictMaxLen, len(i))

    sentenceArr = list(sentence)
    targetArr = []
    linshiArr = []
    initLinshiArr(dictMaxLen, sentence, 0, linshiArr)
    diguiFn(linshiArr, dictMaxLen, targetArr)

    return targetArr, len(targetArr)


tarr, tlen = all_cut(sentence, Dict)
print(tarr, '\ntargetArr长度为：', tlen, '\n递归次数：', diguiCount)

# 目标输出;顺序不重要
target = [
    ['经常', '有意见', '分歧'],
    ['经常', '有意见', '分', '歧'],
    ['经常', '有', '意见', '分歧'],
    ['经常', '有', '意见', '分', '歧'],
    ['经常', '有', '意', '见分歧'],
    ['经常', '有', '意', '见', '分歧'],
    ['经常', '有', '意', '见', '分', '歧'],
    ['经', '常', '有意见', '分歧'],
    ['经', '常', '有意见', '分', '歧'],
    ['经', '常', '有', '意见', '分歧'],
    ['经', '常', '有', '意见', '分', '歧'],
    ['经', '常', '有', '意', '见分歧'],
    ['经', '常', '有', '意', '见', '分歧'],
    ['经', '常', '有', '意', '见', '分', '歧']
]

# 深拷贝
# a = [[1,2]]
# a.append(copy.deepcopy(a[0]))
# a[1].append(1)
# print(a)
# a = "经常有"
# print(a[:3])

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped
# 测试执行时长
wrapped = wrapper(all_cut, sentence, Dict)
print('总耗时：', float(timeit.timeit(wrapped,  number=1)))
