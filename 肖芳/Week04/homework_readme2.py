#week3作业

#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {
    "经常":1,
    "常有":1,
    "有意":1,
    "意见":1,
    "有意见":1,
    "分歧":1,
    "你好":1,
    "好的":1,
}

#待切分文本
sentence = "经常有意见分歧"
window_size = 3


def is_word_prefix(prefix, Dict):
    for key in Dict.keys():
        if key.startswith(prefix):
            return True
    return False


def is_word_suffix(suffix, Dict):
    for key in Dict.keys():
        if key.endswith(suffix):
            return True
    return False

def all_cut(sentence, Dict, can_cut_first=True):
    result = []
    
    # 如果句子为空，返回一个包含空列表的列表
    if not sentence:
        return [[]]
    
    # 从最大窗口开始，逐渐缩小窗口
    for window in range(min(window_size, len(sentence)), 0, -1):
        # 获取当前窗口的词
        current_word = sentence[:window]
        last_word = sentence[window:]

        # 如果不能切分第一个词，则跳过单个字符的窗口
        if not can_cut_first and window == 1:
            continue
        
        # 如果当前窗口的词在词典中
        if len(current_word) == 1 and sentence[window:] == '':
            return [[current_word]]
        elif current_word in Dict:
            # 递归处理剩余的句子
            sub_results = all_cut(sentence[window:], Dict)
            
            # 将当前词插入到所有子结果的开头
            for sub_result in sub_results:
                result.append([current_word] + sub_result)
        elif window == 1 and len(last_word) == 1 and current_word+last_word not in Dict:
            return [[current_word, last_word]]
        elif window == 1 and len(last_word) > 1 and is_word_prefix(last_word[:2], Dict):
            # 当前窗口是单个字符，剩余字符超过2
            # 当前窗口+下一个字符不是一个词后缀
            x = current_word + last_word[0]
            sub_results = all_cut(last_word, Dict, can_cut_first=not is_word_suffix(x, Dict))
            for sub_result in sub_results:
                result.append([current_word] + sub_result)
    return result

result = all_cut(sentence, Dict)
print('result', result)
