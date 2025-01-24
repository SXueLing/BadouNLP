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

def all_cut(Dict,sentence_len):
    if not sentence_len:
        return [[]]

    word_dict = [dict_keys for dict_keys in Dict.keys() if sentence_len[0] == dict_keys[0]]
    words = []
    for lens in range(len(max(word_dict, key=len)), 0, -1):
        if sentence_len[:lens] in [len_word_dict for len_word_dict in word_dict if len(len_word_dict) == lens]:
            first_words = sentence_len[:lens]
            end_sentence = sentence_len[lens:]
            #words.append(first_words)
            results = all_cut(Dict, end_sentence)
            for result in results:
                words.append([first_words] + result)
    return words
  
print(all_cut(Dict,sentence))
