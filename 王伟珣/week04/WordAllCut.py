
def word_all_cut(sentence, vocab, max_len_dict=None):
    if len(sentence) == 1:
        return [[sentence]]

    all_cut_array = []
    for i in range(1, len(sentence)+1):
        word, remain_word = sentence[:i], sentence[i:]
        if word[0] not in max_len_dict.keys() or max_len_dict[word[0]] < i:
            break
        
        if word in vocab.keys():
            if i == len(sentence):
                all_cut_array.append([word])
            else:
                remain_cuts = word_all_cut(remain_word, vocab, max_len_dict)
                for remain_cut in remain_cuts:
                    all_cut_array.append([word] + remain_cut)
    return all_cut_array


def word_max_len(vocab):
    max_len_dict = {}
    for key in vocab.keys():
        word = key[0]
        if word not in max_len_dict.keys() or max_len_dict[word] < len(key):
            max_len_dict[word] = len(key)
    return max_len_dict


if __name__ == "__main__":
    print("world all cut ...")
    vocab = {
        "经常":0.1,
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
        "分":0.1
    }

    sentence = "经常有意见分歧"
    max_len_dict = word_max_len(vocab)
    all_cuts = word_all_cut(sentence, vocab, max_len_dict)

    print(len(all_cuts))
    for cut in all_cuts:
        print(cut)

