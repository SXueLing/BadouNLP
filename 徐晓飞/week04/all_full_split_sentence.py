# 实现基于词表的全切分
words_set = {
    "北京",
    "北京大学",
    "大学生",
    "前来",
    "来报",
    "大学",
    "报道"
}
sentence = "北京大学生前来报道"


# 北京 大学 生 前来 报道        北京 | 大学 | 生 | 前来 | 报道
# 北京大学 生 前来 报道         北京大学 | 生 | 前来 | 报道
# 北京 大学生 前来 报道         北京 | 大学生 | 前来 | 报道
# 北京 大学 生 前 来报 道       北京 | 大学 | 生 | 前 | 来报 | 道
# 北京大学 生 前 来报 道        北京大学 | 生 | 前 | 来报 | 道
# 北京 大学生 前 来报 道        北京 | 大学生 | 前 | 来报 | 道
def find_max_word_length():
    da = -1
    xiao = 9999
    for word in words_set:
        if len(word) > da:
            da = len(word)
        if len(word) < xiao:
            xiao = len(word)
    return xiao, da


def all_split(max_length, min_length):
    words = set()
    for word_pop in words_set:
        for length in range(min_length, max_length + 1):
            words1 = []
            new_sentences = sentence.split(word_pop)
            for i in range(len(new_sentences)):
                sub_sentence = new_sentences[i]
                if sub_sentence == '' and i != len(new_sentences) - 1:
                    words1.append(word_pop)
                    continue
                words1.extend(split_by_max_word_length(sub_sentence, length, word_pop))
                if i != len(new_sentences) - 1:
                    words1.append(word_pop)
            words.add(" | ".join(words1))
    return words


def split_by_max_word_length(sentence, max_word_length: int, ignore_word: str):
    words = []
    lp = 0
    rp = lp + max_word_length
    sub_sentence = sentence[lp:rp]
    while lp < len(sentence):
        if sub_sentence in words_set and sub_sentence != ignore_word:
            words.append(sub_sentence)
            lp += len(sub_sentence)
            rp = lp + max_word_length
            sub_sentence = sentence[lp:rp]
            continue

        if sub_sentence not in words_set:
            if rp - lp <= 1:
                words.append(sub_sentence)
                lp = lp + 1
                rp = lp + max_word_length
                sub_sentence = sentence[lp:rp]
                continue
            rp -= 1
            sub_sentence = sentence[lp:rp]

    return words


if __name__ == '__main__':
    min_length, max_length = find_max_word_length()
    all_words = all_split(max_length, min_length)
    for i in all_words:
        print(i)
