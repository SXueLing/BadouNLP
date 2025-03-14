

def load_corpus(corpus_path):
    with open(corpus_path, 'r', encoding='utf-8') as f:
        return f.read()

def get_stats(tokens):
    counts = {}
    for pair in zip(tokens, tokens[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge_token(vocab,tokens,word_max_length):
    newTokens = []
    ids = []
    i=0
    while i<len(tokens):
        for j in range(word_max_length):
            word = tokens[i:i+word_max_length-j]
            if ' '.join(map(str, word)) in vocab:
                newTokens.append(' '.join(map(str, word)))
                ids.append(vocab[' '.join(map(str, word))])
                i=i+word_max_length-j
                break
    return newTokens,ids

def encord(vocab, sentence,max_length):
    tokens = sentence.encode("utf-8")
    tokens = list(map(int, tokens))
    tokens,ids = merge_token(vocab,tokens,max_length)
    return tokens,ids

def decord(vocab, ids):
    id_to_token = {value: key for key, value in vocab.items()}
    r = []
    for id in ids:
        r.extend(list(map(int, id_to_token[id].split(" "))))
    return bytearray(r).decode("utf-8")

def create_vocab(tokens, num_merges):
    # 初始词表
    vocab = {str(i+1): i+1 for i in range(256)}
    vocab['padding'] = 0
    max_length = 1
    tokens,ids = merge_token(vocab,tokens,max_length)
    for i in range(num_merges):
        stats = get_stats(tokens)
        pairs =  [k for k,v in stats.items() if v > 1]
        for pair in pairs:
            key = ' '.join(map(str,pair))
            max_length = max(max_length, len(key.split(' ')))
            vocab[key] = len(vocab)
        tokens,ids = merge_token(vocab,tokens,max_length)
    return vocab,max_length

if __name__ == '__main__':
    num_merges = 5
    corpus_path = 'corpus.txt'
    text = load_corpus(corpus_path)
    tokens = text.encode("utf-8")
    tokens = list(map(int, tokens[:10000]))

    vocab, max_length = create_vocab(tokens,num_merges)


    sentence = "你吃晚饭了吗?"
    tokens,ids =  encord(vocab, sentence, max_length)
    print(tokens)
    print(ids)
    ids = [415, 682, 1304, 1258, 764, 165, 173, 1702, 144, 151, 63]
    sentence =  decord(vocab,ids)
    # 输出 你吃晚饭了吗?
    print(sentence)
    # 测试其他语种
    sentence = "你好 世界 hello world こんにちは 世界"
    tokens,ids =  encord(vocab, sentence, max_length)
    print(tokens)
    print(ids)
    sentence = decord(vocab, ids)
   # 输出 你好 世界 hello world こんにちは 世界
    print(sentence)
