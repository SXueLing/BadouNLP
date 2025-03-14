

# 将语料替换为utf-8序列
def load_data():
    with open('corpus/bpe_corpus.txt', 'r', encoding='utf-8') as f:
        data = f.read()
        utf8_data = []
        for line in data.split('\n'):
            # print(list(line.encode('utf-8')))
            utf8_data += list(line.encode('utf-8'))
        return utf8_data
        # print(data)

# 统计相邻两个编码出现的频率
def get_stats(tokens):
    # print(tokens)
    stats = {}
    for pair in zip(tokens, tokens[1:]):
        stats[pair] = stats.get(pair, 0) + 1
    sorted_stats_list = sorted(((v, k) for k, v in stats.items()), reverse=True)
    return stats
    # print(sorted(((v, k) for k, v in stats.items()), reverse=True))

# 合并相邻两个编码
def merge_bpe(tokens, pair, idx):
    new_ids = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(tokens[i])
            i += 1
    # print(new_ids)
    return new_ids

# 依照merges进行解码
def decode(merges, ids):
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        # print(p0, p1, vocab[p0], vocab[p1])
        vocab[idx] = vocab[p0] + vocab[p1]
        # print(vocab[idx])
    # print(vocab)
    token = b''.join([vocab[idx] for idx in ids])
    text = token.decode('utf-8', errors='replace')
    # print(text)
    return text

# 根据merges编码
def encode(merges, text):
    tokens = list(text.encode('utf-8'))
    print(tokens)
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        print(stats)
        pair = min(stats, key=lambda p: merges.get(p, float('inf')))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge_bpe(tokens, pair, idx)
    return tokens

def main():
    # 创建merges
    tokens = load_data()
    vocab_size = 276
    num_merges = vocab_size - 256
    merges = {}
    for i in range(num_merges):
        stats = get_stats(tokens)
        top_pair = max(stats, key=stats.get)
        tokens = merge_bpe(tokens, top_pair, 256 + i)
        merges[top_pair] = 256 + i
    # print(merges)
    
    # 译码
    # decoded_tokens = [273, 275, 230, 175, 141, 230, 138, 164, 231, 144, 134, 229, 129, 135, 232, 175, 166, 231, 187, 134, 232, 175, 180, 230, 152, 142, 256, 154, 230, 160, 185, 230, 141, 174, 230, 136, 145, 229, 155, 189, 262, 229, 138, 179, 229, 138, 168, 230, 179, 149, 232, 167, 132, 229, 146, 140, 231, 155, 184, 229, 133, 179, 230, 148, 191, 231, 173, 150, 258, 259, 186, 228, 186, 134, 231, 133, 167, 233, 161, 190, 229, 136, 176, 273, 263, 182, 229, 186, 173, 262, 233, 156, 128, 230, 177, 130, 258, 259, 128, 228, 186, 155, 229, 156, 176, 229, 140, 186, 229, 183, 178, 231, 187, 143, 263, 158, 230, 150, 189, 228, 186, 134, 226, 128, 156, 273, 275, 230, 175, 141, 230, 138, 164, 231, 144, 134, 229, 129, 135, 226, 128, 157, 262, 230, 148, 191, 231, 173, 150, 227, 128, 130, 229, 133, 183, 264, 147, 232, 167, 132, 263, 154, 229, 143, 175, 232, 131, 189, 229, 155, 160, 229, 156, 176, 229, 140, 186, 232, 128, 140, 229, 188, 130, 258, 264, 134, 233, 128, 154, 229, 184, 184, 230, 131, 133, 229, 134, 181, 259, 139, 258, 229, 166, 130, 230, 158, 156, 264, 160, 230, 152, 175, 273, 258, 229, 185, 182, 259, 148, 264, 160, 262, 275, 230, 175, 141, 232, 190, 190, 229, 136, 176, 259, 128, 263, 154, 229, 185, 180, 233, 190, 132, 230, 136, 150, 260, 231, 151, 133, 233, 156, 128, 232, 166, 129, 231, 133, 167, 230, 150, 153, 230, 151, 182, 258, 264, 160, 229, 143, 175, 228, 187, 165, 257, 179, 232, 175, 183, 259, 128, 263, 154, 262, 229, 129, 135, 230, 156, 159, 230, 157, 165, 231, 133, 167, 233, 161, 190, 228, 187, 150, 228, 187, 172, 227, 128, 130]
    # print(decode(merges, decoded_tokens))
    # print(decode(merges,encode(merges, '你好，世界！')))
    print(decode(merges,encode(merges, '独生子女父母护理假详细说明：根据我国的劳动法规和相关政策，为了照顾到独生子女家庭的需求，一些地区已经实施了“独生子女父母护理假”的政策。具体规定可能因地区而异，但通常情况下，如果你是独生子女，并且你的父母达到一定年龄或生病需要照料时，你可以申请一定的假期来照顾他们。')))
        
    

if __name__ == '__main__':
    main()
