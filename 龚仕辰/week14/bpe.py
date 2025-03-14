import re, collections, sys

text = ""
with open('corpus.txt', encoding="gbk") as f:
    for line in f:
        text += line.strip()

'''
先统计词频
'''
def get_vocab(text):
    
    # 初始化为 0
    vocab = collections.defaultdict(int)
    # 去头去尾再根据空格split
    for word in text.strip().split():
        #note: we use the special token </w> (instead of underscore in the lecture) to denote the end of a word
        # 给list中每个元素增加空格，并在最后增加结束符号，同时统计单词出现次数
        vocab[' '.join(list(word)) + ' </w>'] += 1
    return vocab
print(get_vocab(text))
"""
这个函数遍历词汇表中的所有单词，并计算彼此相邻的一对标记。
EXAMPLE:
    word = 'T h e <\w>'
    这个单词可以两两组合成： [('T', 'h'), ('h', 'e'), ('e', '<\w>')]    
输入:
    vocab: Dict[str, int]  # vocab统计了词语出现的词频  
输出:
    pairs: Dict[Tuple[str, str], int] # 字母对，pairs统计了单词对出现的频率
"""
def get_stats(vocab):
    pairs = collections.defaultdict(int)
    
    for word,freq in vocab.items():
        
        # 遍历每一个word里面的symbol，去凑所有的相邻两个内容
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[(symbols[i],symbols[i+1])] += freq

    return pairs
"""
EXAMPLE:
    word = 'T h e <\w>'
    pair = ('e', '<\w>')
    word_after_merge = 'T h e<\w>'    
输入:
    pair: Tuple[str, str] # 需要合并的字符对
    v_in: Dict[str, int]  # 合并前的vocab   
输出:
    v_out: Dict[str, int] # 合并后的vocab    
注意:
    当合并word 'Th e<\w>'中的字符对 ('h', 'e')时，'Th'和'e<\w>'字符对不能被合并。
"""
def merge_vocab(pair, v_in):
    v_out = {}
    # 把pair拆开，然后用空格合并起来，然后用\把空格转义
    bigram = re.escape(' '.join(pair))
    # 自定义一个正则规则, (?<!\S)h\ e(?!\S) 只有前面、后面不是非空白字符(\S)(意思前后得是没东西的)，才匹配h\ e，这样就可以把Th\ e<\w>排除在外
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    
    for v in v_in:
        # 遍历当前的vocabulary，找到匹配正则的v时，才用合并的pair去替换变成新的pair new，如果没有匹配上，那就保持原来的。
        # 比如pair当前是'h'和'e'，然后遍历vocabulary，找到符合前后都没有东西只有'h\ e'的时候就把他们并在一起变成'he'
        new = p.sub(''.join(pair),v)
        # 然后新的合并的数量就是当前vocabulary里面pair对应的数量
        v_out[new] = v_in[v]
    return v_out
def get_tokens(vocab):
    tokens = collections.defaultdict(int)
    for word, freq in vocab.items():
        word_tokens = word.split()
        for token in word_tokens:
            tokens[token] += freq
    return tokens
vocab = get_vocab(text)
print("Vocab =", vocab)
print('==========')
print('Tokens Before BPE')
tokens = get_tokens(vocab)
print('Tokens: {}'.format(tokens))
print('Number of tokens: {}'.format(len(tokens)))
print('==========')
#about 100 merges we start to see common words
num_merges = 100
for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    
    # vocabulary里面pair出现次数最高的作为最先合并的pair
    best = max(pairs, key=pairs.get)
    
    # 先给他合并了再说，当然这里不操作也没什么，到merge_vocab里面都一样
    new_token = ''.join(best)
    vocab = merge_vocab(best, vocab)
    print('Iter: {}'.format(i))
    print('Best pair: {}'.format(best))
    # add new token to the vocab
    tokens[new_token] = pairs[best]
    # deduct frequency for tokens have been merged
    tokens[best[0]] -= pairs[best]
    tokens[best[1]] -= pairs[best]
    print('Tokens: {}'.format(tokens))
    print('Number of tokens: {}'.format(len(tokens)))
    print('==========')
    print('vocab, ', vocab)

def get_tokens_from_vocab(vocab):
    tokens_frequencies = collections.defaultdict(int)
    vocab_tokenization = {}
    for word, freq in vocab.items():
        # 看vocabulary里面的token频率，相当于上面的code中的tokens去除freq为0的
        word_tokens = word.split()
        for token in word_tokens:
            tokens_frequencies[token] += freq
        # vocab和其对应的tokens
        vocab_tokenization[''.join(word_tokens)] = word_tokens
    return tokens_frequencies, vocab_tokenization

def measure_token_length(token): 
    # 如果token最后四个元素是 < / w >
    if token[-4:] == '</w>':
        # 那就返回除了最后四个之外的长度再加上1(结尾)
        return len(token[:-4]) + 1
    else:
        # 如果这个token里面没有结尾就直接返回当前长度
        return len(token)
    
# 如果vocabulary里面找不到要拆分的词，就根据已经有的token现拆
def tokenize_word(string, sorted_tokens, unknown_token='</u>'):
    
    # base case，没词进来了，那拆的结果就是空的
    if string == '':
        return []
    # 已有的sorted tokens没有了，那就真的没这个词了
    if sorted_tokens == []:
        return [unknown_token] * len(string)

    # 记录拆分结果
    string_tokens = []
    
    # iterate over all tokens to find match
    for i in range(len(sorted_tokens)):
        token = sorted_tokens[i]
        
        # 自定义一个正则，然后要把token里面包含句号的变成[.]
        token_reg = re.escape(token.replace('.', '[.]'))
        
        # 在当前string里面遍历，找到每一个match token的开始和结束位置，比如string=good，然后token是o，输出[(2,2),(3,3)]?
        matched_positions = [(m.start(0), m.end(0)) for m in re.finditer(token_reg, string)]
        # if no match found in the string, go to next token
        if len(matched_positions) == 0:
            continue
        # 因为要拆分这个词，匹配上的token把这个word拆开了，那就要拿到除了match部分之外的substring，所以这里要拿match的start
        substring_end_positions = [matched_position[0] for matched_position in matched_positions]
        substring_start_position = 0
        
        
        # 如果有匹配成功的话，就会进入这个循环
        for substring_end_position in substring_end_positions:
            # slice for sub-word
            substring = string[substring_start_position:substring_end_position]
            # tokenize this sub-word with tokens remaining 接着用substring匹配剩余的sorted token，因为刚就匹配了一个
            string_tokens += tokenize_word(string=substring, sorted_tokens=sorted_tokens[i+1:], unknown_token=unknown_token)
            # 先把sorted token里面匹配上的记下来
            string_tokens += [token]
            substring_start_position = substring_end_position + len(token)
        # tokenize the remaining string 去除前头的substring，去除已经匹配上的，后面还剩下substring_start_pos到结束的一段substring没看
        remaining_substring = string[substring_start_position:]
        # 接着匹配
        string_tokens += tokenize_word(string=remaining_substring, sorted_tokens=sorted_tokens[i+1:], unknown_token=unknown_token)
        break
    else:
        # return list of unknown token if no match is found for the string
        string_tokens = [unknown_token] * len(string)
        
    return string_tokens

"""
该函数生成一个所有token的列表，按其长度（第一键）和频率（第二键）排序。

EXAMPLE:
    token frequency dictionary before sorting: {'natural': 3, 'language':2, 'processing': 4, 'lecture': 4}
    sorted tokens: ['processing', 'language', 'lecture', 'natural']
    
INPUT:
    token_frequencies: Dict[str, int] # Counter for token frequency
    
OUTPUT:
    sorted_token: List[str] # Tokens sorted by length and frequency

"""
def sort_tokens(tokens_frequencies):
    # 对 token_frequencies里面的东西，先进行长度排序，再进行频次，sorted是从低到高所以要reverse
    sorted_tokens_tuple = sorted(tokens_frequencies.items(), key=lambda item:(measure_token_length(item[0]),item[1]), reverse=True)
    
    # 然后只要tokens不要频次
    sorted_tokens = [token for (token, freq) in sorted_tokens_tuple]

    return sorted_tokens

#display the vocab
tokens_frequencies, vocab_tokenization = get_tokens_from_vocab(vocab)

#sort tokens by length and frequency
sorted_tokens = sort_tokens(tokens_frequencies)
print("Tokens =", sorted_tokens, "\n")

#print("vocab tokenization: ", vocab_tokenization)

sentence_1 = 'I like natural language processing!'
sentence_2 = 'I like natural languaaage processing!'
sentence_list = [sentence_1, sentence_2]

for sentence in sentence_list:
    
    print('==========')
    print("Sentence =", sentence)
    
    for word in sentence.split():
        word = word + "</w>"

        print('Tokenizing word: {}...'.format(word))
        if word in vocab_tokenization:
            print(vocab_tokenization[word])
        else:
            print(tokenize_word(string=word, sorted_tokens=sorted_tokens, unknown_token='</u>'))
