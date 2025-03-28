# 获取一个255-0的16进制词表，后续加的词 添加到第一位
vocab = [chr(255-i) for i in range(256)]

# 把语料编码成16进制的 字符串
def text_to_hex_escape(text):
    # 将字符串编码为字节对象（默认 UTF-8）
    ids = list(text.encode('utf-8'))
    utfs = [chr(i) for i in ids]
    hex_str = ''.join(utfs)
    return hex_str

# 把语料按照词表进行分词
def cut_corpus(corpus):
  hex_corpus = text_to_hex_escape(corpus)
  print('hex_corpus', hex_corpus)
  result = []
  i = 0
  while i < len(hex_corpus):
      matched = False
      # 按照词表从长到短遍历
      for token in vocab:
          # 如果从当前位置能匹配上词表中的词
          if hex_corpus[i:].startswith(token):
              result.append(token)
              i += len(token)
              matched = True
              break
      # 如果没有匹配到任何词，移动到下一个位置
      if not matched:
          i += 1
  return result

def getMaxPair(ids):
  counts = {}
  for pair in zip(ids[:-1], ids[1:]):
    counts[pair] = counts.get(pair, 0) + 1
  # 获取出现次数最多的pair
  max_pair = max(counts, key=counts.get)
  return max_pair


def bpe(corpus):
    cut_result = cut_corpus(corpus)
    (a, b) = getMaxPair(cut_result)
    vocab.insert(0, f'{a}{b}')
    return cut_result

print('原始词表：',vocab)
corpus = "abababababababababababababababab你好你好"
for i in range(10):
    cut = bpe(corpus)
    print(f'第{i}轮 分词结果：{cut}')

print("最终词表", vocab)

