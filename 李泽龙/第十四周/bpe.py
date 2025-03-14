
#token转换
# 指定要读取的文件路径
file_path = 'E:/nlp_learn/practice/week14/dota2英雄介绍-byRAG/Heroes/矮人直升机.txt'

try:
    # 以只读模式打开文件，并指定编码为 UTF-8
    with open(file_path, 'r', encoding='utf-8') as file:
        # 读取文件的全部内容并赋值给变量 text
        text = file.read()
    #print(text)
except FileNotFoundError:
    print(f"文件 {file_path} 未找到。")
except Exception as e:
    print(f"读取文件时出现错误: {e}")
    
tokens = text.encode("utf-8") # raw bytes
tokens = list(map(int, tokens)) # convert to a list of integers in range 0..255 for convenience

def is_chinese_start(byte):
    return 0xE0 <= byte <= 0xF4
  
def get_stats(ids):
    counts = {}
    i = 0
    while i < len(ids):
        if is_chinese_start(ids[i]):
            # 处理汉字，将整个汉字作为一个单元
            if i + 2 < len(ids) and is_chinese_start(ids[i]) and (0x80 <= ids[i + 1] <= 0xBF) and (
                    0x80 <= ids[i + 2] <= 0xBF):
                char = tuple(ids[i:i + 3])
                counts[char] = counts.get(char, 0) + 1
                i += 3
                continue
        if i < len(ids) - 1:
            pair = (ids[i], ids[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
        i += 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if isinstance(pair, tuple) and len(pair) == 3 and i < len(ids) - 2 and tuple(ids[i:i + 3]) == pair:
            newids.append(idx)
            i += 3
        elif i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

# ---
vocab_size = 276 # the desired final vocabulary size  超参数：预期的最终词表大小，根据实际情况自己设置，大的词表会需要大的embedding层
num_merges = vocab_size - 256
ids = list(tokens) # copy so we don't destroy the original list

merges = {} # (int, int) -> int
for i in range(num_merges):
  stats = get_stats(ids)
  pair = max(stats, key=stats.get)
  idx = 256 + i
  print(f"merging {pair} into a new token {idx}")
  ids = merge(ids, pair, idx)
  merges[pair] = idx

#给出一段编码输出文字转换
vocab = {idx: bytes([idx]) for idx in range(256)}
for pair, idx in merges.items():
    if len(pair) == 2:
        p0, p1 = pair
        vocab[idx] = vocab[p0] + vocab[p1]
    else:
        vocab[idx] = bytes(pair)

def decode(ids):
  # given ids (list of integers), return Python string
  tokens = b"".join(vocab[idx] for idx in ids)
  text = tokens.decode("utf-8", errors="replace")
  return text

print(decode([65, 32, 80, 274, 111, 103, 114, 97, 109, 109, 259, 260, 153, 250, 73, 110, 116, 114, 111, 100, 117, 99, 116, 105, 111, 110, 32, 116, 111, 32, 85, 110, 105, 270, 101,]))

  
  
