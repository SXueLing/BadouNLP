text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
tokens = text.encode("utf-8")
tokens = list(map(int, tokens)) # 转化为utf-8数组

# 统计二元组出现的次数
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

stats = get_stats(tokens)
print(stats, 'stats')
top_pair = max(stats, key=stats.get) # 20，获取出现次数最多的二元组(101, 32)，出现次数为20次
print(top_pair, 'top_pair')
# print(stats.get((32, 111)), 'stats')
# print(sorted(((y,x) for x,y in stats.items()), reverse=True))

# 新建一个数组，
def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

# print(merge([5, 6, 6, 7, 9, 1], (6, 7), 99)) #[5, 6, 99, 9, 1]
# tokens2 = merge(tokens, top_pair, 256)
# print(tokens2)

vocab_size = 260
num_merges = vocab_size - 256
ids = tokens

for i in range(num_merges):
    stats = get_stats(ids)
    top_pair = max(stats, key=stats.get)
    print(f"merging {top_pair} into a new token {i + 256}")
    ids = merge(ids, top_pair, i + 256)

print(ids)





